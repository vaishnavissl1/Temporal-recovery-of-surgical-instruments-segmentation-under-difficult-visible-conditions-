#!/usr/bin/env python3
"""
Analyse one uploaded frame with the released segmenter. Standard library only.

    python -m tools.serve_analyse --release work_v2/release --port 8765

    POST /analyse          body = the image file (PNG/JPEG), Content-Type image/*
    GET  /health

Returns JSON: the segmentation overlay and mask (base64 PNG), the share of the
frame each class covers, and the model's confidence over the instrument pixels.

What it does NOT do, on purpose:
  * It stores nothing. The upload is decoded in memory, scored, and dropped.
  * It does not run the reliability gate. The gate also reads indicators that
    need earlier frames of the same video (temporal overlap, area stability,
    feature similarity); a single uploaded frame has none, so a gate score would
    not mean what it means in the paper. The confidence reported here is the
    max-softmax value, the simple baseline the paper compares the gate against.
  * It says nothing about accuracy: there is no ground truth for an upload.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sitr.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from sitr.runtime import ReleaseBundle  # noqa: E402
from tools.check_stereo import PALETTE  # noqa: E402

MAX_BYTES = 8 * 1024 * 1024
LOCK = threading.Lock()
COLOUR = np.zeros((256, 3), np.uint8)
for _rgb, _i in PALETTE.items():
    COLOUR[_i] = _rgb


def b64png(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


class Analyser:
    def __init__(self, release: str, device: str):
        self.b = ReleaseBundle.load(release, device=device)
        self.h, self.w = int(self.b.manifest["height"]), int(self.b.manifest["width"])
        self.names = self.b.target_names
        self.inst = list(self.b.instrument_ids)

    @torch.no_grad()
    def run(self, data: bytes) -> dict:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("could not read that file as an image")
        oh, ow = arr.shape[:2]
        if min(oh, ow) < 64:
            raise ValueError("image is too small (under 64 pixels on a side)")
        img = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        interp = cv2.INTER_AREA if oh > self.h else cv2.INTER_LINEAR
        small = cv2.resize(img, (self.w, self.h), interpolation=interp)
        x = ((small.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
        t0 = time.perf_counter()
        logits, _ = self.b.model(torch.from_numpy(np.ascontiguousarray(x))[None].to(self.b.device))
        probs = torch.softmax(logits.float() / self.b.temperature, dim=1)[0]
        conf, pred = probs.max(dim=0)
        if self.b.device.type == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        pred = pred.cpu().numpy().astype(np.uint8)
        conf = conf.cpu().numpy()
        inst_mask = np.isin(pred, self.inst)
        colour = COLOUR[pred]
        overlay = (0.55 * small + 0.45 * colour).astype(np.uint8)
        overlay[pred == 0] = small[pred == 0]
        share = np.bincount(pred.ravel(), minlength=len(self.names)) / pred.size
        classes = [dict(id=i, name=self.names[i], share=float(share[i]))
                   for i in range(1, len(self.names)) if share[i] > 0]
        classes.sort(key=lambda c: -c["share"])
        has_inst = bool(inst_mask.any())
        return dict(
            input_size=[int(ow), int(oh)], working_size=[self.w, self.h], model_ms=round(ms, 1),
            overlay=b64png(overlay), mask=b64png(colour), classes=classes,
            instrument_share=float(inst_mask.mean()),
            confidence_instrument=float(conf[inst_mask].mean()) if has_inst else None,
            confidence_all=float(conf.mean()),
            note=("No instrument pixels were predicted in this frame." if not has_inst else
                  "Confidence is the model's own max-softmax value. It is not accuracy: there is no ground truth for an upload."),
        )


def make_handler(an: Analyser, origins: list):
    class H(BaseHTTPRequestHandler):
        def _cors(self):
            o = self.headers.get("Origin", "")
            if "*" in origins or o in origins:
                self.send_header("Access-Control-Allow-Origin", o or "*")
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self.send_response(204); self._cors(); self.end_headers()

        def do_GET(self):
            if self.path.rstrip("/") == "/health":
                self._send(200, dict(ok=True, classes=an.names))
            else:
                self._send(404, dict(error="not found"))

        def do_POST(self):
            if self.path.rstrip("/") != "/analyse":
                return self._send(404, dict(error="not found"))
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = 0
            if n <= 0 or n > MAX_BYTES:
                return self._send(413, dict(error=f"send one image of at most {MAX_BYTES // 2**20} MB"))
            data = self.rfile.read(n)
            try:
                with LOCK:
                    out = an.run(data)
            except ValueError as e:
                return self._send(400, dict(error=str(e)))
            except Exception:   # never echo internals to a visitor
                return self._send(500, dict(error="the model could not process that image"))
            self._send(200, out)

        def log_message(self, fmt, *a):  # no access log of uploads or addresses
            pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=os.path.join(ROOT, "work_v2", "release"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--origins", default="http://localhost:3000,http://127.0.0.1:3000",
                    help="comma-separated allowed browser origins, or *")
    a = ap.parse_args()
    an = Analyser(a.release, a.device)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(an, [o.strip() for o in a.origins.split(",")]))
    print(f"ready on http://{a.host}:{a.port}  device={an.b.device}  uploads are not stored", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

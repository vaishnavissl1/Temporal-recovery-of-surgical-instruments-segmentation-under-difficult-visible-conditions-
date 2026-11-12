#!/usr/bin/env python3
"""
Local analysis server for the website's Analyse page. Standard library plus the project code.

    python -m tools.serve_analyse --device cpu --port 8765

BATCH (the main demo): the page uploads a folder of consecutive frames and the server runs the
released pipeline over them in order (sitr/batch.py), then returns a per-frame table and the
summary numbers.

    POST /batch/new                          -> {"job": id}
    POST /batch/file?job=ID&name=left_frames/frame171.png      (body = the file)
    POST /batch/zip?job=ID                   (body = a zip of the folder)
    POST /batch/run?job=ID                   starts the run, returns at once
    GET  /batch/status?job=ID                -> {state, done, total, message, seconds}
    GET  /batch/result?job=ID                -> the results once state == "done"

SINGLE FRAME (secondary): POST /analyse with the image as the body. Segmentation overlay, class
shares and max-softmax confidence only. It does not run the gate: the gate reads earlier frames of
the same video, and one frame has none.

Privacy: uploaded files are written to a temporary folder that is deleted as soon as the run
finishes (or fails); only the numeric results are kept in memory, for one hour. Nothing is logged.
The server listens on 127.0.0.1 and answers only the origins you list.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sitr.batch import BatchPipeline  # noqa: E402
from sitr.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from sitr.runtime import ReleaseBundle  # noqa: E402
from tools.check_stereo import PALETTE  # noqa: E402

MAX_BYTES = 8 * 1024 * 1024            # one frame
MAX_FILE = 40 * 1024 * 1024            # one file of a batch (a labelled 1280x1024 PNG is about 1 MB)
MAX_ZIP = 600 * 1024 * 1024
MAX_FILES = 1500
RESULT_TTL = 3600
NAME_OK = re.compile(r"^(left_frames|right_frames|labels)/[A-Za-z0-9_.\-]{1,80}\.(png|jpg|jpeg)$|^camera_calibration\.txt$", re.I)
LOCK = threading.Lock()
COLOUR = np.zeros((256, 3), np.uint8)
for _rgb, _i in PALETTE.items():
    COLOUR[_i] = _rgb


def b64png(rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


class Analyser:
    """Single frame."""

    def __init__(self, bundle: ReleaseBundle):
        self.b = bundle
        self.h, self.w = int(bundle.manifest["height"]), int(bundle.manifest["width"])
        self.names = bundle.target_names
        self.inst = list(bundle.instrument_ids)

    @torch.no_grad()
    def run(self, data: bytes) -> dict:
        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError("could not read that file as an image")
        oh, ow = arr.shape[:2]
        if min(oh, ow) < 64:
            raise ValueError("image is too small (under 64 pixels on a side)")
        img = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(img, (self.w, self.h), interpolation=cv2.INTER_AREA if oh > self.h else cv2.INTER_LINEAR)
        x = ((small.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
        t0 = time.perf_counter()
        logits, _ = self.b.model(torch.from_numpy(np.ascontiguousarray(x))[None].to(self.b.device))
        probs = torch.softmax(logits.float() / self.b.temperature, dim=1)[0]
        conf, pred = probs.max(dim=0)
        ms = (time.perf_counter() - t0) * 1000
        pred = pred.cpu().numpy().astype(np.uint8)
        conf = conf.cpu().numpy()
        inst_mask = np.isin(pred, self.inst)
        colour = COLOUR[pred]
        overlay = (0.55 * small + 0.45 * colour).astype(np.uint8)
        overlay[pred == 0] = small[pred == 0]
        share = np.bincount(pred.ravel(), minlength=len(self.names)) / pred.size
        classes = sorted([dict(id=i, name=self.names[i], share=float(share[i])) for i in range(1, len(self.names)) if share[i] > 0],
                         key=lambda c: -c["share"])
        has_inst = bool(inst_mask.any())
        return dict(input_size=[int(ow), int(oh)], working_size=[self.w, self.h], model_ms=round(ms, 1),
                    overlay=b64png(overlay), mask=b64png(colour), classes=classes,
                    instrument_share=float(inst_mask.mean()),
                    confidence_instrument=float(conf[inst_mask].mean()) if has_inst else None,
                    confidence_all=float(conf.mean()),
                    note=("No instrument pixels were predicted in this frame." if not has_inst else
                          "Confidence is the model's own max-softmax value. It is not accuracy: there is no ground truth for an upload."))


class Jobs:
    def __init__(self, pipeline: BatchPipeline):
        self.bp = pipeline
        self.jobs = {}
        self.lock = threading.Lock()
        self.running = None

    def _get(self, jid):
        j = self.jobs.get(jid)
        if j is None:
            raise KeyError("unknown job")
        return j

    def purge(self):
        now = time.time()
        for jid in [k for k, j in self.jobs.items() if now - j["touched"] > RESULT_TTL and j["state"] != "running"]:
            self.drop(jid)

    def drop(self, jid):
        j = self.jobs.pop(jid, None)
        if j:
            shutil.rmtree(j["dir"], ignore_errors=True)

    def new(self):
        self.purge()
        if len(self.jobs) >= 4:
            raise RuntimeError("too many open jobs; try again shortly")
        jid = uuid.uuid4().hex[:12]
        d = tempfile.mkdtemp(prefix="sitr_batch_")
        self.jobs[jid] = dict(dir=d, state="uploading", done=0, total=0, message="", files=0, bytes=0,
                              touched=time.time(), started=None, result=None, error=None)
        return jid

    def add_file(self, jid, name, data):
        j = self._get(jid)
        if j["state"] != "uploading":
            raise ValueError("this job has already started")
        if not NAME_OK.match(name):
            raise ValueError(f"not an expected file: {name!r}")
        if j["files"] >= MAX_FILES:
            raise ValueError("too many files")
        dest = os.path.join(j["dir"], *name.split("/"))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        j["files"] += 1; j["bytes"] += len(data); j["touched"] = time.time()

    def add_zip(self, jid, data):
        j = self._get(jid)
        try:
            z = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            raise ValueError("that is not a zip file")
        n = 0
        for info in z.infolist():
            if info.is_dir() or info.file_size > MAX_FILE:
                continue
            parts = [p for p in info.filename.replace("\\", "/").split("/") if p and not p.startswith(".") and p != "__MACOSX"]
            if parts and parts[-1] == "camera_calibration.txt":
                name = "camera_calibration.txt"
            elif len(parts) >= 2 and parts[-2] in ("left_frames", "right_frames", "labels"):
                name = f"{parts[-2]}/{parts[-1]}"
            else:
                continue
            if NAME_OK.match(name):
                self.add_file(jid, name, z.read(info))
                n += 1
        if n == 0:
            raise ValueError("the zip holds no left_frames/, right_frames/ or labels/ files")

    def start(self, jid):
        j = self._get(jid)
        if j["state"] != "uploading":
            raise ValueError("already started")
        if not os.path.isdir(os.path.join(j["dir"], "left_frames")):
            raise ValueError("left_frames/ was not uploaded")
        with self.lock:
            if self.running:
                raise RuntimeError("another analysis is running; wait for it to finish")
            self.running = jid
        j.update(state="running", started=time.time(), message="Starting")
        threading.Thread(target=self._work, args=(jid,), daemon=True).start()

    def _work(self, jid):
        j = self.jobs[jid]

        def progress(done, total, msg):
            j.update(done=done, total=total, message=msg, touched=time.time())
        try:
            res = self.bp.run(j["dir"], progress)
            j.update(result=res, state="done", message="Finished", done=res["n_frames"], total=res["n_frames"])
        except ValueError as e:
            j.update(state="error", error=str(e))
        except Exception as e:     # never echo internals to a visitor
            j.update(state="error", error="the pipeline could not process these files")
            print("batch error:", type(e).__name__, e, file=sys.stderr, flush=True)
        finally:
            shutil.rmtree(j["dir"], ignore_errors=True)     # the upload is deleted as soon as the run ends
            j["touched"] = time.time()
            with self.lock:
                self.running = None

    def status(self, jid):
        j = self._get(jid)
        return dict(state=j["state"], done=j["done"], total=j["total"], message=j["message"], files=j["files"],
                    bytes=j["bytes"], error=j["error"], seconds=round(time.time() - j["started"], 1) if j["started"] else 0)


def make_handler(an: Analyser, jobs: Jobs, origins: list):
    class H(BaseHTTPRequestHandler):
        def _cors(self):
            o = self.headers.get("Origin", "")
            if "*" in origins or o in origins:
                self.send_header("Access-Control-Allow-Origin", o or "*")
                self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Private-Network", "true")

        def _send(self, code, obj):
            body = json.dumps(obj, default=float).encode()
            self.send_response(code)
            self._cors()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _body(self, limit):
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                n = 0
            if n <= 0 or n > limit:
                raise OverflowError(f"send a body of at most {limit // 2**20} MB")
            return self.rfile.read(n)

        def do_OPTIONS(self):
            self.send_response(204); self._cors(); self.end_headers()

        def do_GET(self):
            u = urlparse(self.path); q = parse_qs(u.query); path = u.path.rstrip("/")
            try:
                if path == "/health":
                    return self._send(200, dict(ok=True, batch=True, device=str(an.b.device)))
                if path == "/batch/status":
                    return self._send(200, jobs.status(q["job"][0]))
                if path == "/batch/result":
                    j = jobs._get(q["job"][0])
                    if j["state"] != "done":
                        return self._send(409, dict(error="not finished", state=j["state"]))
                    return self._send(200, j["result"])
                return self._send(404, dict(error="not found"))
            except KeyError:
                return self._send(404, dict(error="unknown job"))

        def do_POST(self):
            u = urlparse(self.path); q = parse_qs(u.query); path = u.path.rstrip("/")
            try:
                if path == "/analyse":
                    data = self._body(MAX_BYTES)
                    with LOCK:
                        return self._send(200, an.run(data))
                if path == "/batch/new":
                    return self._send(200, dict(job=jobs.new()))
                if path == "/batch/file":
                    jobs.add_file(q["job"][0], q["name"][0], self._body(MAX_FILE))
                    return self._send(200, dict(ok=True))
                if path == "/batch/zip":
                    jobs.add_zip(q["job"][0], self._body(MAX_ZIP))
                    return self._send(200, dict(ok=True))
                if path == "/batch/run":
                    jobs.start(q["job"][0])
                    return self._send(202, dict(ok=True))
                return self._send(404, dict(error="not found"))
            except OverflowError as e:
                return self._send(413, dict(error=str(e)))
            except KeyError:
                return self._send(404, dict(error="unknown job"))
            except (ValueError, RuntimeError) as e:
                return self._send(400 if isinstance(e, ValueError) else 429, dict(error=str(e)))
            except Exception:
                return self._send(500, dict(error="the server could not process that request"))

        def log_message(self, fmt, *a):  # no access log of uploads or addresses
            pass
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default=os.path.join(ROOT, "work_v2", "release"))
    ap.add_argument("--actions", default=os.path.join(ROOT, "work_v2", "actions"),
                    help="the out-of-fold action data the selector is fitted on")
    ap.add_argument("--sgbm-json", default=os.path.join(ROOT, "results", "task2", "stereo_check.json"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--origins", default="http://localhost:3000,http://127.0.0.1:3000",
                    help="comma-separated allowed browser origins, or *")
    a = ap.parse_args()
    t0 = time.time()
    print("loading the released models and fitting the selector ...", flush=True)
    bundle = ReleaseBundle.load(a.release, device=a.device)
    bp = BatchPipeline(a.release, a.actions, a.sgbm_json, device=a.device, bundle=bundle)
    ThreadingHTTPServer.request_queue_size = 256         # the default of 5 drops connections when a folder uploads file after file
    ThreadingHTTPServer.allow_reuse_address = False     # on Windows, reuse lets a second copy silently share the port
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(Analyser(bundle), Jobs(bp), [o.strip() for o in a.origins.split(",")]))
    print(f"ready in {time.time() - t0:.0f} s on http://{a.host}:{a.port}  device={bundle.device}  uploads are deleted when a run ends", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

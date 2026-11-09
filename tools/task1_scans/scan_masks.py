"""Scan every label PNG: unique colours, alpha values, per-class pixel counts, per-class frame counts.
Repairs override originals for training seq_1/seq_4. Original-vs-repair colour stats are saved too."""
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np, cv2

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
junk = lambda p: "__MACOSX" in p.parts or p.name.startswith("._") or p.name == ".DS_Store"

def pack(rgb):
    return (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2].astype(np.uint32)

def work(job):
    key, path = job
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return key, None
    ch = im.shape[2] if im.ndim == 3 else 1
    rgb = im[..., [2, 1, 0]]  # BGR -> RGB
    alpha = np.unique(im[..., 3]).tolist() if ch == 4 else None
    u, c = np.unique(pack(rgb), return_counts=True)
    return key, dict(shape=list(im.shape), dtype=str(im.dtype), alpha=alpha, colors={int(a): int(b) for a, b in zip(u, c)})

def jobs():
    J = []
    for rel in sorted(ROOT.iterdir()):
        if rel.name in ("repairs", "test_data") or not rel.is_dir():
            continue
        for s in (rel / rel.name).iterdir():
            if s.is_dir() and s.name.startswith("seq_"):
                for f in sorted((s / "labels").iterdir()):
                    if f.suffix == ".png" and not junk(f):
                        J.append((f"train/{s.name}/{f.name}", str(f)))
    for d in (ROOT / "test_data" / "test_data").iterdir():
        if d.is_dir():
            for s in d.iterdir():
                if s.is_dir() and s.name.startswith("seq_"):
                    for f in sorted((s / "labels").iterdir()):
                        if f.suffix == ".png" and not junk(f):
                            J.append((f"test/{s.name}/{f.name}", str(f)))
    for f in sorted((ROOT / "repairs" / "repairs").iterdir()):
        if f.suffix == ".png" and not junk(f):
            J.append((f"repair/{f.name}", str(f)))
    return J

if __name__ == "__main__":
    J = jobs()
    print("jobs:", len(J), flush=True)
    with Pool(6) as p:
        res = dict(p.imap_unordered(work, J, chunksize=8))
    bad = [k for k, v in res.items() if v is None]
    print("unreadable:", bad)
    OUT.write_text(json.dumps(res))
    print("saved", OUT)

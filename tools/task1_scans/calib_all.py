import sys, re, hashlib
from pathlib import Path
import numpy as np, cv2
root = Path(sys.argv[1])

def parse(p):
    d = {}
    KNOWN = {"Width","Height","Camera-0-F","Camera-0-C","Camera-0-Alpha","Camera-0-K","Camera-1-F","Camera-1-C","Camera-1-Alpha","Camera-1-K","Extrinsic-Omega","Extrinsic-T"}
    for line in p.read_text(errors="replace").splitlines():
        line = line.split("//")[0].strip()
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip()
        if k not in KNOWN:
            continue
        try:
            d[k] = [float(x) for x in v.split()]
        except ValueError:
            pass
    return d

files = []
for rel in sorted(root.iterdir()):
    if rel.name in ("repairs", "test_data") or not rel.is_dir():
        continue
    for s in (rel / rel.name).iterdir():
        if s.is_dir() and s.name.startswith("seq_"):
            files.append(("train/" + s.name, s / "camera_calibration.txt"))
for d in (root / "test_data/test_data").iterdir():
    if d.is_dir():
        for s in d.iterdir():
            if s.is_dir() and s.name.startswith("seq_"):
                files.append(("test/" + s.name, s / "camera_calibration.txt"))
files.sort(key=lambda t: (t[0].split("/")[0], int(t[0].split("_")[1])))

ALL = ["Width","Height","Camera-0-F","Camera-0-C","Camera-0-Alpha","Camera-0-K","Camera-1-F","Camera-1-C","Camera-1-Alpha","Camera-1-K","Extrinsic-Omega","Extrinsic-T"]
print("seq | fx0 | cx0 cy0 | fx1 | cx1 cy1 | |T| mm | Tx Ty Tz | rot deg | Ty/Tz share | missing | md5")
hashes = {}
for name, p in files:
    d = parse(p)
    missing = [k for k in ALL if k not in d]
    om = np.array(d["Extrinsic-Omega"]); T = np.array(d["Extrinsic-T"])
    rot = np.degrees(np.linalg.norm(om))
    h = hashlib.md5(p.read_bytes()).hexdigest()[:8]
    hashes.setdefault(h, []).append(name)
    print(f"{name:10s} | {d['Camera-0-F'][0]:.1f} | {d['Camera-0-C'][0]:.1f} {d['Camera-0-C'][1]:.1f} | {d['Camera-1-F'][0]:.1f} | {d['Camera-1-C'][0]:.1f} {d['Camera-1-C'][1]:.1f} | {np.linalg.norm(T):.3f} | {T[0]:.3f} {T[1]:.3f} {T[2]:.3f} | {rot:.3f} | {abs(T[0])/np.linalg.norm(T):.3f} | {missing} | {h} | W,H={d.get('Width')},{d.get('Height')}")
print("\nidentical files (md5 groups with >1):", {h: v for h, v in hashes.items() if len(v) > 1})
print("distinct calibration files:", len(hashes), "of", len(files))

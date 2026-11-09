import os, sys, json
from pathlib import Path
from PIL import Image

root = Path(sys.argv[1])
junk = lambda p: "__MACOSX" in p.parts or p.name.startswith("._")

def seqdirs():
    out = []
    for rel in sorted(root.iterdir()):
        if not rel.is_dir() or rel.name in ("repairs", "test_data"):
            continue
        inner = rel / rel.name
        cands = [p for p in inner.iterdir() if p.is_dir() and p.name.startswith("seq_")]
        for s in sorted(cands, key=lambda p: int(p.name.split("_")[1])):
            out.append(("train", rel.name, s))
    td = root / "test_data" / "test_data"
    for d in sorted(td.iterdir()):
        if d.is_dir():
            for s in d.iterdir():
                if s.is_dir() and s.name.startswith("seq_"):
                    out.append(("test", d.name, s))
    return out

print("split | release | seq | left | right | labels | calib | names_match")
for split, rel, s in seqdirs():
    sub = {}
    for k in ("left_frames", "right_frames", "labels"):
        d = s / k
        sub[k] = sorted(p.name for p in d.iterdir() if p.is_file() and not junk(p)) if d.is_dir() else None
    cal = (s / "camera_calibration.txt").exists()
    names = None
    if all(sub.values()):
        names = (set(sub["left_frames"]) == set(sub["right_frames"]), set(sub["left_frames"]) == set(sub["labels"]))
    print(split, "|", rel[:32], "|", s.name, "|", *(len(sub[k]) if sub[k] is not None else "MISSING" for k in sub), "|", cal, "|", names)
    others = [p.name for p in s.iterdir() if p.name not in ("left_frames", "right_frames", "labels", "camera_calibration.txt")]
    if others:
        print("   extra entries in seq dir:", others)

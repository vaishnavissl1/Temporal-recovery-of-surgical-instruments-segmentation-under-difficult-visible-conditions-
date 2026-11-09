import sys
from pathlib import Path
root = Path(sys.argv[1])
junk = lambda p: "__MACOSX" in p.parts or p.name.startswith("._")
dirs = []
for rel in sorted(root.iterdir()):
    if rel.name in ("repairs", "test_data") or not rel.is_dir():
        continue
    for s in (rel / rel.name).iterdir():
        if s.is_dir() and s.name.startswith("seq_"):
            dirs.append(("train", s))
for d in (root / "test_data" / "test_data").iterdir():
    if d.is_dir():
        for s in d.iterdir():
            if s.is_dir() and s.name.startswith("seq_"):
                dirs.append(("test", s))
def names(d):
    return sorted(p.name for p in d.iterdir() if p.is_file() and not junk(p))
for split, s in sorted(dirs, key=lambda t: (t[0], int(t[1].name.split("_")[1]))):
    L, R, Y = (set(names(s / k)) for k in ("left_frames", "right_frames", "labels"))
    if L != Y or L != R:
        print(split, s.name, "| in labels not left:", sorted(Y - L), "| in left not labels:", sorted(L - Y), "| L!=R:", sorted(L ^ R))
    ln = sorted(L)
    print(split, s.name, "first/last left:", ln[0], ln[-1], "| first/last label:", sorted(Y)[0], sorted(Y)[-1])

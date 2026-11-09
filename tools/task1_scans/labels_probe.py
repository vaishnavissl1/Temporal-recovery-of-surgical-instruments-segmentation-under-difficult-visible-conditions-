import sys, json
from pathlib import Path
import numpy as np, cv2
root = Path(sys.argv[1])
tables = {}
for rel in sorted(root.iterdir()):
    if rel.name in ("repairs", "test_data") or not rel.is_dir():
        continue
    p = rel / rel.name / "labels.json"
    print("==", p.relative_to(root), p.exists())
    if p.exists():
        d = json.load(open(p))
        print("  schema:", type(d).__name__, (list(d.keys()) if isinstance(d, dict) else "list of %d" % len(d)))
        items = d["classes"] if isinstance(d, dict) else d
        for it in items:
            print("  ", it)
        tables[rel.name] = items
td = json.load(open(root / "test_data/test_data/labels.json"))["classes"]
tables["test_data"] = td

# colour -> set of names across all tables
col2names = {}
for k, items in tables.items():
    for it in items:
        c = tuple(it["color"][:3])
        col2names.setdefault(c, set()).add((it["name"], k))
print("\n== colours shared by >1 distinct name")
for c, ns in sorted(col2names.items()):
    names = {n for n, _ in ns}
    if len(names) > 1:
        print(c, sorted(names))
print("\n== names with >1 colour")
name2col = {}
for k, items in tables.items():
    for it in items:
        name2col.setdefault(it["name"], set()).add(tuple(it["color"][:3]))
for n, cs in sorted(name2col.items()):
    print(n, sorted(cs))

# properties of one left, right, label from first seq
s = root / "miccai_challenge_2018_release_1/miccai_challenge_2018_release_1/seq_1"
for sub, f in (("left_frames", "frame000.png"), ("right_frames", "frame000.png"), ("labels", "frame000.png")):
    im = cv2.imread(str(s / sub / f), cv2.IMREAD_UNCHANGED)
    print(sub, im.shape, im.dtype)

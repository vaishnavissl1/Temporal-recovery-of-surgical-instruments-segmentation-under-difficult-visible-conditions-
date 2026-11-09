import json, sys
from collections import defaultdict
import numpy as np
from pathlib import Path
sp = Path(sys.argv[1])
res = json.load(open(sp / "mask_scan.json"))

def unpack(k): return ((k >> 16) & 255, (k >> 8) & 255, k & 255)
NAMES = {0:"background-tissue",1:"instrument-shaft",2:"instrument-clasper",3:"instrument-wrist",4:"kidney-parenchyma",5:"covered-kidney",6:"thread",7:"clamps",8:"suturing-needle",9:"suction-instrument",10:"intestine",11:"ultrasound-probe"}
# labels.json palette (union; intestine == small-intestine share one colour)
PAL = {(0,0,0):0,(0,255,0):1,(0,255,255):2,(125,255,12):3,(255,55,0):4,(24,55,125):5,(187,155,25):6,(0,255,125):7,(255,255,125):8,(123,15,175):9,(124,155,5):10,(12,255,141):11}
# extra colours the OFFICIAL run.py remaps but no labels.json lists
EXTRA = {(170,211,7):10,(169,211,7):10,(255,211,34):10,(33,75,170):5,(204,255,8):10,(204,255,80):10,(39,90,206):5,(123,15,255):9}

print("keys:", len(res))
shapes = defaultdict(int); alphas = defaultdict(int)
for k, v in res.items():
    shapes[(tuple(v["shape"]), v["dtype"])] += 1
    alphas[str(v["alpha"])] += 1
print("mask shapes/dtypes:", dict(shapes)); print("alpha value sets:", dict(alphas))

# which keys
train = {k: v for k, v in res.items() if k.startswith("train/")}
test = {k: v for k, v in res.items() if k.startswith("test/")}
rep = {k: v for k, v in res.items() if k.startswith("repair/")}
print("train masks", len(train), "test masks", len(test), "repair masks", len(rep))

def colour_table(group, title):
    tot = defaultdict(int); frames = defaultdict(int)
    for k, v in group.items():
        for c, n in v["colors"].items():
            tot[int(c)] += n; frames[int(c)] += 1
    print(f"\n== {title}: {len(tot)} distinct colours")
    rows = []
    for c, n in sorted(tot.items(), key=lambda t: -t[1]):
        rgb = unpack(c)
        tag = "palette:%s" % NAMES[PAL[rgb]] if rgb in PAL else ("run.py-extra->%s" % NAMES[EXTRA[rgb]] if rgb in EXTRA else "UNMAPPED")
        rows.append((rgb, n, frames[c], tag))
    for r in rows: print("  ", r)
    unused = [NAMES[i] for rgb, i in PAL.items() if rgb not in {r[0] for r in rows}]
    print("   palette entries never used:", unused)
    return tot, frames

colour_table(train, "TRAIN (originals, before repairs override)")
colour_table(test, "TEST")
colour_table(rep, "REPAIR files")

# per-class stats on training with repairs applied
final = dict(train)
for k in rep:
    name = k.split("/")[1]            # seq_1_frame042.png
    s, f = name.rsplit("_", 1)        # seq_1 , frame042.png
    key = f"train/{s}/{f}"
    assert key in final, key
    final[key] = rep[k]

def class_stats(group, title):
    px = np.zeros(12, dtype=np.int64); fr = np.zeros(12, dtype=np.int64); n = 0; unm = 0
    for k, v in group.items():
        n += 1; seen = set()
        for c, cnt in v["colors"].items():
            rgb = unpack(int(c))
            cid = PAL.get(rgb, EXTRA.get(rgb))
            if cid is None: unm += cnt; continue
            px[cid] += cnt; seen.add(cid)
        for cid in seen: fr[cid] += 1
    tot = px.sum()
    print(f"\n== {title}: frames={n}, unmapped px={unm}")
    print("  id | class | pixel share % | frames containing | % frames")
    for i in range(12):
        print(f"  {i:2d} | {NAMES[i]:20s} | {100*px[i]/tot:8.4f} | {fr[i]:5d} | {100*fr[i]/n:5.1f}")
    return px, fr
class_stats(final, "TRAIN with repairs applied")
class_stats(test, "TEST")

# per-sequence class presence (train) -> which sequences contain each class
seqcls = defaultdict(lambda: np.zeros(12, dtype=np.int64))
for k, v in final.items():
    s = k.split("/")[1]
    for c, cnt in v["colors"].items():
        cid = PAL.get(unpack(int(c)), EXTRA.get(unpack(int(c))))
        if cid is not None: seqcls[s][cid] += 1
print("\n== train frames-with-class per sequence (cols = class ids 0..11)")
for s in sorted(seqcls, key=lambda x: int(x.split("_")[1])):
    print(f"  {s:7s}", " ".join(f"{x:4d}" for x in seqcls[s]))
tseq = defaultdict(lambda: np.zeros(12, dtype=np.int64)); tn = defaultdict(int)
for k, v in test.items():
    s = k.split("/")[1]; tn[s] += 1
    for c, cnt in v["colors"].items():
        cid = PAL.get(unpack(int(c)), EXTRA.get(unpack(int(c))))
        if cid is not None: tseq[s][cid] += 1
print("\n== test frames-with-class per sequence")
for s in sorted(tseq): print(f"  {s:7s} n={tn[s]}", " ".join(f"{x:4d}" for x in tseq[s]))

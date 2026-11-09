import sys
from pathlib import Path
import numpy as np, cv2
ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
PAL = {(0,0,0):0,(0,255,0):1,(0,255,255):2,(125,255,12):3,(255,55,0):4,(24,55,125):5,(187,155,25):6,(0,255,125):7,(255,255,125):8,(123,15,175):9,(124,155,5):10,(12,255,141):11}
def seq_dir(split, name):
    if split == "train":
        for rel in ROOT.iterdir():
            p = rel / rel.name / name
            if rel.name not in ("repairs", "test_data") and p.is_dir(): return p
    else:
        for d in (ROOT / "test_data/test_data").iterdir():
            p = d / name
            if p.is_dir(): return p
PICK = [("train","seq_2",30),("train","seq_5",60),("train","seq_9",90),("train","seq_13",10),("train","seq_14",100),
        ("train","seq_4",135),("test","seq_1",120),("test","seq_2",50),("test","seq_3",200),("test","seq_4",100)]
W, H = 480, 384
for split, sq, fr in PICK:
    s = seq_dir(split, sq); fn = f"frame{fr:03d}.png"
    L = cv2.imread(str(s/"left_frames"/fn)); R = cv2.imread(str(s/"right_frames"/fn))
    mp = s/"labels"/fn
    repaired = False
    rp = ROOT/"repairs/repairs"/f"{sq}_frame{fr:03d}.png"
    if split == "train" and rp.exists(): mp, repaired = rp, True
    M = cv2.imread(str(mp), cv2.IMREAD_UNCHANGED)[..., [2,1,0]]
    col = M.copy()
    ov = (0.55*L[..., ::-1] + 0.45*col).astype(np.uint8)[..., ::-1]
    ov[(M.sum(-1) == 0)] = L[(M.sum(-1) == 0)]
    tiles = [cv2.resize(x, (W, H), interpolation=cv2.INTER_AREA) for x in (L, R)] + [cv2.resize(ov, (W, H), interpolation=cv2.INTER_NEAREST)]
    for t, lab in zip(tiles, ("LEFT", "RIGHT", "LEFT + mask")):
        cv2.putText(t, lab, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    img = np.hstack(tiles)
    cv2.putText(img, f"{split}/{sq} {fn}" + (" (repaired mask)" if repaired else ""), (8, H-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)
    cv2.imwrite(str(OUT / f"{split}_{sq}_{fn}"), img)
    print("wrote", OUT / f"{split}_{sq}_{fn}")

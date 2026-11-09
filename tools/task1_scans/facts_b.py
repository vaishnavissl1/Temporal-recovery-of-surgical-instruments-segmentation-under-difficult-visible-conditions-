import sys, json, random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, cv2
from PIL import Image

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
junk = lambda p: "__MACOSX" in p.parts or p.name.startswith("._") or p.name == ".DS_Store"
PAL = {(0,0,0):0,(0,255,0):1,(0,255,255):2,(125,255,12):3,(255,55,0):4,(24,55,125):5,(187,155,25):6,(0,255,125):7,(255,255,125):8,(123,15,175):9,(124,155,5):10,(12,255,141):11}
INSTR = [1,2,3,6,7,8,9,11]   # repo's is_instrument_name(): shaft,clasper,wrist,thread,clamps,needle,suction,probe

def seqs():
    out = {}
    for rel in sorted(ROOT.iterdir()):
        if rel.name in ("repairs","test_data") or not rel.is_dir(): continue
        for s in (rel/rel.name).iterdir():
            if s.is_dir() and s.name.startswith("seq_"): out[f"train/{s.name}"] = s
    for d in (ROOT/"test_data/test_data").iterdir():
        if d.is_dir():
            for s in d.iterdir():
                if s.is_dir() and s.name.startswith("seq_"): out[f"test/{s.name}"] = s
    return out
S = seqs()
frames = lambda s, k: sorted(p for p in (s/k).iterdir() if p.suffix == ".png" and not junk(p))

# ---- 1. image headers, every left/right frame
hdr = Counter()
for key, s in S.items():
    for k in ("left_frames", "right_frames"):
        for p in frames(s, k):
            with Image.open(p) as im:
                hdr[(k, im.size, im.mode, p.suffix)] += 1
print("== image headers (every left/right frame)")
for k, v in sorted(hdr.items()): print("  ", k, v)

# ---- 2. repairs vs originals
def load_class(path):
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    rgb = im[..., [2,1,0]]
    cid = np.full(rgb.shape[:2], 255, np.uint8)
    for c, i in PAL.items():
        cid[(rgb[...,0]==c[0]) & (rgb[...,1]==c[1]) & (rgb[...,2]==c[2])] = i
    return cid
NAMES = ["bg","shaft","clasper","wrist","kidney","covered-kidney","thread","clamps","needle","suction","intestine","probe"]
print("\n== repairs vs originals")
for f in sorted((ROOT/"repairs/repairs").iterdir()):
    if f.suffix != ".png": continue
    sq, fr = f.stem.rsplit("_", 1)
    orig = S[f"train/{sq}"]/"labels"/(fr+".png")
    a, b = load_class(orig), load_class(f)
    ch = a != b
    pairs = Counter(zip(a[ch].tolist(), b[ch].tolist()))
    top = ", ".join(f"{NAMES[x]}->{NAMES[y]}:{n}" for (x,y),n in pairs.most_common(4))
    print(f"  {f.name}: orig exists={orig.exists()} shape_equal={a.shape==b.shape} changed_px={int(ch.sum())} ({100*ch.mean():.3f}%)  top changes: {top}")

# ---- 3. temporal + image statistics
def miou(a, b, classes):
    v = []
    for c in classes:
        A, B = a == c, b == c
        u = (A | B).sum()
        if u: v.append((A & B).sum() / u)
    return float(np.mean(v)) if v else np.nan
def instr_iou(a, b):
    A, B = np.isin(a, INSTR), np.isin(b, INSTR)
    u = (A | B).sum()
    return float((A & B).sum() / u) if u else np.nan
def gray_small(p):
    im = cv2.imread(str(p)); g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    spec = float(((hsv[...,2] > 235) & (hsv[...,1] < 40)).mean())
    blur = float(cv2.Laplacian(g, cv2.CV_64F).var())
    return cv2.resize(g, (320, 256), interpolation=cv2.INTER_AREA).astype(np.float32), float(g.mean()), spec, blur

res = {}
rng = random.Random(0)
for key in ["train/seq_2", "train/seq_9", "train/seq_14", "test/seq_1", "test/seq_4"]:
    s = S[key]
    L = frames(s, "left_frames"); Y = {p.name: p for p in frames(s, "labels")}
    L = [p for p in L if p.name in Y]
    G, M, bright, spec, blur = [], [], [], [], []
    for p in L:
        g, br, sp, bl = gray_small(p); G.append(g); bright.append(br); spec.append(sp); blur.append(bl)
        M.append(load_class(Y[p.name]))
    n = len(L)
    mad_c = [float(np.abs(G[i+1]-G[i]).mean()) for i in range(n-1)]
    iou_c = [miou(M[i], M[i+1], range(1, 12)) for i in range(n-1)]
    ins_c = [instr_iou(M[i], M[i+1]) for i in range(n-1)]
    pairs = []
    while len(pairs) < n-1:
        i, j = rng.randrange(n), rng.randrange(n)
        if abs(i-j) >= 20: pairs.append((i, j))
    mad_r = [float(np.abs(G[i]-G[j]).mean()) for i, j in pairs]
    iou_r = [miou(M[i], M[j], range(1, 12)) for i, j in pairs]
    ins_r = [instr_iou(M[i], M[j]) for i, j in pairs]
    nm = lambda x: float(np.nanmean(x))
    res[key] = dict(n=n, mad_consec=nm(mad_c), mad_random=nm(mad_r), classIoU_consec=nm(iou_c), classIoU_random=nm(iou_r),
                    instrIoU_consec=nm(ins_c), instrIoU_random=nm(ins_r), instrIoU_consec_median=float(np.nanmedian(ins_c)),
                    frac_consec_instrIoU_below_0_5=float(np.mean(np.array(ins_c)[~np.isnan(ins_c)] < 0.5)),
                    brightness=(float(np.min(bright)), nm(bright), float(np.max(bright))),
                    specular_frac=(float(np.min(spec)), nm(spec), float(np.max(spec))),
                    laplacian_var=(float(np.min(blur)), nm(blur), float(np.max(blur))))
    print(f"\n== {key} (n={n})")
    for k, v in res[key].items(): print(f"   {k}: {v}")
OUT.write_text(json.dumps(res, indent=1))

"""Diagnostic (NOT used to re-tune): does the criterion-optimal SGBM setting hide false matches or
over-smooth thin structures?  Rule declared in advance:
  adopt (block 15, uniq 0) unless (a) its implausible-disparity rate among LR-consistent pixels exceeds that of
  (block 9, uniq 5) by > 1 percentage point, or (b) its LR-pass on thin classes is lower than (block 9, uniq 5)'s.
"""
import json, sys, time
sys.path.insert(0, r"C:\Users\Chandaluri Vaishnavi\Downloads\sitr_endovis2018\sitr_endovis2018")
import cv2, numpy as np
from sitr.dataset import discover_sequences
from sitr.stereo import SGBMParams, VerticalCorrection, calibration_path, disparity_pair, get_rectifier, lr_consistent, right_path
from tools.check_stereo import read, read_mask, INSTRUMENT_IDS
cv2.setNumThreads(8)
root = r"C:\Users\Chandaluri Vaishnavi\Downloads\cvrsfesp\endovis2018_cvdataset"
t2 = json.load(open(r"C:\Users\Chandaluri Vaishnavi\Downloads\sitr_endovis2018\sitr_endovis2018\results\task2\stereo_check.json"))
corrs = {k: VerticalCorrection(**v) for k, v in t2["vertical_corrections"].items()}
seqs = {s.key: s for s in discover_sequences(root, require_labels=True, splits=("train",))}
# every 3rd training sequence frame set: 15 train sequences, frame len//2 (the evaluation frames) + len//4
frames = [(s, len(s) // 2) for s in seqs.values()] + [(s, len(s) // 4) for s in seqs.values()]
THIN = (2, 6, 8)   # clasper, thread, needle
configs = [(5, 5), (9, 5), (15, 0)]
acc = {c: dict(n=0, lr=0, instr_n=0, instr_lr=0, thin_n=0, thin_lr=0, band_n=0, band_lr=0, impl=0, cons=0, sec=0.0) for c in configs}
for s, i in frames:
    rc = get_rectifier(calibration_path(s.frame_dir), 1.0, "opencv", "as_is")
    L, R = read(s.frames[i]), read(right_path(s.frames[i]))
    Lr, Rr = rc.rectify(L, R); Rr = corrs[s.key].apply(Rr)
    gt = rc.rectify_labels(read_mask(s.labels[i]))
    area = rc.valid_left
    instr = np.isin(gt, INSTRUMENT_IDS) & area
    thin = np.isin(gt, THIN) & area
    k = np.ones((9, 9), np.uint8)
    band = (cv2.dilate(instr.astype(np.uint8), k) > 0) & (cv2.erode(instr.astype(np.uint8), k) == 0) & area
    for (b, u) in configs:
        t0 = time.time()
        dl, dr = disparity_pair(Lr, Rr, SGBMParams(block=b, uniqueness=u))
        dt = time.time() - t0
        ok = lr_consistent(dl, dr)
        a = acc[(b, u)]
        a["sec"] += dt; a["n"] += int(area.sum()); a["lr"] += int((ok & area).sum())
        a["instr_n"] += int(instr.sum()); a["instr_lr"] += int((ok & instr).sum())
        a["thin_n"] += int(thin.sum()); a["thin_lr"] += int((ok & thin).sum())
        a["band_n"] += int(band.sum()); a["band_lr"] += int((ok & band).sum())
        v = dl[ok & area]
        a["cons"] += int(v.size); a["impl"] += int(((v < 20) | (v >= 190)).sum())
print("config (block,uniq) | LR% all | LR% instrument | LR% THIN (clasper,thread,needle) | LR% instrument-boundary band | implausible disp % of consistent | s/frame")
for c, a in acc.items():
    print(f"{c}: {100*a['lr']/a['n']:.1f} | {100*a['instr_lr']/max(a['instr_n'],1):.1f} | {100*a['thin_lr']/max(a['thin_n'],1):.1f} (n={a['thin_n']}) | "
          f"{100*a['band_lr']/max(a['band_n'],1):.1f} | {100*a['impl']/max(a['cons'],1):.2f} | {a['sec']/len(frames):.2f}")
big, ref = acc[(15, 0)], acc[(9, 5)]
a_ = 100 * big["impl"] / big["cons"] - 100 * ref["impl"] / ref["cons"]
b_ = 100 * big["thin_lr"] / big["thin_n"] - 100 * ref["thin_lr"] / ref["thin_n"]
print(f"\nrule (a): implausible-rate difference (15,0) - (9,5) = {a_:+.2f} pp (adopt-(9,5) if > +1.00)")
print(f"rule (b): thin-class LR difference (15,0) - (9,5) = {b_:+.2f} pp (adopt-(9,5) if < 0)")
print("DECISION:", "(9,5)" if (a_ > 1.0 or b_ < 0) else "(15,0) criterion-optimal")

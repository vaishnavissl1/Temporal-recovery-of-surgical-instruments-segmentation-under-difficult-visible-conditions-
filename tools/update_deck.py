#!/usr/bin/env python3
"""
Task R.1 -- update the results-bearing slides of the owner's deck IN PLACE (python-pptx): text only, same template, theme, header and layout. Idempotent: it always starts from the
untouched original (`CV Project PPT_original.pptx`, copied on the first run) and writes `CV Project PPT.pptx`, so it can be re-run whenever a result changes.

    python -m tools.update_deck --deck "C:/Users/Chandaluri Vaishnavi/Downloads/cvrsfesp/cvavish"

Every number is read from the result files through tools/paper_numbers.py (the same macros as the paper); none is typed here. Slides touched: 2 (cost and consecutive-frame overlap),
5 (architecture picture), 6 (methodology as built), 7 (two dataset facts), 8 (metric list), 9, 10, 11 (results, findings, conclusion). Slide 1's three placeholders are the owner's and are left alone.
"""
import argparse
import copy
import os
import shutil

from pptx import Presentation
from pptx.util import Emu

import tools.paper_numbers as PN


def values():
    V = {}
    for name, f, path, fmt, scale in PN.SPEC:
        d = PN.J(f)
        try:
            v = PN.dig(d, path) if d is not None else None
            V[name] = "--" if v is None else fmt.format(int(v) if fmt.endswith("d}") else v * scale)
        except Exception:  # noqa: BLE001
            V[name] = "--"
    return V


def walk(shapes):
    for sh in shapes:
        yield sh
        if sh.shape_type == 6:
            yield from walk(sh.shapes)


def find(slide, sid):
    for sh in walk(slide.shapes):
        if sh.shape_id == sid:
            return sh
    raise KeyError(sid)


def set_paras(shape, texts, keep_first=0):
    """Replace the paragraphs of a text frame by `texts` (the first `keep_first` paragraphs, e.g. a heading, are left as they are). Formatting is copied from the paragraph it replaces."""
    tf = shape.text_frame
    paras = list(tf.paragraphs)
    body = paras[keep_first:]
    proto = copy.deepcopy(body[0]._p)
    for p in body:
        p._p.getparent().remove(p._p)
    parent = tf._txBody
    for t in texts:
        el = copy.deepcopy(proto)
        parent.append(el)
    new = list(tf.paragraphs)[keep_first:]
    for p, t in zip(new, texts):
        runs = p.runs
        runs[0].text = t
        for r in runs[1:]:
            r._r.getparent().remove(r._r)


def replace_in(shape, old, new):
    for p in shape.text_frame.paragraphs:
        if old in p.text:
            full = p.text.replace(old, new)
            p.runs[0].text = full
            for r in p.runs[1:]:
                r._r.getparent().remove(r._r)
            return True
    raise KeyError(old)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", required=True, help="folder holding 'CV Project PPT.pptx' and completearch.png")
    a = ap.parse_args()
    cur, orig = os.path.join(a.deck, "CV Project PPT.pptx"), os.path.join(a.deck, "CV Project PPT_original.pptx")
    if not os.path.exists(orig):
        shutil.copy2(cur, orig)
    prs = Presentation(orig)
    V = values()
    S = prs.slides
    base = PN.J("results/taskP/baselines.json")

    # ---- slide 2
    sh = find(S[1], 3)
    set_paras(sh, [p.text for p in sh.text_frame.paragraphs])  # normalise
    ps = sh.text_frame.paragraphs
    for p in ps:
        if p.text.startswith("Correcting every frame"):
            p.runs[0].text = (f"Correcting every frame to protect against those stretches is wasteful, because most frames need nothing, and it is not free: the temporal and refinement corrections "
                              f"applied to every frame cost {V['CostAllTwo']} ms per frame against {V['CostBase']} ms for segmentation with the gate alone.")
            for r in p.runs[1:]:
                r._r.getparent().remove(r._r)
        if p.text.startswith("We measured the consequence"):
            p.runs[0].text = (f"We measured the consequence: consecutive ground-truth instrument masks overlap at {V['KMean']} IoU on average over {V['KPairs']} training pairs, "
                              f"and {V['KBelow']}% of consecutive pairs overlap by less than half.")
            for r in p.runs[1:]:
                r._r.getparent().remove(r._r)

    # ---- slide 5: the architecture picture
    s5 = S[4]
    pic = [x for x in s5.shapes if x.shape_type == 13][0]
    left, top, width, height = pic.left, pic.top, pic.width, pic.height
    pic._element.getparent().remove(pic._element)
    s5.shapes.add_picture(os.path.join(a.deck, "completearch.png"), left, top, width, height)

    # ---- slide 6: methodology as built
    sh = find(S[5], 4)
    set_paras(sh, [
        "Stage 1 - Base segmentation. CAFormer-S18 encoder initialised from surgical self-supervised weights, FPN decoder with a full-resolution branch, 26 M parameters. Trained at 512x640, evaluated at the native 1280x1024.",
        "Stage 2 - Reliability gate. A small network over nine observable indicators (confidence, entropy, margin, mask compactness, fragmentation, temporal consistency, mask stability, two feature similarities). It regresses achieved instrument IoU and needs no ground truth at inference.",
        f"Routing threshold. Set by conformal risk control at a stated budget, fitted out of fold. The budget was met out of fold and not on test ({V['RiskAcceptedTest']} against {V['Alpha']}). A second, separate threshold governs admission to the temporal memory.",
        "Stage 3 - Temporal memory. A causal buffer of the K most recent frames whose reliability cleared the admission threshold. Constant memory, no lookahead.",
        "Stage 4 - Selective correction. Candidate actions are generated only for flagged frames: temporal recovery (A2) and morphological refinement (A3). A selector predicts each action's gain and may abstain. Stereo label transfer (A1) and its fusion with A2 were tested and retired; hallucination repair (A4) is evaluated separately.",
        "Stage 5 - Identity verification. Hungarian assignment between current regions and remembered instruments. As built it returns the mask unchanged, so it cannot alter any score; a ground-truth ceiling shows what a relabelling stage could add.",
        "Non-regression by construction. Frames above the routing threshold are returned unmodified, so the system behaves exactly like its base model on reliable frames.",
        "Cross-view audit (evaluated, not adopted). The right camera is segmented by the same model and the two label maps are compared with row-wise and set statistics that need rectification only. It failed its pre-set test.",
    ])

    # ---- slide 7: two dataset facts
    sh = find(S[6], 4)
    replace_in(sh, "so 11 of 12 classes are scored", f"so {V['ClassesScored']} of the 11 foreground classes are scored")
    replace_in(sh, "7 distinct rig files", f"{V['JRigFiles']} distinct calibration files describing {V['JRigs']} rigs")
    replace_in(sh, "Only this dataset is used.", f"Training and every decision use EndoVis 2018 only. EndoVis 2017 (CC BY 4.0 copy, {V['ZsFrames']} frames, {V['ZsSeqs']} sequences) is a zero-shot, evaluation-only test, scored once and mapped to instrument against background.")

    # ---- slide 8: metric list stays; state the number of scored classes
    sh = find(S[7], 12)
    replace_in(sh, "this is where rare-class and relabelling gains appear.", f"rare-class and relabelling gains appear here ({V['ClassesScored']} classes on test).")
    replace_in(sh, "AUROC - ranking quality of the reliability gate, reported for comparability.", "AUROC - ranking quality of a failure detector, reported for comparability.")
    replace_in(sh, "All 95% intervals use a bootstrap that resamples whole sequences, because frames within a sequence are correlated.", "All 95% intervals resample whole sequences, because frames within a sequence are correlated.")
    replace_in(sh, " Frame-level intervals are not reported.", "")

    # ---- slide 9: results
    sh = find(S[8], 5)
    bl = (f"Same recipe, official mIoU: ours {V['BlOursMiou']}, SegFormer-B0 {V['BlSegMiou']}, U-Net from scratch {V['BlUnetMiou']}. DeepLabV3+ did not train stably (four runs) and is absent.")
    zs = (f"EndoVis 2017, zero-shot, scored once (instrument against background only): mean IoU {V['ZsIou']}, against {V['ZsIouRef']} on 2018 test under the same mapping. No large drop on this narrow task.")
    set_paras(sh, [
        f"Base segmentation reaches {V['BaseMiou']} mIoU and {V['BaseMcIoU']} mcIoU on the four official test sequences under the exact challenge metric, matching the official script to {V['OfficialDiff']}.",
        bl,
        zs,
        f"Failure detection, same frames, risk = 1 - official IoU: one-pass max-softmax has test AURC {V['MsAurcTest']} against {V['GateAurcTest']} for the gate; no interval separates them. The gate leads only on its own label.",
        f"On the {V['FlaggedPct']}% of test frames the gate flags, the refinement is worth {V['FlagAThree']} instrument IoU (oracle {V['FlagOracle']}), but {V['GapGtOnly']} counting only ground-truth classes, as the official metric does: the gain is invented classes removed.",
        f"Selective correction keeps the official mIoU ({V['MiouMSix']}) and costs {V['CostSelNoA']} ms against {V['CostAllTwo']} ms for correcting every frame. Harmful: {V['SelNoAHarmAll']}% of all frames, {V['SelNoAHarmCorr']}% of corrected ones, against {V['FourHarmAll']}% and {V['FourHarmCorr']}% for blind fusion.",
        f"{V['HallFrames']}% of test frames contain a class the model invents; the official metric cannot see it.",
    ])

    # ---- slide 10: key findings (the negatives stay)
    sh = find(S[9], 5)
    set_paras(sh, [
        f"Temporal recovery is limited by the data, not merely weak: consecutive ground-truth masks overlap at {V['KMean']} IoU, and the temporal action exists for {V['ATwoAvail']} of {V['OofRouted']} flagged training frames.",
        f"Motion compensation is not the cause. Pasting, optical flow and region matching leave {V['UnchangedIdOof']} to {V['UnchangedRgTest']}% of flagged frames unchanged, and region matching is lower ({V['AlignRgDiff']}).",
        f"Where defined, the stereo axis carries more mask information than the temporal axis: a round trip keeps {V['GStereo']} of the mask against {V['GFlow']} for the best temporal aligner, on {V['GValid']}% of instrument pixels.",
        f"Cross-view indicators did not clear the pre-set test: AUROC gain {V['DGain']}, interval {V['DLo']} to {V['DHi']}. Reported as a failure of the stated criterion; the AURC change was not the test.",
        f"The gate's ranking is usable (AUROC {V['GateAurocOwnTest']} on its own label, test) but its scores are optimistic: mean {V['GateScoreTest']} against {V['GateAchTest']} achieved, as the missed risk budget ({V['RiskAcceptedTest']} against {V['Alpha']}) suggests.",
        f"Negatives: disparity indicators and per-pixel occlusion ({V['HOcc']}) at chance; TTA ({V['ETtaPts']} points) and a four-model ensemble ({V['EEnsPts']}) do not beat the single model; the identity stage returns its input, ceiling {V['QMcPts']} mcIoU points.",
    ])

    # ---- slide 11: conclusion
    sh = find(S[10], 5)
    set_paras(sh, [
        f"The segmenter scores {V['BaseMiou']} mIoU and no correction changes that: the gain on flagged frames is mostly invented classes removed, which the official metric cannot see.",
        "On the official-IoU risk, simple confidence scores detect failures as well as the gate.",
        f"Selectivity is the defensible gain: {V['CostSelNoA']} ms against {V['CostAllTwo']} ms for correcting every frame, with reliable frames untouched.",
        f"Under one recipe the segmenter leads SegFormer-B0 and a from-scratch U-Net; scored once on EndoVis 2017 with no adaptation it separates instrument from background at {V['ZsIou']} mean IoU.",
        f"We measure what had been assumed: consecutive masks overlap at {V['KMean']} IoU; two definition choices move mIoU by over ten points; harmful corrections are reported with both denominators.",
        "Limitations: one training dataset, four test sequences, one run per model, DeepLabV3+ baseline absent, conformal budget and 40 ms target not met, needle at zero IoU. Future work: a relabelling identity stage, needle training, more sequences and datasets.",
    ])
    prs.save(cur)
    print("deck updated:", cur)


if __name__ == "__main__":
    main()

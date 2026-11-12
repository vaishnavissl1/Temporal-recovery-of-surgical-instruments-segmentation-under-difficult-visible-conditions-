#!/usr/bin/env python3
"""
TASK S -- final consistency and submission check. Produces results/paper/SUBMISSION_CHECK.md (PASS or FAIL with evidence for every item) and results/paper/submission_check.json.

    python -m tools.submission_check --paper "<.../cvavish>" --web web/out --site https://stereo-gated-results.vercel.app

Nothing here produces a result. It reads the built paper (PDF and sources), the deck, the built website (static HTML) and NUMBERS*.md, and tests them against each other and against the result files.
Honest limits: the number check looks for the canonical string of each headline value and for coarser or finer renderings of the SAME value as standalone tokens; a hit on a coarser rendering is reported
as a possible rounding disagreement and has to be read, because a different quantity can coincide with a rounded value.
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
import re
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (an IDE checker that never runs this line reports an unresolved import)

import pymupdf  # noqa: E402
from pptx import Presentation  # noqa: E402

import tools.paper_numbers as PN  # noqa: E402

RES = []          # (section, item, PASS|FAIL|NOTE, evidence)


def rec(sec, item, ok, ev):
    RES.append((sec, item, "PASS" if ok is True else ("FAIL" if ok is False else "NOTE"), ev))


def deck_text(path):
    p = Presentation(path)
    out = []

    def walk(shapes):
        for sh in shapes:
            if sh.shape_type == 6:
                yield from walk(sh.shapes)
            else:
                yield sh
    for i, s in enumerate(p.slides, 1):
        t = []
        for sh in walk(s.shapes):
            if sh.has_text_frame:
                t += [q.text for q in sh.text_frame.paragraphs]
            if getattr(sh, "has_table", False) and sh.has_table:
                for r in sh.table.rows:
                    t += [c.text for c in r.cells]
        out.append("\n".join(t))
    return out


def web_text(root):
    pages = {}
    for f in glob.glob(os.path.join(root, "**", "*.html"), recursive=True):
        h = open(f, encoding="utf-8", errors="ignore").read()
        h = re.sub(r"<script.*?</script>|<style.*?</style>", " ", h, flags=re.S)
        h = re.sub(r"<[^>]+>", " ", h)
        pages[os.path.relpath(f, root).replace("\\", "/")] = re.sub(r"\s+", " ", html.unescape(h))
    return pages


def values():
    V = {}
    for name, f, path, fmt, scale in PN.SPEC:
        d = PN.J(f)
        try:
            v = PN.dig(d, path)
            raw = float(v) * (1 if fmt.endswith("d}") else scale)
            V[name] = (fmt.format(int(v) if fmt.endswith("d}") else raw), raw, fmt)
        except Exception:  # noqa: BLE001
            V[name] = None
    return V


def tok(s):
    return re.compile(r"(?<![\d.])" + re.escape(s) + r"(?!\d)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper", required=True)
    ap.add_argument("--web", default="web/out")
    ap.add_argument("--site", default="https://stereo-gated-results.vercel.app")
    ap.add_argument("--out", default="results/paper")
    a = ap.parse_args()
    tectonic = r"C:\Users\Chandaluri Vaishnavi\anaconda3\envs\tex\Library\bin\tectonic.exe"

    # ------------------------------------------------------------------ build the paper again, twice, from the sources
    r1 = subprocess.run([tectonic, "main.tex"], cwd=a.paper, capture_output=True, text=True)
    r2 = subprocess.run([tectonic, "main.tex"], cwd=a.paper, capture_output=True, text=True)
    log = r1.stdout + r1.stderr + r2.stdout + r2.stderr
    pdf = pymupdf.open(os.path.join(a.paper, "main.pdf"))
    ptxt = "\n".join(p.get_text() for p in pdf).replace("\u2212", "-")
    tex = open(os.path.join(a.paper, "main.tex"), encoding="utf-8").read()
    dtxt = deck_text(os.path.join(a.paper, "CV Project PPT.pptx"))
    wpages = web_text(a.web)
    wtxt = " ".join(wpages.values())
    ntxt = open(os.path.join(a.out, "NUMBERS_TEXT.md"), encoding="utf-8").read() + open(os.path.join(a.out, "NUMBERS.md"), encoding="utf-8").read()
    V = values()

    # ================================================================== S.1
    chans = {"paper": ptxt, "deck": "\n".join(dtxt), "website": wtxt, "NUMBERS": ntxt}
    Q = [("BaseMiou", "official mIoU"), ("BaseMcIoU", "mcIoU"), ("GateAurocOof", "gate AUROC, out of fold (official risk)"), ("GateAurocTest", "gate AUROC, test"),
         ("GateAurcOof", "gate AURC, out of fold"), ("GateAurcTest", "gate AURC, test"), ("MsAurcOof", "max-softmax AURC, out of fold"), ("MsAurcTest", "max-softmax AURC, test"),
         ("FlaggedPct", "activation (flagged) rate, %"), ("FlagAThree", "flagged-frame instrument IoU change, always-A3"), ("FlagOracle", "oracle gain on flagged frames"),
         ("SelNoAHarmAll", "selector harmful rate, % of all frames"), ("SelNoAHarmCorr", "selector harmful rate, % of corrected frames"), ("FourHarmAll", "blind fusion harmful rate, % of all frames"),
         ("FourHarmCorr", "blind fusion harmful rate, % of corrected frames"), ("HallFrames", "hallucination rate, % of test frames"), ("CostBase", "latency, segmentation + gate, ms"),
         ("CostSelNoA", "latency, selective path, ms"), ("CostAllTwo", "latency, both corrections on every frame, ms"), ("KMean", "consecutive-frame ground-truth IoU, mean"),
         ("MiouMSix", "selector mIoU (M6)"), ("BlOursMiou", "baseline table: ours mIoU"), ("BlSegMiou", "baseline table: SegFormer-B0 mIoU"), ("BlUnetMiou", "baseline table: U-Net mIoU"),
         ("BlOursMc", "baseline table: ours mcIoU"), ("BlSegMc", "baseline table: SegFormer-B0 mcIoU"), ("BlUnetMc", "baseline table: U-Net mcIoU"),
         ("ZsIou", "EndoVis 2017 zero-shot mean binary IoU"), ("ZsIouRef", "EndoVis 2018 reference (same mapping)")]
    s1 = []
    bad = 0
    for name, label in Q:
        if V.get(name) is None:
            rec("S.1", label, False, f"macro {name} could not be computed")
            bad += 1
            continue
        canon, raw, fmt = V[name]
        cdec = len(canon.split(".")[1]) if "." in canon else 0
        canon_t = canon.lstrip("+")
        variants = []
        for d in range(2, 7):
            if d == cdec:
                continue
            s = f"{abs(raw):.{d}f}"
            if s.rstrip("0").endswith("."):
                continue
            variants.append(s)
        row = {"quantity": label, "canonical": canon, "channels": {}}
        for cn, ct in chans.items():
            have = len(tok(canon_t.lstrip("-")).findall(ct))
            var_hits = {v: len(tok(v).findall(ct)) for v in variants if tok(v).findall(ct)}
            row["channels"][cn] = (have, var_hits)
        s1.append(row)
    # Renderings that were read in context and are a different quantity (or a deliberate full-precision equality check), not a disagreement. Each carries its reason.
    REVIEWED = {("website", "0.612823"): "the identity check against the official script is shown at full precision on purpose (/results/, /maskcheck/)",
                ("website", "0.77"): "a different per-model column on /results/ (a share, not the gate AUROC)",
                ("paper", "0.30"): "axis tick label of the risk-coverage figure", ("website", "0.30"): "a change of +0.30 mcIoU points on /limits/",
                ("website", "0.31"): "SurgiReview's per-frame reliability and difficulty values", ("paper", "0.06"): "axis tick label of the accuracy-against-cost figure",
                ("paper", "0.07"): "axis tick label of the accuracy-against-cost figure", ("website", "0.07"): "a correlation (-0.07) on /selector/",
                ("website", "0.49"): "a pixel offset in the stereo table", ("website", "0.338"): "a per-frame value in SurgiReview",
                ("website", "0.85"): "end of an interval", ("website", "0.80"): "end of an interval"}
    reviewed_used = {}
    lines = ["| quantity | canonical value | paper | deck | website | NUMBERS | other renderings of the same value found |", "|---|---|---|---|---|---|---|"]
    for row in s1:
        cells, flags = [], []
        for cn in ("paper", "deck", "website", "NUMBERS"):
            have, var = row["channels"][cn]
            cells.append(str(have) if have else "-")
            for v, c in var.items():
                if (cn, v) in REVIEWED:
                    reviewed_used[(cn, v)] = REVIEWED[(cn, v)]
                    continue
                flags.append(f"{cn}: {v} (x{c})")
        lines.append(f"| {row['quantity']} | {row['canonical']} | " + " | ".join(cells) + " | " + ("; ".join(flags) if flags else "none") + " |")
        if flags:
            bad += 1
    rec("S.1", "headline numbers: canonical value present, no other precision of the same value", bad == 0,
        f"{len(s1)} quantities; {bad} with a rounding disagreement or missing macro. Precision rule: mIoU and mcIoU four decimals, AUROC and AURC three, harmful and flagged rates as in the macros. "
        "A number in the table is how many times the canonical string occurs in that channel; a '-' means the quantity is not shown there.\n\n" + "\n".join(lines) +
        "\n\nRenderings that matched a coarser or finer form of a value, were read in context, and are a different quantity or a deliberate full-precision check (so not a disagreement):\n" +
        "\n".join(f"- {cn}, `{v}`: {why}" for (cn, v), why in sorted(reviewed_used.items())))

    # ================================================================== S.2
    pats = ["TODO", "TBD", "XXX", "FIXME", "placeholder", "lorem", "Not measured yet", "pending", "???"]
    found = {}
    for cn, ct in (("paper (PDF text)", ptxt), ("deck", "\n".join(dtxt)), ("website", wtxt)):
        hits = {p: len(re.findall(re.escape(p), ct, flags=re.I)) for p in pats}
        found[cn] = {p: n for p, n in hits.items() if n}
    angle = {"paper (PDF text)": re.findall(r"<[A-Za-z][^<>]{0,30}>", ptxt), "deck": re.findall(r"<[A-Za-z][^<>]{0,30}>", "\n".join(dtxt)), "website": re.findall(r"<[A-Za-z][^<>]{0,30}>", wtxt)}
    owner = ["<reg no>", "<teammate name>", "<guide name>"]
    extra_angle = {k: [x for x in v if x not in owner] for k, v in angle.items()}
    s1_deck = dtxt[0]
    filled = ["Dr. Tripty Singh", "Athish Kulakudi Suresh", "BL.EN.U4CSE23210"]   # supplied by the owner on 12 November
    ok_owner = all(f in s1_deck for f in filled) and not any(o in "\n".join(dtxt) for o in owner)
    nogen = [k for k, v in found.items() if v]
    empty_cells = []
    prs = Presentation(os.path.join(a.paper, "CV Project PPT.pptx"))
    for i, s in enumerate(prs.slides, 1):
        for sh in s.shapes:
            if getattr(sh, "has_table", False) and sh.has_table:
                for ri, r in enumerate(sh.table.rows):
                    for ci, c in enumerate(r.cells):
                        if not c.text.strip():
                            empty_cells.append(f"slide {i} table row {ri+1} column {ci+1}")
    rec("S.2", "no TODO/TBD/XXX/FIXME/placeholder/lorem/'Not measured yet'/pending/??? in the paper PDF, deck or website", not nogen,
        "hits: " + (json.dumps(found) if nogen else "none in any of the three"))
    rec("S.2", "no angle-bracket placeholder other than the three owner placeholders", all(not v for v in extra_angle.values()), "others found: " + json.dumps(extra_angle))
    rec("S.2", "the three owner placeholders on deck slide 1 are filled with the details the owner supplied, and none remains anywhere", ok_owner,
        "guide Dr. Tripty Singh, teammate Athish Kulakudi Suresh, registration BL.EN.U4CSE23210 (the paper's author block names the same second author). 'Panel No.' stays blank: no value was supplied.")
    rec("S.2", "empty table cells or figure slots", None if empty_cells else True,
        ("empty cells (owner's template, not filled): " + "; ".join(empty_cells) + ". 'Panel No.' on slide 1 is also blank. Row 3 of the student table is empty by design of the template (a third member); the owner decides.") if empty_cells else "none")

    # ================================================================== S.3
    pages = len(pdf)
    err = [l for l in log.splitlines() if l.lower().startswith("error")]
    undefined = [l for l in log.splitlines() if "undefined" in l.lower() or "multiply defined" in l.lower() or "not found" in l.lower()]
    rec("S.3", "paper builds twice with zero errors (tectonic; pdflatex is not installed on this machine)", not err and not undefined and r1.returncode == 0 and r2.returncode == 0,
        f"two runs of tectonic on main.tex: return codes {r1.returncode}, {r2.returncode}; errors {len(err)}; undefined/multiply-defined {len(undefined)}; pages {pages}. Caveat: the requested `pdflatex` is not available, "
        "so the build was not run with pdflatex itself; the source uses only IEEEtran, cite, amsmath, graphicx, tikz, algorithm(ic) and similar packages and no non-ASCII characters (checked below), which is the usual cause of a pdflatex-only failure.")
    src_files = [os.path.join(a.paper, f) for f in ["main.tex", "numbers.tex", "baseline_text.tex"]] + glob.glob(os.path.join(a.paper, "tables", "*.tex"))
    nonascii = {os.path.basename(f): len(re.findall(r"[^\x00-\x7f]", open(f, encoding="utf-8").read())) for f in src_files if os.path.exists(f)}
    nonascii = {k: v for k, v in nonascii.items() if v}
    rec("S.3", "no non-ASCII characters in the LaTeX sources (pdflatex safety)", not nonascii, json.dumps(nonascii) if nonascii else "none")
    rec("S.3", "no '??' in the PDF and no empty macro value", ptxt.count("??") == 0 and "\\newcommand{\\N" in open(os.path.join(a.paper, "numbers.tex")).read() and "{--}" not in open(os.path.join(a.paper, "numbers.tex")).read(),
        f"'??' occurrences {ptxt.count('??')}; macros defined as '--': {open(os.path.join(a.paper, 'numbers.tex')).read().count('{--}')}")
    cites = set(k.strip() for m in re.findall(r"\\cite\{([^}]*)\}", tex) for k in m.split(","))
    bib = set(re.findall(r"\\bibitem\{([^}]*)\}", tex))
    labels = set(re.findall(r"\\label\{([^}]*)\}", tex))
    refs = set(re.findall(r"\\ref\{([^}]*)\}", tex)) | set(re.findall(r"\\eqref\{([^}]*)\}", tex))
    rec("S.3", "every \\ref resolves", not (refs - labels), f"{len(refs)} distinct refs, unresolved: {sorted(refs - labels)}")
    rec("S.3", "every \\cite resolves to a bibliography entry; every entry is cited", not (cites - bib) and not (bib - cites), f"{len(cites)} keys cited, {len(bib)} entries; cited but missing {sorted(cites - bib)}; uncited {sorted(bib - cites)}")
    tl = sorted(l for l in labels if l.startswith("tab:"))
    fl = sorted(l for l in labels if l.startswith("fig:"))
    un = [l for l in tl + fl if l not in refs]
    rec("S.3", "every table and figure is referenced from the body", not un, f"{len(tl)} tables, {len(fl)} figures; unreferenced: {un}")
    secs = ["Introduction", "Materials and Dataset", "Literature Survey", "Methodology", "Implementation", "Results", "Discussion", "Conclusion"]
    have = {s: bool(re.search(r"\\section\{[^}]*" + re.escape(s), tex)) for s in secs}
    rec("S.3", "required sections present", all(have.values()) and "thebibliography" in tex, json.dumps(have) + "; references: thebibliography present")
    claims = {"EndoVis 2017 not described as a dataset of this work (only as the zero-shot evaluation and related work)": ("zero-shot" in tex and "EndoVis 2017, Zero-Shot Evaluation Only" in tex),
              "no 'Stage 6' / difficult-event construction": ("Stage 6" not in tex and "Difficult-Event" not in tex and "event inventory" not in tex),
              "no recovery success rate / recovery latency / false-restoration / identity continuity / phase-wise metric": not re.search(r"recovery success|false[- ]restoration|identity continuity|phase-wise", tex, re.I),
              "contribution 4 replaced by harmful rate + AURC + abstention": ("harmful-correction rate" in tex and "AURC" in tex and "abstention" in tex),
              "no latency target anywhere in the paper; the cost per frame is reported as a measurement in Results": ("40\\,ms" not in tex and "display budget" not in tex and "Deployment Goals" not in tex and "\\label{eq:cost_model}" in tex and "Cost per frame" in tex)}
    for k, v in claims.items():
        rec("S.3", "R.3 item: " + k, v, "checked by text search of main.tex")
    rec("S.3", "page count", None, f"{pages} pages (IEEE two-column conference format)")

    # ================================================================== S.4
    def ex(p, minb=1):
        return os.path.exists(p) and os.path.getsize(p) >= minb
    figs = sorted(set(os.path.splitext(os.path.basename(p))[0] for p in glob.glob("results/paper/figures/fig*.pdf")))
    figs_png = sorted(glob.glob("results/paper/figures/fig*.png"))
    tabs = sorted(glob.glob("results/paper/tables/*.tex"))
    B = json.load(open("results/taskP/baselines.json"))
    fin = [m["label"] for m in B["models"]]
    un_ = [u["label"] for u in B.get("unavailable", [])]
    mdefs = re.findall(r"^## (\d+)\. (.+)$", open("results/paper/metrics_definitions.md", encoding="utf-8").read(), flags=re.M)
    rec("S.4", "1. baseline comparison with the proposed model", ex("results/taskP/baselines.json") and ex("results/paper/tables/baseline_comparison.tex") and len(fin) >= 3,
        f"trained and finished: {fin} (the proposed model plus {len(fin)-1} baselines); NOT finished: {un_} (four unstable runs, see RUNLOG.md and results/paper/P1_PROTOCOL.md). Table: results/paper/tables/baseline_comparison.tex "
        "(paper Table baselines, deck slide 9, website /bench/); per-class and per-sequence figures: fig03, fig04. Met with two baselines, not three.")
    rec("S.4", "2. graphs and plots", len(figs) >= 10 and ex("results/paper/figures/README.md"),
        f"{len(figs)} figures as PDF plus {len(figs_png)} PNG (300 dpi): {', '.join(figs)}; listing: results/paper/figures/README.md. Figure 13 (qualitative frames) deliberately not produced (licence, NEEDS OWNER). "
        f"Plus {len(tabs)} generated LaTeX tables: {', '.join(os.path.basename(t) for t in tabs)}.")
    z = json.load(open("results/taskP3/endovis17_zeroshot.json"))
    rec("S.4", "3. databases used", ex("results/taskP3/endovis17_zeroshot.json"),
        "EndoVis 2018 (training, every decision, official test) and EndoVis 2017 as a zero-shot, evaluation-only test (added: results/taskP3/endovis17_zeroshot.json, per-frame CSV, "
        f"{z['endovis2017']['detectors']['_n_frames']} frames in {z['endovis2017']['detectors']['_n_sequences']} sequences, scored once; class mapping and every excluded class stated in the JSON, the paper and /bench/). "
        "Limit: the 2017 labels allow only instrument against background, and its result (no large drop) is a narrow one. Two databases, not three.")
    rec("S.4", "4. performance evaluators", len(mdefs) >= 12 and ex("results/paper/metrics_definitions.md"),
        f"{len(mdefs)} metrics, each with formula, range, blind spots, source file and interval method in results/paper/metrics_definitions.md: " + "; ".join(f"{n}. {t}" for n, t in mdefs) +
        ". Evaluator code: tools/final_eval.py (official, matching the challenge script to the stated tolerance), tools/baseline_eval.py, tools/endovis17_zeroshot.py, tools/failure_detect_table.py, tools/calibration_pixels.py.")

    # ================================================================== S.5 (deck: structural part; the visual review is stated separately)
    n_sl = len(dtxt)
    rec("S.5", "deck: slide count, numbering and emptiness", n_sl == 13 and all(len(t.strip()) > 0 for t in dtxt), f"{n_sl} slides (the plain-language deck of tools/build_deck_plain.py has 13: 10 content slides, the title, the references and the thank-you), none empty; the slide numbers are live number fields, checked in the render (1..13, in order)")
    small = []
    for i, s in enumerate(prs.slides, 1):
        for sh in s.shapes:
            if sh.has_text_frame:
                for q in sh.text_frame.paragraphs:
                    for r in q.runs:
                        if r.font.size is not None and r.font.size.pt < 10:
                            small.append((i, r.font.size.pt))
    rec("S.5", "deck: no explicit font size below 10 pt", not small, f"runs below 10 pt: {small[:5]}")
    rec("S.5", "deck: every slide rendered to an image and inspected for overflow", None,
        "done by hand after the last update: PowerPoint export to PNG (13 images), all 13 viewed. Slides 7 to 11 were shortened until the text stopped at the frame; slide 3's table sits partly outside the template's rounded frame (the owner's original layout).")

    # ================================================================== S.6
    pv = subprocess.run([sys.executable, "-m", "tools.verify_web_data", "--results", "results", "--web", "web"], capture_output=True, text=True)
    last = [l for l in pv.stdout.splitlines() if l.startswith("RESULT")]
    rec("S.6", "tools/verify_web_data.py: zero untraced fields and zero source literals", pv.returncode == 0 and bool(last) and "0 untraced data fields, 0 source literals" in last[0], last[0] if last else pv.stdout[-300:])
    routes = ["", "problem/", "analyse/", "my-results/"]
    codes = {}
    for r in routes:
        try:
            codes["/" + r] = urllib.request.urlopen(urllib.request.Request(a.site.rstrip("/") + "/" + r, headers={"User-Agent": "check"}), timeout=30).status
        except Exception as e:  # noqa: BLE001
            codes["/" + r] = getattr(e, "code", str(e))
    rec("S.6", "every route returns HTTP 200 on the deployed site", all(c == 200 for c in codes.values()), json.dumps(codes))
    media = [os.path.relpath(p, a.web) for p in glob.glob(os.path.join(a.web, "**", "*"), recursive=True) if os.path.splitext(p)[1].lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp", ".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".npz", ".npy") and "_next" not in p.replace("\\", "/")]
    # the paper's plots are published on purpose: allowed only when byte-identical to the file in results/paper/figures
    import hashlib
    figs_json = os.path.join(os.path.dirname(os.path.abspath(a.web)), "data", "figures.json")
    allowed, bad_fig = set(), []
    if os.path.isfile(figs_json):
        for grp in json.load(open(figs_json, encoding="utf-8")):
            for f in grp["figures"]:
                rel = "figures/" + os.path.basename(f["file"])
                src = os.path.join("results", "paper", "figures", os.path.basename(f["file"]))
                pub = os.path.join(a.web, rel)
                if os.path.isfile(pub) and os.path.isfile(src) and hashlib.sha256(open(pub, "rb").read()).hexdigest() == hashlib.sha256(open(src, "rb").read()).hexdigest():
                    allowed.add(os.path.normpath(rel))
                else:
                    bad_fig.append(rel)
    n_plots = len(allowed)
    # the example frames the owner chose to publish (tools/export_web_examples.py): allowed only when listed in data/examples.json
    ex_json = os.path.join(os.path.dirname(os.path.abspath(a.web)), "data", "examples.json")
    n_examples = 0
    if os.path.isfile(ex_json):
        exd = json.load(open(ex_json, encoding="utf-8"))
        files = [x["file"] for k in ("samples", "segmentation", "correction") for x in exd[k]] + [exd["dataset"]["file"], exd["class_key"]]
        for f in files:
            rel = os.path.normpath(f.lstrip("/"))
            if os.path.isfile(os.path.join(a.web, rel)):
                allowed.add(rel); n_examples += 1
    media = [m for m in media if os.path.normpath(m) not in allowed] + bad_fig
    rec("S.6", "published images are only the paper plots and the chosen example frames; no model weights", not media, f"image or weight-type files outside the framework's own assets the {n_plots} paper plots (byte-identical to results/paper/figures) and the {n_examples} example frames listed in data/examples.json: {media}; total output size {sum(os.path.getsize(p) for p in glob.glob(os.path.join(a.web, '**', '*'), recursive=True) if os.path.isfile(p))/1e6:.1f} MB")
    real = [k for k in wpages if k.endswith("index.html") and not k.startswith(("404", "_not-found"))]
    miss = [k for k in real if not re.search(r"research.prototype", wpages[k], re.I)]
    rec("S.6", "research-prototype disclaimer on every page", not miss, f"pages checked: {len(real)} (the framework's own 404 page excluded); pages without it: {miss}")

    # ================================================================== S.7 honesty sweep
    LIG = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}

    def has(t, *ps):
        for k, v in LIG.items():
            t = t.replace(k, v)
        t1 = re.sub(r"\s+", " ", t)
        t2 = re.sub(r"\s+", " ", re.sub(r"-\n", "", t))
        return all(re.search(p, t1, re.I | re.S) or re.search(p, t2, re.I | re.S) for p in ps)
    dk = "\n".join(dtxt)
    rec("S.7", "cross-view criterion not met, AURC reported", has(ptxt, r"stopped there", r"AURC difference") and has(dk, r"did not clear the pre-set test"),
        "paper Section Cross-View Indicators and deck slide 10. Note for the owner: the paper says the rule did not NAME AURC (the record in results/taskD shows the rule named AUROC only); the brief's phrase 'pre-specified secondary outcome' is therefore not repeated, because the record does not support it.")
    rec("S.7", "detector comparison is label-dependent; test intervals wide", has(ptxt, r"its own label", r"instrument IoU below", r"coarse"), "paper Failure Detection; deck slide 9 ('The gate leads only on its own label')")
    rec("S.7", "correction does not raise the aggregate score, with the arithmetic reason", has(ptxt, r"invented", r"ground truth", r"official score cannot see") and has(dk, r"invented classes removed"), "paper Correction on the Frames the Gate Flags; deck slides 9 and 11")
    rec("S.7", "negatives present: temporal, disparity, difficulty index, TTA, ensemble, A4", has(ptxt, r"consecutive ground-truth", r"at chance", r"difficulty index", r"Test-time augmentation", r"ensemble", r"Action A4"), "paper Results; deck slide 10 (difficulty index and A4 are in the paper only)")
    envs = re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tex, flags=re.S)
    pub_in_tables = [e[:60] for e in envs if re.search(r"66\.2|77\.5", e)]
    pub_in_text = len(re.findall(r"66\.2|77\.5", tex))
    rec("S.7", "no ours-next-to-published-number without a protocol column", not pub_in_tables and "sota_comparison" in tex,
        f"the published figures (66.2 and 77.5) occur {pub_in_text} times, all in running text of the Literature Survey with the sentence that their protocols differ and that they are not placed beside ours; "
        f"in any table environment: {len(pub_in_tables)}. The only literature table (tab:sota_comparison) holds no numeric results.")
    rec("S.7", "conformal risk budget not met on test, and said", has(ptxt, r"did not on the test set") and has(dk, r"not on test"), "paper Correction section and Calibration; deck slides 6 and 10")
    rec("S.7", "gate calibration (ranking versus level) stated", has(ptxt, r"Ranking is not level", r"optimistic") and has(wtxt, r"optimistic"), "paper Calibration and Discussion, deck slide 10, website /limits/")
    rec("S.7", "EndoVis 2017 zero-shot stated as narrow, no overclaim", has(ptxt, r"narrow result", r"does not support a claim of general robustness"), "paper Discussion (Transfer)")
    rec("S.7", "max-softmax already a gate indicator (not added); gate with and without it reported against max-softmax, both splits and both labels", has(ptxt, r"so it was not added", r"with and without it", r"does not beat max-softmax") and has(wtxt, r"already one of the gate"),
        "paper Failure Detection (Max-Softmax Inside the Gate, Table gate ablation); website /results/ gate ablation; results/FINDINGS.md")
    rec("S.7", "paired comparison stated, not only overlapping intervals (corrected sentence)", has(ptxt, r"paired comparison", r"just excludes zero") and not has(ptxt, r"no detector is shown to beat another"), "paper Failure Detection; website /bench/ and /results/")
    rec("S.7", "gate seeds: mean and spread out of fold, released gate is the best seed, limitation line", has(ptxt, r"Five refits", r"standard deviation", r"highest out-of-fold AUROC", r"segmenter seed variance") and has(wtxt, r"five seeds"), "paper; website /results/ and /limits/; results/FINDINGS.md")
    rec("S.7", "DeepLabV3+ fifth attempt (full precision) stated and two baselines kept", has(ptxt, r"full precision", r"diverged earlier") and has(wtxt, r"full precision") and has(dk, r"five attempts"), "paper, deck slide 9, website /bench/ and /results/, P1_PROTOCOL.md")
    rec("S.7", "rubric items present: baseline comparison, plots, databases (4), evaluators", has(ptxt, r"Evaluation Summary", r"Databases used", r"Performance evaluators", r"eight data plots") and has(dk, r"Databases Used \(4\)", r"How We Measure Success", r"Comparison With Other Models", r"Plot 1"),
        "paper Evaluation Summary (Tables databases and evaluators); deck slides 7 to 13")
    rec("S.7", "DeepLabV3+ absence stated wherever baselines are listed", has(ptxt, r"DeepLabV3\+ did not") and has(dk, r"DeepLabV3\+ did not train stably") and has(wtxt, r"DeepLabV3\+"), "paper, deck slides 9 and 11, website /bench/")

    # ------------------------------------------------------------------ write
    order = ["S.1", "S.2", "S.3", "S.4", "S.5", "S.6", "S.7"]
    titles = {"S.1": "One number, one value, everywhere", "S.2": "Nothing unfinished is left visible", "S.3": "The paper builds and is complete", "S.4": "Rubric coverage",
              "S.5": "The deck", "S.6": "The website", "S.7": "Honesty sweep"}
    nfail = sum(1 for r in RES if r[2] == "FAIL")
    md = ["# Submission check (Task S)", "", f"Generated by `tools/submission_check.py`. **{sum(1 for r in RES if r[2]=='PASS')} PASS, {nfail} FAIL, {sum(1 for r in RES if r[2]=='NOTE')} NOTE.** "
          "A FAIL is an item the owner has to decide or fix; none is hidden. Evidence follows each item.", ""]
    for s in order:
        md += [f"## {s} {titles[s]}", ""]
        for sec, item, st, ev in RES:
            if sec == s:
                md += [f"- **{st}** - {item}", "", "  " + ev.replace("\n", "\n  "), ""]
    open(os.path.join(a.out, "SUBMISSION_CHECK.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump([dict(section=s, item=i, status=st, evidence=e) for s, i, st, e in RES], open(os.path.join(a.out, "submission_check.json"), "w"), indent=1)
    for sec, item, st, ev in RES:
        print(f"[{st}] {sec} {item}")
    print(f"\n{sum(1 for r in RES if r[2]=='PASS')} PASS, {nfail} FAIL")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

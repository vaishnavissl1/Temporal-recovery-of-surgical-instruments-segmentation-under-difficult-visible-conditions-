#!/usr/bin/env python3
"""
Plain-language deck. Starts from the owner's untouched template (`CV Project PPT_original.pptx`) and writes `CV Project PPT.pptx`:

    python -m tools.build_deck_plain --deck "C:/Users/Chandaluri Vaishnavi/Downloads/cvrsfesp/cvavish"

Kept from the owner's deck, untouched: slide 1 (title block and its three placeholders), the literature table, the references, the thank-you slide.
Everything else is new and written for a listener with no background: the problem, what earlier work missed, one architecture diagram, one methodology flowchart, the four
databases, the evaluators, the baseline comparison, six plots, the zero-shot test, what worked and what did not, the conclusion. Diagrams are native PowerPoint shapes (editable).
Every number is read through tools/paper_numbers.py (the paper's macros); none is typed here. Speaker notes carry a short plain-words script for each slide.
This script supersedes tools/update_deck.py for the final deck (update_deck.py edits the old wording in place).
"""
import argparse
import copy
import os
import shutil

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

import tools.paper_numbers as PN
from tools.update_deck import values

FONT = "Times New Roman"
GUIDE = "Dr. Tripty Singh"
MATE_NAME, MATE_REG = "Athish Kulakudi Suresh", "BL.EN.U4CSE23210"
INK = RGBColor(0x1F, 0x2A, 0x44)
BLUE, BLUE_F = RGBColor(0x2F, 0x6D, 0xB5), RGBColor(0xDC, 0xE8, 0xF7)
GREEN, GREEN_F = RGBColor(0x2E, 0x7D, 0x4F), RGBColor(0xDD, 0xF0, 0xE3)
ORANGE, ORANGE_F = RGBColor(0xC2, 0x6A, 0x12), RGBColor(0xFC, 0xE8, 0xCB)
GREY, GREY_F = RGBColor(0x55, 0x5B, 0x66), RGBColor(0xEE, 0xEF, 0xF2)
X0, X1, Y0, Y1 = 0.62, 9.38, 2.0, 6.7   # the usable area inside the template's rounded frame (inches)


# ---------------------------------------------------------------- helpers
def walk(shapes):
    for sh in shapes:
        yield sh
        if sh.shape_type == 6:
            yield from walk(sh.shapes)


def style_run(r, size, bold=False, color=INK, italic=False):
    r.font.name = FONT
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    rPr = r._r.get_or_add_rPr()
    for tag in ("a:cs",):
        el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", FONT)


def set_bullet(p, on=True, indent=0.22):
    pPr = p._p.get_or_add_pPr()
    for t in ("a:buNone", "a:buChar", "a:buFont"):
        for el in pPr.findall(qn(t)):
            pPr.remove(el)
    if on:
        pPr.set("marL", str(int(Inches(indent))))
        pPr.set("indent", str(-int(Inches(indent))))
        f = etree.SubElement(pPr, qn("a:buFont"))
        f.set("typeface", "Arial")
        b = etree.SubElement(pPr, qn("a:buChar"))
        b.set("char", "\u2022")
    else:
        pPr.set("marL", "0")
        pPr.set("indent", "0")
        etree.SubElement(pPr, qn("a:buNone"))


def textbox(slide, x, y, w, h, paras, size=16, bullets=True, color=INK, align=None, space_after=6, anchor=None, line=1.0):
    """paras: list of str or (str, dict(bold=, size=, color=, italic=)) or list of runs [(text, fmt), ...]."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    if anchor:
        tf.vertical_anchor = anchor
    first = True
    for item in paras:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        runs = item if isinstance(item, list) else [item]
        for run in runs:
            text, fmt = (run, {}) if isinstance(run, str) else run
            r = p.add_run()
            r.text = text
            style_run(r, fmt.get("size", size), fmt.get("bold", False), fmt.get("color", color), fmt.get("italic", False))
        p.space_after = Pt(space_after)
        p.line_spacing = line
        if align:
            p.alignment = align
        set_bullet(p, bullets)
    return tb


def box(slide, x, y, w, h, title, desc=None, fill=BLUE_F, edge=BLUE, tsize=14, dsize=11.5, shape=MSO_SHAPE.ROUNDED_RECTANGLE, dash=False):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.color.rgb = edge
    s.line.width = Pt(1.75)
    if dash:
        s.line.dash_style = 4  # dash
    s.shadow.inherit = False
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = 0.12
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.06)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = title
    style_run(r, tsize, True, INK)
    if desc:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.CENTER
        r2 = p2.add_run()
        r2.text = desc
        style_run(r2, dsize, False, INK)
    return s


def arrow(slide, x1, y1, x2, y2, color=INK, width=2.25, dash=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(width)
    if dash:
        c.line.dash_style = 4
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return c


def label(slide, x, y, w, text, size=12, color=INK, bold=False, align=PP_ALIGN.CENTER, italic=False):
    return textbox(slide, x, y, w, 0.3, [(text, dict(bold=bold, italic=italic))], size=size, bullets=False, color=color, align=align, space_after=0)


def badge(slide, x, y, n, color=BLUE):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(0.34), Inches(0.34))
    s.fill.solid()
    s.fill.fore_color.rgb = color
    s.line.fill.background()
    s.shadow.inherit = False
    tf = s.text_frame
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = str(n)
    style_run(r, 13, True, RGBColor(255, 255, 255))


def table(slide, x, y, w, rows, col_w, size=12, header_fill=BLUE, row_h=0.4, first_col_bold=True):
    nr, nc = len(rows), len(rows[0])
    gs = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    t = gs.table
    tblPr = t._tbl.tblPr
    tblPr.set("bandRow", "0")
    tblPr.set("firstRow", "0")
    sid = tblPr.find(qn("a:tableStyleId"))
    if sid is not None:
        sid.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"   # "No Style, Table Grid"
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.02)
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if (j == 0 or len(val) > 12) else PP_ALIGN.CENTER
            r = p.add_run()
            r.text = val
            hdr = i == 0
            style_run(r, size, hdr or (j == 0 and first_col_bold), RGBColor(255, 255, 255) if hdr else INK)
            if hdr:
                c.fill.solid()
                c.fill.fore_color.rgb = header_fill
            elif i % 2 == 0:
                c.fill.solid()
                c.fill.fore_color.rgb = RGBColor(0xF3, 0xF7, 0xFC)
            else:
                c.fill.solid()
                c.fill.fore_color.rgb = RGBColor(255, 255, 255)
    return gs


def new_slide(prs, layout, number_proto, title, tsize=30):
    s = prs.slides.add_slide(layout)
    for ph in list(s.placeholders):
        if ph.placeholder_format.type != 1:   # keep only the title
            ph._element.getparent().remove(ph._element)
    t = s.shapes.title
    t.left, t.top, t.width, t.height = Emu(381001), Emu(978677), Emu(8229600), Emu(1005206)
    tf = t.text_frame
    tf.text = ""
    r = tf.paragraphs[0].add_run()
    r.text = title
    r.font.name = FONT
    r.font.size = Pt(tsize)
    r.font.color.rgb = RGBColor(0, 0, 0)
    el = copy.deepcopy(number_proto)
    for p_ in el.iter(qn("a:p")):
        for ch in list(p_):
            if ch.tag in (qn("a:r"), qn("a:fld")):
                p_.remove(ch)
        fld = etree.Element(qn("a:fld"))
        fld.set("id", "{B6F15528-21DE-4FAA-801E-634DDDAF4B2B}")
        fld.set("type", "slidenum")
        etree.SubElement(fld, qn("a:rPr")).set("lang", "en-IN")
        etree.SubElement(fld, qn("a:t")).text = "#"
        p_.insert(0, fld)
    s.shapes._spTree.append(el)
    return s


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def delete_slide(prs, slide):
    lst = prs.slides._sldIdLst
    for sid in list(lst):
        if prs.part.related_part(sid.rId) is slide.part:
            prs.part.drop_rel(sid.rId)
            lst.remove(sid)
            return


def num(s):
    return float(s.replace("+", ""))


# ---------------------------------------------------------------- the deck
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deck", required=True)
    ap.add_argument("--assets", default="results/paper/deck_assets")
    a = ap.parse_args()
    cur, orig = os.path.join(a.deck, "CV Project PPT.pptx"), os.path.join(a.deck, "CV Project PPT_original.pptx")
    if not os.path.exists(orig):
        shutil.copy2(cur, orig)
    os.makedirs(a.assets, exist_ok=True)
    fig = "results/paper/figures"
    prs = Presentation(orig)
    V = values()
    old = list(prs.slides)

    # ---------- slide 1: the three owner placeholders, filled with the details the owner gave
    def set_text(frame, old_text, new_text):
        for p_ in frame.paragraphs:
            for r_ in p_.runs:
                if old_text in r_.text:
                    r_.text = r_.text.replace(old_text, new_text)
                    return True
        return False
    for sh in old[0].shapes:
        if sh.has_text_frame and "<guide name>" in sh.text_frame.text:
            assert set_text(sh.text_frame, "<guide name>", GUIDE), "guide placeholder not found"
        if sh.has_table:
            for row in sh.table.rows:
                if row.cells[1].text.strip() == "<reg no>":
                    assert set_text(row.cells[1].text_frame, "<reg no>", MATE_REG) and set_text(row.cells[2].text_frame, "<teammate name>", MATE_NAME)
    layout = old[8].slide_layout
    num_proto = [sh for sh in old[8].shapes if sh.name.startswith("Slide Number")][0]._element

    # ---------- bar chart of the three models (numbers from the same macros as the paper)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    models = [("Ours", V["BlOursMiou"], V["BlOursMc"], "#0072B2"), ("SegFormer-B0", V["BlSegMiou"], V["BlSegMc"], "#E69F00"), ("U-Net", V["BlUnetMiou"], V["BlUnetMc"], "#009E73")]
    fig_, ax = plt.subplots(figsize=(4.6, 2.9), dpi=300)
    w = 0.36
    for k, (nm, mi, mc, col) in enumerate(models):
        ax.bar(k - w / 2, num(mi), w, color=col, edgecolor="black", linewidth=0.6)
        ax.bar(k + w / 2, num(mc), w, color=col, edgecolor="black", linewidth=0.6, hatch="//", alpha=0.75)
        ax.text(k - w / 2, num(mi) + 0.012, mi, ha="center", fontsize=8)
        ax.text(k + w / 2, num(mc) + 0.012, mc, ha="center", fontsize=8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([m[0] for m in models], fontsize=9)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("score (0 to 1, higher is better)", fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc="white", ec="black"), plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", hatch="//")],
              labels=["mIoU (main score)", "mcIoU (every class counts equally)"], fontsize=7, loc="upper right", frameon=False)
    fig_.tight_layout()
    bar_png = os.path.join(a.assets, "deck_baseline_bars.png")
    fig_.savefig(bar_png)
    plt.close(fig_)

    new = {}

    # ---------- 2. the problem, and what earlier work missed (one slide)
    s = new_slide(prs, layout, num_proto, "The Problem and What Earlier Work Missed", 26)
    textbox(s, X0 + 0.1, 2.0, 8.5, 4.7, [
        "In robotic surgery the surgeon works by looking at a camera. A computer program can draw an outline around every instrument in each video frame. This is called segmentation.",
        [("Most of the time it works well. But in short stretches (smoke, blood, glare, tools crossing each other) it makes mistakes. ", {}), ("The program does not know when it is wrong.", dict(bold=True))],
        f"Checking and fixing every frame is slow ({V['CostAllTwo']} ms per frame, against {V['CostSelNoA']} ms when only the doubtful frames are fixed), and a \"fix\" can even spoil a frame that was already good.",
        "Earlier papers on this dataset report accuracy only. None says when the model is unsure, none counts how often a fix makes things worse, and none releases its trained model, so we trained the comparison models ourselves.",
        [("Our goal: let the system check itself, and fix only the frames it is unsure about.", dict(bold=True, color=BLUE))],
    ], size=16, space_after=9, line=1.03)
    notes(s, "Say it simply: surgeons see through a camera, the computer draws an outline on each tool. It is usually right, sometimes wrong, and it never says which. Fixing every frame costs time and can make good frames worse. Earlier papers chase higher accuracy; nobody asks when to trust the output. So we built something that checks its own work first.")
    new[2] = s

    # ---------- 5. architecture (simple)
    s = new_slide(prs, layout, num_proto, "How the System Works (Architecture)", 28)
    bw, bh = 1.8, 1.0
    xs = [0.66, 2.91, 5.16, 7.41]
    y1, y2 = 2.2, 4.35
    box(s, xs[0], y1, bw, bh, "Video frame", "One picture from the surgical camera", GREY_F, GREY)
    box(s, xs[1], y1, bw, bh, "1. Segmenter", "Draws an outline around every tool", BLUE_F, BLUE)
    box(s, xs[2], y1, bw, bh, "2. Checker", "Scores how sure it is of the outline", ORANGE_F, ORANGE)
    box(s, xs[3], y1, bw, bh, "Final outline", "What the surgeon-side software receives", BLUE_F, BLUE)
    arrow(s, xs[0] + bw, y1 + bh / 2, xs[1], y1 + bh / 2)
    arrow(s, xs[1] + bw, y1 + bh / 2, xs[2], y1 + bh / 2)
    arrow(s, xs[2] + bw, y1 + bh / 2, xs[3], y1 + bh / 2, GREEN)
    textbox(s, xs[2] + bw - 0.02, y1 + bh / 2 - 0.36, 0.5, 0.3, [("Sure", dict(bold=True, color=GREEN))], size=12, bullets=False, space_after=0, align=PP_ALIGN.CENTER)
    box(s, xs[1], y2, bw, bh + 0.1, "Memory", "Keeps the last 3 frames it was sure about", GREEN_F, GREEN)
    box(s, xs[2], y2, bw, bh + 0.1, "3. Fixer", "Two quick fixes: reuse a recent good outline, or tidy rough edges", GREEN_F, GREEN, dsize=11)
    box(s, xs[3], y2, bw, bh + 0.1, "4. Chooser", "Picks the better fix, or keeps the original", GREEN_F, GREEN)
    arrow(s, xs[2] + bw / 2, y1 + bh, xs[2] + bw / 2, y2, ORANGE)
    textbox(s, xs[2] + bw / 2 + 0.08, y1 + bh + 0.18, 1.7, 0.7, [(f"Unsure: about {V['FlaggedPct']}% of frames", dict(bold=True, color=ORANGE))], size=12, bullets=False, space_after=0)
    arrow(s, xs[1] + bw, y2 + (bh + 0.1) / 2, xs[2], y2 + (bh + 0.1) / 2, GREEN)
    arrow(s, xs[2] + bw, y2 + (bh + 0.1) / 2, xs[3], y2 + (bh + 0.1) / 2, GREEN)
    arrow(s, xs[3] + bw / 2, y2, xs[3] + bw / 2, y1 + bh, GREEN)
    label(s, 0.66, 5.7, 8.5, "Sure frames go straight through, untouched. Only doubtful frames are fixed.", 14, INK, True)
    label(s, 0.66, 6.05, 8.5, "Tried and dropped: using the second (right) camera, and a step that re-labels tools between frames. Neither helped.", 11.5, GREY, False, italic=True)
    notes(s, "Walk left to right. A frame goes into the segmenter, which draws outlines. The checker gives a confidence score. If it is sure, the outline goes straight out. If it is unsure (about three frames in ten), the fixer tries two quick repairs using the memory of recent good frames, and the chooser either picks the better one or keeps the original.")
    new[5] = s

    # ---------- 6. methodology flowchart
    s = new_slide(prs, layout, num_proto, "How We Did It (Methodology)", 28)
    bw, bh = 1.95, 1.55
    xs = [0.66, 2.86, 5.06, 7.26]
    y1, y2 = 2.1, 4.45
    steps = [
        ("Collect data", f"EndoVis 2018: 19 surgery videos. 15 to learn from, 4 kept as the final exam.", BLUE_F, BLUE),
        ("Train the segmenter", "Teach it to colour every tool part and tissue type in each frame.", BLUE_F, BLUE),
        ("Train the checker", "Teach a small network to guess how good each outline is, using videos the model had not seen.", ORANGE_F, ORANGE),
        ("Set the \"unsure\" line", f"Pick a cut-off score ({V['Tau']}) so bad frames are rarely passed as good.", ORANGE_F, ORANGE),
    ]
    for k, (t, d, f, e) in enumerate(steps):
        box(s, xs[k], y1, bw, bh, t, d, f, e, 14, 11.5)
        badge(s, xs[k] - 0.1, y1 - 0.14, k + 1, e)
        if k:
            arrow(s, xs[k - 1] + bw, y1 + bh / 2, xs[k], y1 + bh / 2)
    steps2 = [
        ("Final exam", f"Run once on the 4 unseen test videos ({V['TestFrames']} frames).", GREEN_F, GREEN),
        ("Compare", "Train SegFormer-B0 and U-Net the same way and compare scores.", GREEN_F, GREEN),
        ("New-data test", f"EndoVis 2017: {V['ZsFrames']} frames, no retraining, scored once.", GREEN_F, GREEN),
    ]
    xs2 = [xs[3], xs[2], xs[1]]
    for k, (t, d, f, e) in enumerate(steps2):
        box(s, xs2[k], y2, bw, bh, t, d, f, e, 14, 11.5)
        badge(s, xs2[k] - 0.1, y2 - 0.14, 5 + k, e)
        if k:
            arrow(s, xs2[k - 1], y2 + bh / 2, xs2[k] + bw, y2 + bh / 2)
    arrow(s, xs[3] + bw / 2, y1 + bh, xs[3] + bw / 2, y2)
    textbox(s, 0.66, 4.6, 2.0, 1.2, [("Rule we kept: nothing is chosen by looking at the test videos.", dict(italic=True))], size=12, bullets=False, color=GREY)
    notes(s, "Seven steps. The key rule: every decision, including the cut-off, was made on training videos only; the four test videos were scored once at the end. The 2017 data was never used for training or tuning.")
    new[6] = s

    # ---------- 7. databases
    s = new_slide(prs, layout, num_proto, "Databases Used (4)", 30)
    rows = [
        ["Database", "What it contains", "How we used it"],
        ["1. EndoVis 2018", f"Robot kidney-surgery videos, 12 labelled classes (tool parts, tissue types). 19 videos.", f"Learning (15 videos, {V['OofFrames']} frames) and the final test (4 videos, {V['TestFrames']} frames)."],
        ["2. EndoVis 2017", f"Surgical frames labelled tool against background. {V['ZsFrames']} frames from {V['ZsSeqs']} videos used.", "Evaluation only: never trained on, scored once."],
        ["3. SurgeNetXL", "A large collection of surgical video, used by its authors for self-supervised pretraining.", "Starting weights of our segmenter. We did not retrain it."],
        ["4. ImageNet", "General photographs from everyday life.", "Starting weights of the SegFormer-B0 comparison model."],
    ]
    table(s, X0 + 0.05, 2.1, 8.6, rows, [1.7, 3.8, 3.1], size=12.5, row_h=0.78)
    label(s, X0 + 0.05, 6.1, 8.6, "Two databases are used for results (1 and 2); two only supply starting weights (3 and 4).", 13, INK, True)
    notes(s, "Four databases in total. Only EndoVis 2018 was used to train and test. EndoVis 2017 is a surprise test on a different year of the same challenge. The other two only provide pre-trained starting weights; their images were not used by us.")
    new[7] = s

    # ---------- 8. evaluators
    s = new_slide(prs, layout, num_proto, "How We Measure Success (Evaluators)", 28)
    rows = [
        ["Measure", "In plain words", "Better is"],
        ["mIoU (official)", "How well the drawn outline overlaps the true one, from 0 to 1.", "higher"],
        ["mcIoU, per-class IoU", "The same, with every class counted equally, so rare ones matter.", "higher"],
        ["Boundary F1, clDice", "Are the edges and the thin parts (shaft, thread, needle) right?", "higher"],
        ["AURC", "How costly the mistakes are if the system keeps only its most sure frames.", "lower"],
        ["AUROC", "How well its \"unsure\" score picks out the bad frames.", "higher"],
        ["Calibration error (ECE)", "When it says \"90% sure\", is it right about 90% of the time?", "lower"],
        ["Harmful-fix rate", "How often a fix makes a good frame worse.", "lower"],
        ["Size and speed", "Parameters, GFLOPs, milliseconds per frame.", "lower"],
    ]
    table(s, X0 + 0.05, 2.05, 8.6, rows, [2.15, 5.55, 0.9], size=11.5, row_h=0.43)
    label(s, X0 + 0.05, 6.0, 8.6, "All ranges are 95% intervals that resample whole videos, never single frames.", 12, GREY, False, italic=True)
    notes(s, "IoU is simply overlap: draw both outlines, divide the shared area by the combined area. Everything else is a different angle: per class, edges, how well the system knows when it is wrong, how honest its confidence is, how often a fix hurts, and how fast.")
    new[8] = s

    # ---------- 9. baseline comparison
    s = new_slide(prs, layout, num_proto, "Comparison With Other Models (Baselines)", 28)
    rows = [
        ["Model", "mIoU", "mcIoU", "Size (M)", "ms/frame", "Gap to ours (95% interval)"],
        ["Ours (CAFormer-S18)", V["BlOursMiou"], V["BlOursMc"], V["BlOursParams"], V["BlOursMs"], "-"],
        ["SegFormer-B0", V["BlSegMiou"], V["BlSegMc"], V["BlSegParams"], V["BlSegMs"], f"{V['BlSegDiff']} pts ({V['BlSegLo']} to {V['BlSegHi']})"],
        ["U-Net, from scratch", V["BlUnetMiou"], V["BlUnetMc"], V["BlUnetParams"], V["BlUnetMs"], f"{V['BlUnetDiff']} pts ({V['BlUnetLo']} to {V['BlUnetHi']})"],
    ]
    table(s, X0 + 0.05, 2.05, 8.6, rows, [2.1, 0.8, 0.85, 0.9, 0.95, 3.0], size=11.5, row_h=0.4)
    s.shapes.add_picture(bar_png, Inches(X0 + 0.05), Inches(3.78), width=Inches(4.3))
    textbox(s, 5.1, 3.85, 4.2, 2.8, [
        "Same data, same recipe, same scoring for all three.",
        "Ours is higher on all four test videos.",
        f"SegFormer-B0 is much smaller ({V['BlSegParams']} M) and faster ({V['BlSegMs']} ms): the price of accuracy is size.",
        "DeepLabV3+ did not train stably (five attempts, one in full precision), so a fourth model is left out.",
        "Starting weights differ between models, so this compares design plus head start.",
    ], size=12.5, space_after=5, line=1.0)
    notes(s, "We trained the other two models ourselves with the same settings because nobody publishes a trained model for this dataset. Ours is best. Be upfront about two things: the smaller model is much faster, and DeepLabV3+ failed to train so it is not in the table.")
    new[9] = s

    # ---------- 9. plots: per class and per sequence (one slide)
    s = new_slide(prs, layout, num_proto, "Plot 1 and Plot 2: Each Class, Each Test Video", 26)
    s.shapes.add_picture(os.path.join(fig, "fig03_per_class_baselines.png"), Inches(1.45), Inches(1.95), width=Inches(7.1))
    s.shapes.add_picture(os.path.join(fig, "fig04_per_sequence_baselines.png"), Inches(X0 + 0.3), Inches(4.4), width=Inches(3.3))
    textbox(s, 4.3, 4.55, 5.05, 2.1, [
        "Plot 1: big, clear parts (shaft, clasper, kidney) score well for all models; ours leads most on thin parts such as thread. The suturing needle is 0.000 for every model: it is rare in training.",
        f"Plot 2: ours is ahead on each of the four test videos (scores {V['SeqFourMiou']} to {V['SeqThreeMiou']}); the video matters more than the model.",
    ], size=11.5, space_after=5, line=1.0)
    notes(s, "Top: each group of bars is one class, taller is better. Bottom left: score on each of the four test videos; ours is highest in every group, but notice how much the score depends on the video. The needle is a known weakness for all three models.")
    new[10] = s

    # ---------- 10. plots: risk coverage and calibration (one slide)
    s = new_slide(prs, layout, num_proto, "Plot 3 and Plot 4: Spotting Bad Frames, Honest Confidence", 24)
    s.shapes.add_picture(os.path.join(fig, "fig01_risk_coverage_test.png"), Inches(X0 + 0.05), Inches(1.95), height=Inches(4.7))
    s.shapes.add_picture(os.path.join(fig, "fig06_reliability.png"), Inches(4.45), Inches(1.95), width=Inches(5.0))
    textbox(s, 4.45, 3.9, 4.95, 2.8, [
        f"Plot 3 (left): keep only the frames the system is most sure about and the error should fall. Our checker (AURC {V['GateAurcTest']}) is not better than simple max-softmax ({V['MsAurcTest']}); lower is better. Max-softmax is already one of its nine inputs, and five re-trainings give AUROC {V['GaSeedAurocMean']} (spread {V['GaSeedAurocSd']}).",
        f"Plot 4 (right): after one adjustment the pixel confidence is more honest (error {V['CalEceBefore']} down to {V['CalEceAfter']}). The checker ranks frames well but its scores run too high ({V['GateScoreTest']} against {V['GateAchTest']} achieved): trust its ranking, not its level.",
    ], size=11, space_after=5, line=1.0)
    notes(s, "Left: a risk-coverage plot. Dropping the least sure frames should make the error fall; a perfect oracle is the lowest dotted line. Our checker does this, but a very simple confidence score does it just as well. Right: a perfectly honest system lies on the diagonal. The checker orders frames sensibly but is over-optimistic about the actual score, so its threshold is not a guarantee on new videos.")
    new[12] = s

    # ---------- 11. new data, what worked, what did not, conclusion (one slide)
    s = new_slide(prs, layout, num_proto, "Results Summary: New Data, What Worked, What Did Not", 24)
    tiles = [(V["ZsIou"], "EndoVis 2017, never seen", GREEN_F, GREEN), (V["ZsIouRef"], "our 2018 test, same task", BLUE_F, BLUE), (f"{V['ZsBelow']}%", "of 2017 frames score below 0.5", GREY_F, GREY)]
    for k, (big, small, f, e) in enumerate(tiles):
        box(s, X0 + 0.1 + k * 2.9, 1.95, 2.7, 0.95, big, small, f, e, 24, 11)
    textbox(s, X0 + 0.05, 2.93, 8.6, 0.55, [f"Scored once on {V['ZsFrames']} frames of {V['ZsSeqs']} videos, no retraining. Only \"tool or not tool\": the parts where our model is weakest are left out, so this does not prove it works everywhere."], size=11, bullets=False, color=GREY)
    box(s, X0 + 0.05, 3.5, 4.25, 0.36, "What worked", None, GREEN_F, GREEN, 12.5)
    box(s, 4.95, 3.5, 4.25, 0.36, "What did not", None, ORANGE_F, ORANGE, 12.5)
    textbox(s, X0 + 0.05, 3.9, 4.3, 2.0, [
        f"Beats both other models we trained ({V['BlOursMiou']} against {V['BlSegMiou']} and {V['BlUnetMiou']}).",
        f"Holds up on new data: {V['ZsIou']} on EndoVis 2017.",
        f"More honest confidence: error {V['CalEceBefore']} down to {V['CalEceAfter']}.",
        f"Fixing only the doubtful {V['FlaggedPct']}% of frames: {V['CostSelNoA']} ms instead of {V['CostAllTwo']} ms, same official score.",
    ], size=11, space_after=3, line=1.0)
    textbox(s, 4.95, 3.9, 4.3, 2.0, [
        f"The checker is no better than a simple confidence score ({V['GateAurcTest']} against {V['MsAurcTest']}), and its scores are too high. Its risk target was met out of fold, not on test ({V['RiskAcceptedTest']} against {V['Alpha']}).",
        "Comparing the two cameras' outlines did not clear the pre-set test we fixed in advance, so the second camera is not used.",
        "The fixes barely change the official score: the measured gain is mostly invented classes removed, which the official score cannot see.",
        "The needle is still at 0.000.",
    ], size=11, space_after=3, line=1.0)
    label(s, X0 + 0.05, 6.0, 8.6, f"Conclusion: a segmenter ({V['BaseMiou']}) ahead of two models trained the same way, and a self-check that saves time without lowering the official score. Next: a re-labelling step, needle training, more videos.", 11.5, INK, True)
    notes(s, "Top: the new-data test. We feared a big drop and did not see one, but only tool-versus-background was tested. Being open about the negative results is a strength: the segmenter is good and the numbers are honest, but the extra checking machinery is not shown to beat simple baselines. Close with the open items.")
    new[14] = s

    # ---------- order
    keep = {0: old[0], 2: old[2], 11: old[11], 12: old[12]}
    for sl in old:
        if sl not in keep.values():
            delete_slide(prs, sl)
    lst = prs.slides._sldIdLst
    id_of = {}
    for sid in list(lst):
        id_of[prs.part.related_part(sid.rId)] = sid
    order = [keep[0], new[2], keep[2]] + [new[k] for k in (5, 6, 7, 8, 9, 10, 12, 14)] + [keep[11], keep[12]]
    els = [id_of[sl.part] for sl in order]
    for el in els:
        lst.remove(el)
    for el in els:
        lst.append(el)
    prs.save(cur)
    print("deck written:", cur, "slides:", len(prs.slides))


if __name__ == "__main__":
    main()

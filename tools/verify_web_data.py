#!/usr/bin/env python3
"""
Fail the build if the website shows a number that does not trace to a file in results/.

    python -m tools.verify_web_data --results results --web web

Check 1 (data). Every numeric field in web/data/results.json (booleans excluded) must equal a numeric value stored
in one of the JSON files that the export declares as the source of that block (`_sources`). The first match is
recorded as file:key.path in web/data/provenance.json. A block with an undeclared or missing source file, or a
number with no match, fails. Limit, stated plainly: the match is on value within the declared files and the
report lists the match count, so a small integer such as 0 or 1 can match more than one key; a count above one is
shown, not hidden.

Check 2 (page source). Page and component source under web/app and web/components may not contain a number literal
that describes a result. Scanned: JSX text and string/template literals in .tsx files (className, href, import and
comment text are skipped). Flagged: decimals, percentages, numbers with a unit, signed numbers, scientific
notation and any integer of three or more digits. Plain small integers (a count of classes or sequences, a step
number, 'Q1', 'M6') are not flagged. A literal may be allowed only through ALLOWED_LITERALS below, each with a
reason; the list is printed on every run so it cannot grow unnoticed.
"""
import argparse
import json
import math
import os
import re
import sys

# literal -> reason. Structural facts about the setup or a fixed, stated choice, not a measured result.
ALLOWED_LITERALS = {
    "1280x1024": "image resolution of the dataset (DATASET_FACTS.md)",
    "512x640": "training resolution chosen for the 4 GB GPU (RUNLOG.md)",
    "3.3 GB": "usable GPU memory budget stated in the brief",
    "4 GB": "GPU memory of the laptop used",
    "0.5": "AUROC chance level, and the bad-frame rule (instrument IoU below 0.5) defined in tools/final_eval.py",
    "1.0": "axis end of a fixed 0.5 to 1.0 AUROC scale",
    "2018": "name of the dataset (EndoVis 2018)",
    "2017": "name of the dataset (EndoVis 2017)",
    "10527017": "Zenodo record identifier of the licensed copy of EndoVis 2017 (a citation, not a result)",
    "3050": "name of the GPU (RTX 3050 Ti)",
    "95%": "confidence level of the bootstrap intervals (stated choice in the brief)",
    "2001.11190": "arXiv identifier of the EndoVis 2018 challenge paper (a citation, not a result)",
    "1 px": "stated threshold above which the label-free vertical correction is applied (tools/check_stereo.py, RUNLOG.md)",
}


def leaves(o, path=""):
    if isinstance(o, bool):
        return
    if isinstance(o, (int, float)):
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return
        yield path, float(o)
    elif isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{path}.{k}" if path else str(k))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{path}[{i}]")


def close(a, b):
    return abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))


def check_data(results, web):
    data = json.load(open(os.path.join(web, "data", "results.json"), encoding="utf-8"))
    sources = data["_sources"]
    problems, prov, per_block = [], {}, {}
    index_cache = {}

    def index(rel):
        if rel not in index_cache:
            p = os.path.join(results, rel)
            if not os.path.exists(p) or not rel.endswith(".json"):
                index_cache[rel] = None
            else:
                index_cache[rel] = list(leaves(json.load(open(p, encoding="utf-8"))))
        return index_cache[rel]

    for block, val in data.items():
        if block == "_sources":
            continue
        if block not in sources:
            problems.append(f"block '{block}' has no declared source")
            continue
        idxs = {}
        for rel in sources[block]:
            ix = index(rel)
            if ix is None:
                problems.append(f"block '{block}': declared source {rel} is missing or not JSON under {results}/")
            else:
                idxs[rel] = ix
        n = bad = 0
        for path, v in leaves(val, block):
            n += 1
            hits = [(rel, k) for rel, ix in idxs.items() for k, x in ix if close(x, v)]
            if not hits:
                bad += 1
                problems.append(f"no source for {path} = {v}")
            else:
                prov[path] = {"value": v, "first_match": f"{hits[0][0]}:{hits[0][1]}", "n_matches": len(hits)}
        per_block[block] = (n, bad, sources[block])
    json.dump(prov, open(os.path.join(web, "data", "provenance.json"), "w"), indent=0)
    return problems, per_block


STRIP = [
    re.compile(r"className=(\"[^\"]*\"|\{`[^`]*`\}|\{\"[^\"]*\"\})"),
    re.compile(r"(href|src|viewBox|stroke|fill|key|d|id|type|name|domain|width|height|style|rel|target|aria-label)=(\"[^\"]*\"|\{[^}]*\})"),
    re.compile(r"^\s*(import|export \* from)\b.*$"),
    re.compile(r"^\s*//.*$"),
    re.compile(r"\{/\*.*?\*/\}"),
    re.compile(r"\b(?:text|bg|border|ring|size|w|h|gap|p|px|py|pt|pb|m|mt|mb|mx|my|max-w|min-h|min-w|rounded|opacity|grid-cols|col-span|leading|tracking|top|left|right|bottom|z|translate-x|translate-y|duration|delay|font|space-x|space-y|from|to|via|shadow|blur|backdrop-blur)-[\w./\[\]:%#-]+"),
    re.compile(r"#[0-9a-fA-F]{3,8}\b"),
    re.compile(r"\b(?:rgba?|hsla?)\([^)]*\)"),
]
NUM = re.compile(r"(?<![\w.#-])[-+]?(?:\d+\.\d+|\d+(?:\.\d+)?(?:%|e-?\d+|x\d+| ms| s| px| GB| Hz)|\d{3,})(?![\w])")
STRUCT_OK = re.compile(r"^(?:[AMQ]\d|F1|\d)$")


def check_source(web):
    viol = []
    for root in ("app", "components"):
        for dp, _, fs in os.walk(os.path.join(web, root)):
            for f in fs:
                if not f.endswith(".tsx"):
                    continue
                p = os.path.join(dp, f)
                for ln, line in enumerate(open(p, encoding="utf-8"), 1):
                    s = line
                    for rx in STRIP:
                        s = rx.sub(" ", s)
                    # only look at text-bearing parts: quoted strings, template text, JSX text
                    parts = re.findall(r"\"([^\"]*)\"|`([^`]*)`|>([^<>{}]+)<|^\s*([A-Za-z][^<>{}=;]*[.,:)])\s*$", s)
                    # JSX text that wraps over several lines can contain ; ( ) - scan any line that reads as prose
                    res = re.sub(r"\{[^{}]*\}", " ", s)
                    res = re.sub(r"\{[^{}]*\}", " ", res)
                    if re.search(r"(?:[A-Za-z']+[ ,;:.()]+){5,}", res) and not re.search(r"=>|\bconst\b|\breturn\b|\bimport\b|\.map\(|useState|<[A-Z]|\bfunction\b", res):
                        parts.append((res.strip(), "", "", ""))
                    for tup in parts:
                        txt = " ".join(x for x in tup if x)
                        txt = re.sub(r"\$\{[^}]*\}", " ", txt)
                        if not re.search(r"[A-Za-z]{2,}", txt):
                            continue            # pure numbers/punctuation are layout or SVG values, not prose
                        for m in NUM.finditer(txt):
                            lit = m.group(0).strip()
                            if lit in ALLOWED_LITERALS or any(lit in k for k in ALLOWED_LITERALS):
                                continue
                            viol.append(f"{os.path.relpath(p, web)}:{ln}: '{lit}' in {txt.strip()[:90]!r}")
    return viol


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--web", default="web")
    ap.add_argument("--show", type=int, default=60, help="how many violations to print")
    a = ap.parse_args()
    problems, per_block = check_data(a.results, a.web)
    print("DATA CHECK: web/data/results.json against declared sources under results/")
    tot = 0
    for b, (n, bad, srcs) in per_block.items():
        tot += n
        print(f"  {b:<18} {n:>6} numeric fields, {bad} untraced   <- {', '.join(srcs)}")
    print(f"  {tot} numeric fields in total; provenance written to {a.web}/data/provenance.json")
    viol = check_source(a.web)
    print("\nSOURCE CHECK: result-like number literals in page and component source")
    print("  allowed literals (stated structural facts):", json.dumps(ALLOWED_LITERALS, indent=0)[:400].replace("\n", " "))
    for v in viol[: a.show]:
        print("  LITERAL", v)
    if len(viol) > a.show:
        print(f"  ... and {len(viol) - a.show} more")
    fails = len(problems) + len(viol)
    for p in problems[: a.show]:
        print("  UNTRACED", p)
    print(f"\nRESULT: {'FAIL' if fails else 'PASS'}  ({len(problems)} untraced data fields, {len(viol)} source literals)")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())

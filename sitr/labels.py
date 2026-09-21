"""
EndoVis 2018 (Robotic Scene Segmentation) label handling.

The class table is DERIVED FROM YOUR COPY OF THE DATA, never hardcoded: a wrong
palette does not raise, it silently destroys the labels, so this module reads
every `labels.json` in the download and unions them.

Three properties of the real release are handled explicitly, because each one
breaks a naive reader:

1. Two different JSON schemas ship in the same download.
     test_data/labels.json            {"classes": [{name, active, color:[R,G,B,A], classid}]}
     miccai_challenge_release_*/labels.json   [{name, color:[R,G,B], classid}]
   The alpha channel (128) must be dropped; the label PNGs are RGB.

2. Each release carries its own labels.json listing only the classes that occur
   in its procedures. The union over all releases is the true class table.
   The official `run.py` shipped with the test data declares NUM_CLASSES = 12,
   which is the expected size of that union.

3. `__MACOSX/` shadow trees contain AppleDouble `._*` files that look like PNGs
   and JSONs but are 1 KB of resource-fork junk. They are excluded everywhere.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IGNORE_INDEX = 255

#: Directory names that are never dataset content.
JUNK_DIRS = frozenset({"__macosx", ".git", ".ipynb_checkpoints", "__pycache__"})

#: Official class count declared by the challenge's own run.py.
OFFICIAL_NUM_CLASSES = 12


def is_junk_path(path: str) -> bool:
    """True for __MACOSX trees, AppleDouble ._* files and .DS_Store."""
    parts = os.path.normpath(path).split(os.sep)
    for p in parts:
        if p.lower() in JUNK_DIRS:
            return True
        if p.startswith("._") or p == ".DS_Store":
            return True
    return False


def prune_junk(dirnames: List[str]) -> None:
    """In-place filter for os.walk's dirnames, to stop descent into junk."""
    dirnames[:] = [d for d in dirnames
                   if d.lower() not in JUNK_DIRS and not d.startswith("._")]


# Substrings deciding whether a class is a surgical INSTRUMENT, i.e. part of
# Omega^fg in eqs. (6)-(10).  EndoVis 2018 annotates anatomy too, so
# "not background" is emphatically not "instrument".
INSTRUMENT_NAME_PATTERNS: Tuple[str, ...] = (
    "instrument", "shaft", "wrist", "clasper", "jaw", "clamp", "needle",
    "thread", "suction", "probe", "ultrasound", "scissor", "grasper",
)

#: Names that match an instrument pattern but are anatomy / background.
NON_INSTRUMENT_OVERRIDES: Tuple[str, ...] = (
    "background", "tissue", "kidney", "parenchyma", "intestine", "bowel",
    "fascia", "covered",
)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")


def is_instrument_name(name: str) -> bool:
    n = _norm(name)
    if any(p in n for p in NON_INSTRUMENT_OVERRIDES):
        return False
    return any(p in n for p in INSTRUMENT_NAME_PATTERNS)


@dataclass
class ClassEntry:
    id: int                                  # canonical contiguous id
    name: str
    color: Optional[Tuple[int, int, int]]    # RGB (alpha dropped), or None
    raw_id: Optional[int]                    # classid from labels.json
    instrument: bool
    sources: Tuple[str, ...] = ()            # which labels.json files listed it
    aliases: Tuple[str, ...] = ()            # other names releases used for it


@dataclass
class ClassMap:
    classes: List[ClassEntry]
    encoding: str                            # "rgb" | "id"
    source: str = ""

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.classes]

    @property
    def instrument_ids(self) -> List[int]:
        return [c.id for c in self.classes if c.instrument]

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"encoding": self.encoding, "source": self.source,
                       "classes": [asdict(c) for c in self.classes]}, f, indent=2)

    @staticmethod
    def from_json(path: str) -> "ClassMap":
        with open(path) as f:
            payload = json.load(f)
        classes = [
            ClassEntry(
                id=int(c["id"]), name=str(c["name"]),
                color=tuple(c["color"]) if c.get("color") is not None else None,
                raw_id=None if c.get("raw_id") is None else int(c["raw_id"]),
                instrument=bool(c["instrument"]),
                sources=tuple(c.get("sources", ())),
                aliases=tuple(c.get("aliases", ())),
            )
            for c in payload["classes"]
        ]
        return ClassMap(classes, payload["encoding"], payload.get("source", path))

    # ------------------------------------------------------------ decoding
    def rgb_lut(self) -> np.ndarray:
        """24-bit LUT: (r<<16 | g<<8 | b) -> canonical id, else IGNORE_INDEX."""
        if self.encoding != "rgb":
            raise ValueError("rgb_lut() requires an RGB-encoded class map")
        lut = np.full(1 << 24, IGNORE_INDEX, dtype=np.uint8)
        for c in self.classes:
            if c.color is None:
                raise ValueError(f"class '{c.name}' has no colour")
            r, g, b = (int(v) for v in c.color)
            lut[(r << 16) | (g << 8) | b] = c.id
        return lut

    def id_lut(self) -> np.ndarray:
        lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
        for c in self.classes:
            raw = c.id if c.raw_id is None else c.raw_id
            if not 0 <= int(raw) < 256:
                raise ValueError(f"raw id {raw} for '{c.name}' outside uint8 range")
            lut[int(raw)] = c.id
        return lut


# --------------------------------------------------------------------------- #
# labels.json parsing
# --------------------------------------------------------------------------- #
_COLOR_KEYS = ("color", "colour", "rgb", "rgba")
_ID_KEYS = ("classid", "class_id", "id", "label", "value")
_NAME_KEYS = ("name", "classname", "class_name", "label_name")


def _entry_from_dict(d: dict):
    name = next((str(d[k]) for k in _NAME_KEYS if k in d), None)
    if name is None:
        return None
    color = next((d[k] for k in _COLOR_KEYS if k in d and d[k] is not None), None)
    raw_id = next((d[k] for k in _ID_KEYS if k in d and d[k] is not None), None)
    if color is not None:
        # RGBA -> RGB: the release's test_data/labels.json carries alpha=128,
        # but the label PNGs themselves are 3-channel.
        color = [int(v) for v in list(color)[:3]]
    if raw_id is not None:
        try:
            raw_id = int(raw_id)
        except (TypeError, ValueError):
            raw_id = None
    active = d.get("active", True)
    return name, color, raw_id, bool(active)


def parse_labels_json(path: str):
    """Return [(name, rgb|None, raw_id|None, active), ...] for one labels.json."""
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "classes" in raw:
        raw = raw["classes"]

    out = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                parsed = _entry_from_dict(item)
                if parsed is not None:
                    out.append(parsed)
    elif isinstance(raw, dict):
        for name, val in raw.items():
            if isinstance(val, dict):
                parsed = _entry_from_dict({**val, "name": val.get("name", name)})
                if parsed is not None:
                    out.append(parsed)
            elif isinstance(val, (list, tuple)) and len(val) >= 3:
                out.append((str(name), [int(v) for v in list(val)[:3]], None, True))
            elif isinstance(val, int):
                out.append((str(name), None, int(val), True))
    if not out:
        raise ValueError(f"could not parse any class entries from {path}")
    return out


def find_labels_json(root: str) -> List[str]:
    """Every real labels.json under `root`, shallowest first, junk excluded."""
    found: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        prune_junk(dirnames)
        if is_junk_path(dirpath):
            continue
        for fn in filenames:
            if fn.startswith("._"):
                continue
            if fn.lower() in ("labels.json", "label.json", "classes.json"):
                found.append(os.path.join(dirpath, fn))
    found.sort(key=lambda p: (p.count(os.sep), p))
    return found


def build_class_map(root: str,
                    instrument_names: Optional[Sequence[str]] = None,
                    include_inactive: bool = True) -> ClassMap:
    """Union the class tables of every labels.json in the download.

    Each release lists only the classes present in its own procedures, so the
    union is the real class table.  Colour conflicts (same name, different
    colour, or same colour, different name) are raised rather than silently
    resolved — they would corrupt every mask decoded afterwards.
    """
    paths = find_labels_json(root)
    if not paths:
        raise FileNotFoundError(
            f"No labels.json found under {root!r}. Run "
            "`python -m tools.audit_dataset --root <root> --scan-colors` to list "
            "the label values actually present, then write the mapping explicitly."
        )

    by_name: Dict[str, dict] = {}
    for path in paths:
        for name, color, raw_id, active in parse_labels_json(path):
            if not active and not include_inactive:
                continue
            key = _norm(name)
            rec = by_name.setdefault(key, {"name": key, "color": None,
                                           "raw_id": None, "sources": []})
            rec["sources"].append(os.path.relpath(path, root))
            if color is not None:
                if rec["color"] is not None and tuple(rec["color"]) != tuple(color):
                    raise ValueError(
                        f"class '{key}' has conflicting colours across "
                        f"labels.json files: {rec['color']} vs {color} "
                        f"(sources: {rec['sources']}). Resolve before training."
                    )
                rec["color"] = color
            if raw_id is not None and rec["raw_id"] is None:
                rec["raw_id"] = raw_id

    entries = list(by_name.values())
    has_color = all(e["color"] is not None for e in entries)
    encoding = "rgb" if has_color else "id"

    merged_report: List[str] = []
    if encoding == "rgb":
        # Releases disagree on names for identical colours (EndoVis 2018 calls
        # the same (124,155,5) region "small-intestine" in one labels.json and
        # "intestine" in another). The colour is what is actually encoded in
        # the PNGs, so those are the SAME pixels and must become one class --
        # keeping them separate would split one region across two class ids and
        # make every IoU on it wrong. The canonical name is the one the most
        # releases use, with the more specific (longer) name breaking ties.
        by_color: Dict[Tuple[int, int, int], List[dict]] = {}
        for e in entries:
            by_color.setdefault(tuple(e["color"]), []).append(e)

        entries = []
        for color, group in by_color.items():
            if len(group) == 1:
                entries.append(group[0])
                continue
            group.sort(key=lambda g: (-len(set(g["sources"])), -len(g["name"]),
                                      g["name"]))
            keep, dropped = group[0], group[1:]
            keep["aliases"] = sorted({d["name"] for d in dropped})
            for d in dropped:
                keep["sources"].extend(d["sources"])
                if keep["raw_id"] is None:
                    keep["raw_id"] = d["raw_id"]
            merged_report.append(
                f"colour {color}: merged {[d['name'] for d in dropped]} "
                f"into '{keep['name']}' (same pixels, different release naming)")
            entries.append(keep)

        # The reverse case is NOT mergeable: one name with two colours means
        # the encoding itself is ambiguous and no mask can be decoded safely.
        name_colors: Dict[str, Tuple[int, int, int]] = {}
        for e in entries:
            c = tuple(e["color"])
            if e["name"] in name_colors and name_colors[e["name"]] != c:
                raise ValueError(
                    f"class '{e['name']}' has two different colours: "
                    f"{name_colors[e['name']]} and {c}. The label encoding is "
                    "ambiguous; resolve before training.")
            name_colors[e["name"]] = c

    if merged_report:
        print("class-table merges (same colour, different release naming):")
        for m in merged_report:
            print(f"  ! {m}")

    # background first (canonical id 0), then by the classid the release gave.
    entries.sort(key=lambda e: (0 if "background" in e["name"] else 1,
                                e["raw_id"] if e["raw_id"] is not None else 1 << 30,
                                e["name"]))

    override = {_norm(n) for n in (instrument_names or [])}
    classes: List[ClassEntry] = []
    for i, e in enumerate(entries):
        inst = e["name"] in override if override else is_instrument_name(e["name"])
        classes.append(ClassEntry(
            id=i, name=e["name"],
            color=tuple(e["color"]) if e["color"] is not None else None,
            raw_id=e["raw_id"], instrument=bool(inst) and i != 0,
            sources=tuple(sorted(set(e["sources"]))),
            aliases=tuple(e.get("aliases", ())),
        ))

    cm = ClassMap(classes, encoding, source="; ".join(
        os.path.relpath(p, root) for p in paths))
    if not cm.instrument_ids:
        raise ValueError(
            "No instrument classes identified. Pass --instrument-classes "
            f"explicitly. Available names: {cm.names}"
        )
    return cm


# --------------------------------------------------------------------------- #
# Task presets
# --------------------------------------------------------------------------- #
def task_remap(cm: ClassMap, task: str) -> Tuple[np.ndarray, List[str], List[int]]:
    """Return (LUT over canonical ids, target class names, instrument target ids).

    full        every annotated class — the challenge's own 12-class task
    binary      background vs any instrument
    parts       background + shaft / wrist / clasper (articulated-part setting)
    instruments background + each instrument class kept separate, all anatomy
                merged into background. This is the setting the framework is
                actually about: instrument identity matters for Stage 5, anatomy
                classes do not, and merging them removes ~40% of the label noise
                from classes the recovery mechanism will never touch.
    """
    n = cm.num_classes
    lut = np.full(n, IGNORE_INDEX, dtype=np.uint8)

    if task == "full":
        lut[:] = np.arange(n, dtype=np.uint8)
        return lut, list(cm.names), list(cm.instrument_ids)

    if task == "binary":
        lut[:] = 0
        for c in cm.classes:
            if c.instrument:
                lut[c.id] = 1
        return lut, ["background", "instrument"], [1]

    if task == "instruments":
        lut[:] = 0
        names, inst_ids = ["background"], []
        for c in cm.classes:
            if c.instrument:
                lut[c.id] = len(names)
                inst_ids.append(len(names))
                names.append(c.name)
        return lut, names, inst_ids

    if task == "parts":
        part_keys = ("shaft", "wrist", "clasper", "jaw")
        names, inst_ids = ["background"], []
        lut[:] = 0
        for key in part_keys:
            matched = [c for c in cm.classes if key in c.name and c.instrument]
            if not matched:
                continue
            tid = len(names)
            names.append(key)
            inst_ids.append(tid)
            for c in matched:
                lut[c.id] = tid
        if len(names) == 1:
            raise ValueError(
                f"task='parts' found no shaft/wrist/clasper classes in {cm.names}")
        return lut, names, inst_ids

    raise ValueError(f"unknown task {task!r} (full|binary|parts|instruments)")


# --------------------------------------------------------------------------- #
# label image -> canonical ids
# --------------------------------------------------------------------------- #
def decode_label_image(img: np.ndarray, cm: ClassMap,
                       rgb_lut: Optional[np.ndarray] = None,
                       id_lut: Optional[np.ndarray] = None) -> np.ndarray:
    """Decode a label PNG (loaded as RGB or grayscale) to canonical class ids."""
    if img.ndim == 3 and img.shape[2] >= 3:
        if cm.encoding == "id":
            return (id_lut if id_lut is not None else cm.id_lut())[img[:, :, 0]]
        rgb = img[:, :, :3].astype(np.uint32)
        keys = (rgb[:, :, 0] << 16) | (rgb[:, :, 1] << 8) | rgb[:, :, 2]
        lut = rgb_lut if rgb_lut is not None else cm.rgb_lut()
        return lut[keys]

    if cm.encoding == "rgb":
        raise ValueError(
            "class map is RGB-encoded but this label image is single-channel; "
            "run tools/audit_dataset.py --scan-colors to inspect the data.")
    return (id_lut if id_lut is not None else cm.id_lut())[img]


def scan_label_colors(label_paths: Sequence[str], limit: int = 60) -> Dict:
    """Distinct values actually present in a sample of label files."""
    colors: Dict[Tuple[int, int, int], int] = {}
    ids: Dict[int, int] = {}
    multichannel = False
    scanned = 0
    for p in label_paths[:limit]:
        if is_junk_path(p):
            continue
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        scanned += 1
        if img.ndim == 3 and img.shape[2] >= 3:
            multichannel = True
            bgr = img[:, :, :3].reshape(-1, 3)
            uniq, cnt = np.unique(bgr, axis=0, return_counts=True)
            for u, c in zip(uniq, cnt):
                key = (int(u[2]), int(u[1]), int(u[0]))   # BGR -> RGB
                colors[key] = colors.get(key, 0) + int(c)
        else:
            uniq, cnt = np.unique(img, return_counts=True)
            for u, c in zip(uniq, cnt):
                ids[int(u)] = ids.get(int(u), 0) + int(c)
    return {"multichannel": multichannel,
            "rgb_values": sorted(colors.items(), key=lambda kv: -kv[1]),
            "id_values": sorted(ids.items(), key=lambda kv: -kv[1]),
            "files_scanned": scanned}

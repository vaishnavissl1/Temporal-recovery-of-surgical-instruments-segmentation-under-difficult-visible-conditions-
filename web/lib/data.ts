import raw from "@/data/results.json";

// Everything on the site comes from this one file, written by tools/export_web_data.py from results/*.json.
export const R = raw;
export type Model = (typeof raw.models)[number];

export const pct = (v: number, d = 1) => `${v.toFixed(d)}%`;
export const f3 = (v: number) => v.toFixed(3);
export const f4 = (v: number) => v.toFixed(4);
export const signed = (v: number, d = 4) => `${v >= 0 ? "+" : ""}${v.toFixed(d)}`;

export const model = (id: string): Model => raw.models.find((m) => m.id === id)!;

export const SHORT: Record<string, string> = {
  M1: "Segmentation only",
  M2: "+ stereo evidence",
  M3: "+ temporal only",
  M4: "Blind stereo-temporal fusion",
  M5: "Fusion + refinement",
  M6: "Selective choice (ours)",
};

export const SET_LABEL: Record<string, string> = {
  paper4: "Original 4 indicators",
  extended: "9 indicators (no stereo)",
  extended_stereo: "9 + 3 stereo indicators",
  nontemporal: "Without temporal indicators",
  nontemporal_stereo: "Without temporal + stereo",
  stereo3: "Stereo indicators alone",
};

export const PAIR_LABEL: Record<string, string> = {
  "extended_stereo vs extended": "Add stereo to the 9 indicators",
  "nontemporal_stereo vs nontemporal": "Add stereo when temporal is absent",
  "nontemporal_stereo vs extended": "Stereo instead of temporal",
  "stereo3 vs nontemporal": "Stereo alone vs non-temporal",
  "extended vs nontemporal": "Add temporal indicators",
};

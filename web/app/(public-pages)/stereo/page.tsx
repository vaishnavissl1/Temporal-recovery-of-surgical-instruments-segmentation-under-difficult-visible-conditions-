"use client";

import { useState } from "react";
import { Callout, DataTable, HBars, LinkButton, PageHero, Section, Toggle } from "@/components/ui";
import { PAIR_LABEL, R, SET_LABEL, f3, signed } from "@/lib/data";

export default function StereoPage() {
  const [tab, setTab] = useState<"geometry" | "gate">("geometry");
  const [split, setSplit] = useState<"oof" | "test">("oof");
  const S = R.stereo;
  const gate = split === "oof" ? S.gate_oof : S.gate_test;
  const paired = split === "oof" ? S.paired_oof : S.paired_test;
  const trainSC = S.self_calibration.filter((x) => x.seq.startsWith("train"));
  const applied = trainSC.filter((x) => x.applied).length;
  const testApplied = S.self_calibration.filter((x) => x.seq.startsWith("test") && x.applied).length;
  const verdictText = (v: string) => (v.includes("TIED") ? "No difference" : v.includes("better") || v.includes("BETTER") ? "Better" : v.includes("worse") || v.includes("WORSE") ? "Worse" : "See interval");

  return (
    <main>
      <PageHero
        eyebrow="Stereo evidence"
        title="Stereo works geometrically. Did it help the gate?"
        lead="Two separate questions: can we rectify the stereo pairs and match them, and does the resulting disparity evidence improve failure detection?"
      >
        <LinkButton href="/selector/">What the selector did with it</LinkButton>
      </PageHero>

      <Section title="Pick a question" eyebrow="Two views">
        <Toggle
          options={[
            { id: "geometry", label: "1. Is the geometry usable?" },
            { id: "gate", label: "2. Does it improve failure detection?" },
          ]}
          value={tab}
          onChange={setTab}
          label="Question"
        />

        {tab === "geometry" && (
          <div className="mt-8 flex flex-col gap-8">
            <Callout tone="good" title="Verdict: calibrated stereo is viable">
              None of the three stop conditions tripped. After rectification the mean vertical error is small, and {pct1(S.lr_consistent_pct.refined)} of pixels pass the left-right consistency check ({pct1(S.instrument_lr_pct.refined)} on instrument pixels).
              But the shipped calibration leaves a several-pixel vertical tilt on some training sequences. A label-free per-sequence correction was applied to {applied} of {trainSC.length} training sequences (it implied more than 1 px) and to {testApplied} of the 4 test sequences.
            </Callout>
            <div className="bg-white border border-zinc-200 rounded-xl p-5 md:p-8">
              <h3 className="text-lg font-medium text-zinc-900">Vertical error after rectification, per sequence</h3>
              <p className="text-xs text-zinc-500 mb-6">Mean absolute vertical disparity of matched features (pixels, lower is better). Dark = calibration as shipped, green = with the label-free correction.</p>
              <div className="flex flex-col gap-2.5">
                {S.pairs.map((p) => (
                  <div key={p.seq} className="grid grid-cols-[6.5rem_1fr_5.5rem] gap-3 items-center text-xs">
                    <span className="text-zinc-700">{p.seq}</span>
                    <div className="flex flex-col gap-1">
                      <div className="h-2.5 bg-zinc-100 rounded"><div className="h-full rounded bg-zinc-800" style={{ width: `${Math.min(100, (p.calib_only / 13) * 100)}%` }} /></div>
                      <div className="h-2.5 bg-zinc-100 rounded"><div className="h-full rounded bg-emerald-500" style={{ width: `${Math.min(100, (p.self_calibrated / 13) * 100)}%` }} /></div>
                    </div>
                    <span className="tabular-nums text-right text-zinc-500">{p.calib_only.toFixed(1)} &rarr; {p.self_calibrated.toFixed(1)}</span>
                  </div>
                ))}
              </div>
            </div>
            <DataTable
              head={["Sequence", "Largest implied correction (px)", "Applied"]}
              align={["l", "r", "r"]}
              rows={S.self_calibration.map((x) => [x.seq, x.max_abs_px.toFixed(2), x.applied ? "yes" : "no (under 1 px)"])}
            />
          </div>
        )}

        {tab === "gate" && (
          <div className="mt-8 flex flex-col gap-8">
            <Toggle
              options={[
                { id: "oof", label: "Out-of-fold training (2235 frames)" },
                { id: "test", label: "Official test (997 frames)" },
              ]}
              value={split}
              onChange={setSplit}
              label="Split"
            />
            <div className="bg-white border border-zinc-200 rounded-xl p-5 md:p-8">
              <h3 className="text-lg font-medium text-zinc-900">Failure detection by indicator set (AUROC, higher is better)</h3>
              <p className="text-xs text-zinc-500 mb-6">0.5 is chance. Scale 0.5 to 1.0.</p>
              <HBars
                domain={[0.5, 1]}
                fmt={f3}
                bars={gate.map((g) => ({ label: SET_LABEL[g.set] ?? g.set, value: g.auc, color: g.set.includes("stereo") ? "#0ea5e9" : "#18181b" }))}
              />
            </div>
            <DataTable
              head={["Comparison", "AUROC change", "95% interval", "Reading"]}
              align={["l", "r", "r", "l"]}
              rows={paired.map((p) => [PAIR_LABEL[p.pair] ?? p.pair, signed(p.auc_diff), `${signed(p.auc_lo)} to ${signed(p.auc_hi)}`, verdictText(p.verdict)])}
            />
            <p className="text-xs text-zinc-500 -mt-4">Intervals are paired bootstraps over sequences ({split === "test" ? "only 4 sequences, so read it as a warning, not a test" : "15 sequences"}).</p>
            <div className="grid md:grid-cols-2 gap-4">
              <Callout tone="bad" title="Stereo indicators do not add anything">
                Adding them to the 9-indicator gate changes AUROC by {signed(paired[0].auc_diff)} with an interval that spans zero. Stereo indicators alone reach only {f3(gate.find((g) => g.set === "stereo3")!.auc)}.
              </Callout>
              <Callout tone="good" title="Temporal cues do the work">
                Removing the temporal indicator costs the most AUROC of any single indicator. The brief expected stereo to add +0.05 to +0.15; the measurement is about zero.
              </Callout>
            </div>
            <div className="bg-white border border-zinc-200 rounded-xl p-5 md:p-8">
              <h3 className="text-lg font-medium text-zinc-900">Which indicators matter (out-of-fold)</h3>
              <p className="text-xs text-zinc-500 mb-6">AUROC lost when that indicator is removed. Negative means removing it helped.</p>
              <HBars fmt={(v) => signed(v, 3)} bars={[...S.leave_one_out].sort((a, b) => b.auc_drop - a.auc_drop).map((x) => ({ label: x.indicator, value: x.auc_drop, color: x.auc_drop >= 0 ? "#10b981" : "#f43f5e" }))} />
            </div>
          </div>
        )}
      </Section>
    </main>
  );
}

function pct1(v: number) {
  return `${v.toFixed(1)}%`;
}

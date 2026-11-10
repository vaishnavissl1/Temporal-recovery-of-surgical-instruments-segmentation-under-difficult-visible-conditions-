"use client";

import { useState } from "react";
import { Callout, DataTable, HBars, LinkButton, PageHero, Section, Toggle } from "@/components/ui";
import { R, f3, f4, pct, signed } from "@/lib/data";

const ACTIONS = [
  { id: "A1", name: "Stereo", text: "Use the right camera view's label where the left view is unsure." },
  { id: "A2", name: "Temporal", text: "Recover a label from the recent past using the temporal memory." },
  { id: "A3", name: "Refine", text: "Class-conditioned morphological clean-up of thin structures." },
  { id: "FUSE", name: "Fusion", text: "Stereo and temporal combined (the blind baseline)." },
];

const COLORS = (p: string) => (p === "selector" ? "#10b981" : p.startsWith("oracle") ? "#a1a1aa" : p.includes("A3") || p.includes("refine") ? "#ef4444" : "#18181b");

export default function SelectorPage() {
  const [split, setSplit] = useState<"test" | "oof">("test");
  const [act, setAct] = useState("A3");
  const [rec, setRec] = useState<"test_always" | "oof_always" | "oof_selective">("test_always");
  const Sel = R.selector;
  const rows = (split === "test" ? Sel.test : Sel.oof).filter((r) => r.policy !== "A0 (keep)");
  const sc = split === "test" ? Sel.stop_check_test : Sel.stop_check_oof;
  const q = (Sel.prediction_quality_oof as Record<string, { pearson: number; mae: number; true_mean: number; pred_mean: number; n: number }>)[act];
  const r = R.recovery[rec];
  const sel = rows.find((x) => x.policy === "selector")!;
  const a3 = rows.find((x) => x.policy === "always-A3 (routed)")!;

  return (
    <main>
      <PageHero
        eyebrow="Candidate fixes and the selector"
        title="Choosing the right fix per frame"
        lead="For each frame the gate routes, four fixes are possible. A learned selector predicts the gain of each and abstains when none is worth it. We compare it with simply always applying one fix."
      >
        <LinkButton href="/results/">Back to the results</LinkButton>
      </PageHero>

      <Section eyebrow="The four fixes" title="Inspect a candidate" lead="Select a fix to see how well the selector could predict its benefit on held-out training frames.">
        <Toggle options={ACTIONS.map((a) => ({ id: a.id, label: `${a.id} ${a.name}` }))} value={act} onChange={setAct} label="Action" />
        <div className="grid md:grid-cols-2 gap-4 mt-6">
          <div className="bg-white border border-zinc-200 rounded-xl p-6">
            <h3 className="text-lg font-medium text-zinc-900">{ACTIONS.find((a) => a.id === act)!.name}</h3>
            <p className="text-sm text-zinc-600 mt-2">{ACTIONS.find((a) => a.id === act)!.text}</p>
            <dl className="grid grid-cols-2 gap-4 mt-6 text-sm">
              <div><dt className="text-xs uppercase text-zinc-400">Prediction correlation</dt><dd className="text-2xl font-[family-name:var(--font-urbanist)] text-zinc-900">{q.pearson.toFixed(2)}</dd></div>
              <div><dt className="text-xs uppercase text-zinc-400">Mean error</dt><dd className="text-2xl font-[family-name:var(--font-urbanist)] text-zinc-900">{q.mae.toFixed(3)}</dd></div>
              <div><dt className="text-xs uppercase text-zinc-400">True mean gain</dt><dd className="text-2xl font-[family-name:var(--font-urbanist)] text-zinc-900">{signed(q.true_mean)}</dd></div>
              <div><dt className="text-xs uppercase text-zinc-400">Predicted mean gain</dt><dd className="text-2xl font-[family-name:var(--font-urbanist)] text-zinc-900">{signed(q.pred_mean)}</dd></div>
            </dl>
          </div>
          <Callout tone={q.pearson > 0.3 ? "good" : "warn"} title={q.pearson > 0.3 ? "The selector can partly predict this fix" : "The selector can barely predict this fix"}>
            Correlation between predicted and measured IoU change on {q.n} routed out-of-fold frames is {q.pearson.toFixed(2)}. Stereo (A1) is essentially unpredictable because it changes almost nothing; the selector is moderately informed about temporal and fusion fixes.
          </Callout>
        </div>
      </Section>

      <Section tint eyebrow="Policies compared" title="Selector versus always-apply baselines" lead="Average IoU change per frame, over all frames in the split. The oracle is the best possible choice per frame (an upper bound, not achievable).">
        <Toggle options={[{ id: "test", label: "Official test (997 frames)" }, { id: "oof", label: "Held-out training (2235 frames)" }]} value={split} onChange={setSplit} label="Split" />
        <div className="bg-white border border-zinc-200 rounded-xl p-5 md:p-8 mt-6">
          <HBars domain={[0, 0.03]} fmt={(v) => signed(v)} bars={rows.map((p) => ({ label: p.policy, value: p.mean_delta_all, color: COLORS(p.policy), sub: `${p.activation_pct.toFixed(0)}% of frames corrected` }))} />
        </div>
        <div className="mt-6">
          <DataTable
            head={["Policy", "Corrected", "Mean gain (all)", "Harmful of corrected", "Improved of corrected", "CPU ms/frame"]}
            highlight={[rows.findIndex((x) => x.policy === "selector")]}
            rows={rows.map((p) => [p.policy, pct(p.activation_pct), signed(p.mean_delta_all), pct(p.harmful_pct_of_corrected, 2), pct(p.improved_pct_of_corrected, 1), p.cpu_ms.toFixed(0)])}
          />
          <p className="text-xs text-zinc-500 mt-3">Units here are the change in instrument IoU at the 512x640 working resolution, so they differ from the full-resolution official numbers on the Results page.</p>
        </div>
        <div className="grid md:grid-cols-2 gap-4 mt-8">
          <Callout tone="bad" title="The selector does not beat the simplest baseline">
            Always applying the morphological refinement (A3) gains {signed(a3.mean_delta_all)} versus {signed(sel.mean_delta_all)} for the selector, with similar harm ({pct(a3.harmful_pct_of_corrected, 1)} vs {pct(sel.harmful_pct_of_corrected, 1)}).
            A3 uses no stereo and no temporal information.
          </Callout>
          <Callout tone="good" title="There was headroom, so trying was worthwhile">
            The oracle gains {signed(sc.oracle)} versus {signed(sc.always_A2)} for always-temporal (difference {signed(sc.oracle_minus_always_A2)}); the pre-registered stop check passed.
          </Callout>
        </div>
      </Section>

      <Section eyebrow="Does correction help at all" title="Temporal recovery, measured on its own" lead="Before building the selector we measured whether correcting routed frames helps. It does, by very little.">
        <Toggle
          options={[
            { id: "test_always", label: "Test, correct all routed" },
            { id: "oof_always", label: "Held-out train, all frames" },
            { id: "oof_selective", label: "Held-out train, gated frames" },
          ]}
          value={rec}
          onChange={setRec}
          label="Setting"
        />
        <div className="grid md:grid-cols-2 gap-4 mt-6">
          <DataTable
            head={["Metric", "Before", "After", "95% interval of change"]}
            rows={[
              ["Instrument IoU", f4(r.instrument_before), f4(r.instrument_after), `${signed(r.ci[0])} to ${signed(r.ci[1])}`],
              ["Official-metric IoU", f4(r.official_before), f4(r.official_after), `${signed(r.official_ci[0])} to ${signed(r.official_ci[1])}`],
            ]}
          />
          <Callout tone="warn" title="Real but tiny">
            {r.routed} of {r.frames} frames were corrected. The change is statistically above zero for instrument IoU, but on the official metric it is a few ten-thousandths. Almost every corrected frame is unchanged.
          </Callout>
        </div>
      </Section>

      <Section tint eyebrow="Abstention" title="How cautious should the selector be?" lead="The margin is the predicted gain below which the selector keeps the original. We fixed it at 0.01 in advance; the sweep is shown for information only.">
        <DataTable
          head={["Margin", "Frames corrected", "Mean gain (all)", "Harmful of corrected"]}
          highlight={[Sel.margin_sweep_oof.findIndex((m) => m.margin === Sel.margin)]}
          rows={Sel.margin_sweep_oof.map((m) => [m.margin.toFixed(3), pct(m.activation_pct_of_all), signed(m.mean_delta_all), pct(m.harmful_pct_of_corrected, 2)])}
        />
        <p className="text-xs text-zinc-500 mt-3">Out-of-fold, {Sel.n_oof} frames. A larger margin corrects fewer frames and gains less; harm per corrected frame does not fall in a steady way.</p>
        <div className="mt-8"><LinkButton href="/limits/" variant="dark">See claims and limits</LinkButton></div>
      </Section>
    </main>
  );
}

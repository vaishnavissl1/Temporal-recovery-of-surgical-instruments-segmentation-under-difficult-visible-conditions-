"use client";

import { useState } from "react";
import { Callout, DataTable, HBars, LinkButton, PageHero, Section, Toggle, type Bar } from "@/components/ui";
import { R, SHORT, f3, model, pct } from "@/lib/data";

type MetricId = "mIoU" | "boundary_f1" | "activation_pct" | "harmful_pct_of_all" | "harmful_pct_of_corrected" | "latency_mean_ms";
const METRICS: { id: MetricId; label: string; better: string; fmt: (v: number) => string; domain?: [number, number]; note: string }[] = [
  { id: "mIoU", label: "mIoU", better: "higher is better", fmt: f3, domain: [0, 1], note: "Official challenge mean IoU over classes present in each frame. Scale starts at 0 so differences are not exaggerated." },
  { id: "boundary_f1", label: "Boundary F1", better: "higher is better", fmt: f3, domain: [0, 1], note: "Instrument boundary F1 within 3 px tolerance." },
  { id: "activation_pct", label: "Frames corrected", better: "lower = cheaper", fmt: (v) => v.toFixed(1) + "%", domain: [0, 100], note: "Share of frames where a correction was applied." },
  { id: "harmful_pct_of_all", label: "Harmful (all frames)", better: "lower is better", fmt: (v) => v.toFixed(2) + "%", domain: [0, 2], note: "Frames where the correction lowered instrument IoU by more than 0.01, as a share of every test frame." },
  { id: "harmful_pct_of_corrected", label: "Harmful (corrected frames)", better: "lower is better", fmt: (v) => v.toFixed(2) + "%", domain: [0, 4], note: "The same, as a share of only the frames that were corrected. Selective methods look worse on this definition because they correct fewer frames." },
  { id: "latency_mean_ms", label: "Latency", better: "lower is better", fmt: (v) => v.toFixed(0) + " ms", domain: [0, 2500], note: "Mean per-frame time composed from measured components on the laptop GPU." },
];

const ID_COLOR: Record<string, string> = { M1: "#71717a", M2: "#0ea5e9", M3: "#8b5cf6", M4: "#f59e0b", M5: "#ef4444", M6: "#10b981" };

export default function ResultsPage() {
  const [metric, setMetric] = useState<MetricId>("mIoU");
  const [view, setView] = useState<"seq" | "class" | "quart">("seq");
  const [cls, setCls] = useState("M1");
  const met = METRICS.find((m) => m.id === metric)!;
  const bars: Bar[] = R.models.map((m) => ({ label: `${m.id}  ${SHORT[m.id]}`, value: m[metric] as number, color: ID_COLOR[m.id] }));
  const seqs = Object.keys(model("M1").per_sequence);
  const m4 = R.headline_M4_vs_M6;

  return (
    <main>
      <PageHero
        eyebrow="Results"
        title="Six models, four official test sequences"
        lead={`${R.meta.n_test_frames} labelled frames scored with the official challenge metric at 1280x1024. Switch the metric below; the scale is fixed so a tiny gap looks tiny.`}
      >
        <LinkButton href="/selector/">The selector in detail</LinkButton>
        <LinkButton href="/limits/" variant="outline">
          What this means
        </LinkButton>
      </PageHero>

      <Section eyebrow="Compare" title="M1 to M6 on one metric" lead="M1 is the base segmenter. M2 to M5 add corrections to every frame. M6 chooses selectively.">
        <Toggle options={METRICS.map((m) => ({ id: m.id, label: m.label }))} value={metric} onChange={setMetric} label="Metric" />
        <div className="mt-8 bg-white border border-zinc-200 rounded-xl p-5 md:p-8">
          <div className="flex items-baseline justify-between flex-wrap gap-2 mb-6">
            <h3 className="text-lg font-medium text-zinc-900">{met.label}</h3>
            <span className="text-xs text-zinc-500">{met.better}</span>
          </div>
          <HBars bars={bars} domain={met.domain} fmt={met.fmt} />
          <p className="text-xs text-zinc-500 mt-6 max-w-2xl">{met.note}</p>
        </div>
        <div className="grid md:grid-cols-2 gap-4 mt-6">
          <Callout tone="warn" title="All six mIoU values sit in a 0.0012 band">
            M1 {f3(model("M1").mIoU)} to M5 {f3(model("M5").mIoU)}. The gap between sequences (0.41 to 0.81) is hundreds of times larger. No correction changes the result in a way that matters.
          </Callout>
          <Callout tone="good" title="The cost result">
            M6 corrects {pct(m4.activation_pct[1])} of frames versus {pct(m4.activation_pct[0])} for blind fusion (M4), and harms {m4.harmful_pct_of_all[1].toFixed(2)}% of all frames versus{" "}
            {m4.harmful_pct_of_all[0].toFixed(2)}%. Per corrected frame it is worse ({m4.harmful_pct_of_corrected[1].toFixed(1)}% vs {m4.harmful_pct_of_corrected[0].toFixed(1)}%). Both are reported.
          </Callout>
        </div>
      </Section>

      <Section tint eyebrow="Full table" title="Every number">
        <DataTable
          head={["Model", "mIoU", "Boundary F1", "clDice shaft", "clDice thread", "Corrected", "Harmful (all)", "Latency"]}
          highlight={[5]}
          rows={R.models.map((m) => [
            `${m.id} ${SHORT[m.id]}`,
            f3(m.mIoU),
            f3(m.boundary_f1),
            f3(m.cldice.shaft),
            f3(m.cldice.thread),
            pct(m.activation_pct),
            pct(m.harmful_pct_of_all, 2),
            `${m.latency_mean_ms!.toFixed(0)} ms`,
          ])}
        />
        <p className="text-xs text-zinc-500 mt-3">
          Needle clDice is 0.000 for every model. Latency for M2 and M4 is dominated by the CPU stereo matcher (about 1.75 s per frame); the segmentation forward pass is about 72 ms.
        </p>
      </Section>

      <Section eyebrow="Break it down" title="Where the score comes from">
        <Toggle
          options={[
            { id: "seq", label: "By sequence" },
            { id: "class", label: "By class" },
            { id: "quart", label: "By difficulty quartile" },
          ]}
          value={view}
          onChange={setView}
          label="Breakdown"
        />
        <div className="mt-6">
          {view === "seq" && (
            <>
              <DataTable
                head={["Sequence", ...R.models.map((m) => m.id)]}
                rows={seqs.map((s) => [s, ...R.models.map((m) => f3((m.per_sequence as Record<string, number>)[s]))])}
              />
              <p className="text-xs text-zinc-500 mt-3">Sequence 4 is hard for every method (0.41) and sequence 3 is easy (0.81). Method differences are in the third decimal.</p>
            </>
          )}
          {view === "class" && (
            <>
              <Toggle options={R.models.map((m) => ({ id: m.id, label: m.id }))} value={cls} onChange={setCls} label="Model" />
              <div className="bg-white border border-zinc-200 rounded-xl p-5 md:p-8 mt-4">
                <HBars
                  domain={[0, 1]}
                  bars={Object.entries(model(cls).per_class).map(([k, v]) => ({ label: k, value: v as number | null, color: ID_COLOR[cls] }))}
                />
              </div>
              <p className="text-xs text-zinc-500 mt-3">
                Suturing needle scores 0 and suction instrument does not appear in the test set (n/a), so 11 of 12 classes are scored. Classes are averaged per frame where present, as in the official script.
              </p>
            </>
          )}
          {view === "quart" && (
            <>
              <DataTable
                head={["Difficulty quartile (M6)", "Frames", "mIoU", "Corrected", "Harmful (all)"]}
                rows={model("M6").by_difficulty_quartile.map((q) => [
                  `Q${q.quartile}${q.quartile === 1 ? " (easiest by index)" : q.quartile === 4 ? " (hardest by index)" : ""}`,
                  q.n,
                  f3(q.mIoU),
                  pct(q.activation_pct),
                  pct(q.harmful_pct_of_all, 2),
                ])}
              />
              <p className="text-xs text-zinc-500 mt-3">The difficulty index did not validate (Spearman +0.08 on test), so treat quartiles as descriptive only.</p>
            </>
          )}
        </div>
      </Section>

      <Section tint eyebrow="Trust" title="The scorer matches the official script">
        <div className="grid md:grid-cols-3 gap-4">
          <Callout tone="good" title={`Official run.py: ${R.crosscheck.official_run_py_mean_iou.toFixed(6)}`}>Our M1 mIoU: {R.crosscheck.our_M1_mIoU.toFixed(6)}. Absolute difference {R.crosscheck.abs_difference.toExponential(1)} over {R.crosscheck.n_frames_scored} frames.</Callout>
          <Callout tone="info" title="Gate detection (test)">The reliability gate detects frames where the base prediction is bad with AUROC {R.gate_detection.auroc_detect_bad_base_frames.toFixed(3)} ({pct(100 * R.gate_detection.bad_rate)} of frames are bad).</Callout>
          <Callout tone="warn" title="Not comparable to papers">We do not compare against published EndoVis numbers: protocols and class handling differ.</Callout>
        </div>
      </Section>
    </main>
  );
}

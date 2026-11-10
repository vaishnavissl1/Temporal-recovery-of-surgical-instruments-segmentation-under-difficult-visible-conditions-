"use client";

import { useState } from "react";
import { Check, X } from "lucide-react";
import { Callout, Card, LinkButton, PageHero, Section, Toggle } from "@/components/ui";
import { R } from "@/lib/data";

const SUPPORTED = [
  "Our evaluation reproduces the official challenge metric exactly (difference 1e-16).",
  "Calibrated stereo rectification is usable on this data, and a label-free vertical correction repairs the sequences where the shipped calibration is tilted.",
  "A reliability gate built from confidence and temporal indicators detects bad frames (AUROC about 0.84 on test).",
  "Selective correction matches blind fusion's mIoU while correcting about a quarter of the frames, at roughly a third of the compute.",
];
const UNSUPPORTED = [
  "That stereo evidence improves failure detection or correction (measured: no difference).",
  "That any correction raises mIoU in a way that matters (at most +0.0012).",
  "That the difficulty index predicts failures (it did not validate).",
  "That the learned selector beats a fixed morphological refinement (it does not).",
  "That our method outperforms published results (no comparison was made; protocols differ).",
];
const LIMITS = [
  "Only four test sequences: intervals are coarse. Read them as warnings, not tests.",
  `The conformal risk target was not met on test: bad-and-accepted rate 0.077 against a 0.05 budget.`,
  "The temporal memory in the action data holds uncorrected predictions; chains of corrections are not simulated.",
  "Suturing needle scores 0 and suction instrument is absent from test, so 11 of 12 classes are scored.",
  "Latency is composed from separately measured components on a laptop GPU, not timed end to end.",
  "One training run per fold; no seed variation for the segmenter.",
  "Training resolution was capped at 512x640 by 4 GB of GPU memory.",
];
const SENTENCES = {
  yes: [
    `"On the four official test sequences the selective policy reaches the same mIoU as always-on fusion (0.613) while correcting 26% of frames and costing about 0.8 s instead of 2.2 s per frame."`,
    `"In our experiments, stereo disparity evidence did not improve reliability estimation beyond temporal indicators."`,
  ],
  no: [`"Stereo gating improves surgical segmentation."`, `"Our method outperforms published results."`, `"The selector chooses the best correction for each frame."`],
};

export default function LimitsPage() {
  const [tab, setTab] = useState<"sup" | "uns" | "lim" | "words">("sup");
  return (
    <main>
      <PageHero
        eyebrow="Claims and limits"
        title="What the evidence supports, and what it does not"
        lead="Pick a tab. Every item is tied to a measurement in the results files; nothing here is tuned after seeing a test number."
      >
        <LinkButton href="/results/">Check the numbers</LinkButton>
      </PageHero>

      <Section eyebrow="Be precise" title="Four lists">
        <Toggle
          options={[
            { id: "sup", label: "Supported" },
            { id: "uns", label: "Not supported" },
            { id: "lim", label: "Limitations" },
            { id: "words", label: "Sentences to use / avoid" },
          ]}
          value={tab}
          onChange={setTab}
          label="List"
        />
        <div className="mt-8 max-w-3xl flex flex-col gap-3">
          {tab === "sup" &&
            SUPPORTED.map((t) => (
              <Card key={t} className="flex gap-3 !p-4">
                <Check className="size-5 text-emerald-600 shrink-0 mt-0.5" />
                <p className="text-sm text-zinc-700">{t}</p>
              </Card>
            ))}
          {tab === "uns" &&
            UNSUPPORTED.map((t) => (
              <Card key={t} className="flex gap-3 !p-4">
                <X className="size-5 text-rose-600 shrink-0 mt-0.5" />
                <p className="text-sm text-zinc-700">{t}</p>
              </Card>
            ))}
          {tab === "lim" &&
            LIMITS.map((t) => (
              <Card key={t} className="!p-4">
                <p className="text-sm text-zinc-700">{t}</p>
              </Card>
            ))}
          {tab === "words" && (
            <>
              <p className="text-sm font-medium text-emerald-700">Defensible</p>
              {SENTENCES.yes.map((t) => (
                <Card key={t} className="!p-4 border-emerald-200"><p className="text-sm text-zinc-700">{t}</p></Card>
              ))}
              <p className="text-sm font-medium text-rose-700 mt-4">Not defensible</p>
              {SENTENCES.no.map((t) => (
                <Card key={t} className="!p-4 border-rose-200"><p className="text-sm text-zinc-700">{t}</p></Card>
              ))}
            </>
          )}
        </div>
      </Section>

      <Section tint eyebrow="Overall" title="What this adds up to">
        <div className="grid md:grid-cols-2 gap-4 max-w-4xl">
          <Callout tone="bad" title="The stereo premise is not supported">
            Stereo is geometrically viable but added nothing to failure detection or correction on this data. The temporal indicator did the work.
          </Callout>
          <Callout tone="good" title="The cost saving is real">
            Selective correction touches {R.models[5].activation_pct.toFixed(0)}% of frames and runs in {R.models[5].latency_mean_ms!.toFixed(0)} ms per frame on average, against {R.models[3].latency_mean_ms!.toFixed(0)} ms for blind fusion, with the same mIoU.
          </Callout>
        </div>
        <div className="mt-8"><LinkButton href="/run/" variant="dark">Reproduce it</LinkButton></div>
      </Section>
    </main>
  );
}

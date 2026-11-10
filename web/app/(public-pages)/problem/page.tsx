"use client";

import { useState } from "react";
import { Callout, Card, LinkButton, PageHero, Section } from "@/components/ui";
import { R } from "@/lib/data";

const STEPS = [
  {
    id: "seg",
    name: "1. Segment",
    short: "Base segmenter",
    body: "A CAFormer-S18 backbone pre-trained on surgical video (SurgeNetXL) with a feature-pyramid decoder and a full-resolution branch labels every pixel with one of 12 classes. Trained at 512x640 on a 4 GB laptop GPU, scored at 1280x1024.",
    out: "A class map per frame, plus per-pixel confidence.",
  },
  {
    id: "gate",
    name: "2. Gate",
    short: "Reliability gate",
    body: "A small learned model reads indicators for each frame (confidence, temporal consistency, area stability, boundary and fragment measures, optional stereo cues) and predicts whether the segmentation is trustworthy. A conformal threshold decides which frames are routed for possible correction.",
    out: `A reliability score r. Frames with r below 0.64 are routed (${R.meta.n_routed} of ${R.meta.n_test_frames} test frames).`,
  },
  {
    id: "act",
    name: "3. Candidates",
    short: "Candidate fixes",
    body: "Three cheap fixes are prepared for each routed frame. A1 uses the right camera view where the left view is unsure. A2 recovers a label from the recent past (temporal recovery). A3 applies a class-conditioned morphological clean-up for thin structures such as threads and shafts. Two baselines apply fusion on every frame.",
    out: "Up to four corrected masks, each with a measurable change in IoU.",
  },
  {
    id: "sel",
    name: "4. Select",
    short: "Benefit selector",
    body: "A regression model predicts, for each candidate, how much the frame's IoU would change. It picks the best candidate, and abstains (keeps the original) when the predicted gain is below a fixed margin of 0.01.",
    out: "One action per routed frame, or no action.",
  },
  {
    id: "eval",
    name: "5. Evaluate",
    short: "Honest scoring",
    body: "Six models (M1 to M6) are scored on the official test sequences with the official challenge metric at full resolution, with boundary F1, thin-structure clDice, harmful-correction rate, difficulty quartiles and latency. All gates and selectors are fit on out-of-fold training predictions only.",
    out: "The Results page.",
  },
];

const FACTS = [
  ["Training sequences", "15 (3-fold cross-fitting, class-aware folds)"],
  ["Test sequences", `${R.meta.n_test_sequences} official, ${R.meta.n_test_frames} labelled frames`],
  ["Classes", "12 in the task, 11 present in test (suction instrument absent)"],
  ["Input", "Stereo pairs, per-sequence calibration, 1280x1024"],
  ["Hardware", "RTX 3050 Ti laptop GPU, 4 GB (about 3.3 GB usable)"],
  ["Metric", "Official EndoVis challenge IoU (background excluded)"],
];

export default function ProblemPage() {
  const [step, setStep] = useState("seg");
  const s = STEPS.find((x) => x.id === step)!;
  return (
    <main>
      <PageHero
        eyebrow="Problem and approach"
        title="Which frames need fixing, and what is the cheapest fix?"
        lead="Surgical scene segmentation degrades with smoke, blood, glare and occlusion. We test whether stereo geometry can flag those failures and guide a targeted correction."
      >
        <LinkButton href="/results/">Skip to results</LinkButton>
        <LinkButton href="/run/" variant="outline">
          Run it
        </LinkButton>
      </PageHero>

      <Section eyebrow="Pipeline" title="Five steps, click through them" lead="Choose a step to see what it does and what it produces.">
        <div className="flex flex-wrap gap-2" role="tablist">
          {STEPS.map((x) => (
            <button
              key={x.id}
              role="tab"
              aria-selected={x.id === step}
              onClick={() => setStep(x.id)}
              className={`px-5 py-3 rounded-lg text-sm text-left transition cursor-pointer border ${
                x.id === step ? "bg-zinc-900 text-white border-zinc-900" : "bg-white border-zinc-200 text-zinc-700 hover:border-zinc-400"
              }`}
            >
              <span className="block font-medium">{x.name}</span>
              <span className={`block text-xs ${x.id === step ? "text-zinc-300" : "text-zinc-400"}`}>{x.short}</span>
            </button>
          ))}
        </div>
        <Card className="mt-6 max-w-3xl">
          <h3 className="text-xl font-medium text-zinc-900">{s.short}</h3>
          <p className="text-zinc-600 mt-3 leading-relaxed">{s.body}</p>
          <p className="text-sm text-zinc-900 mt-4">
            <span className="font-medium">Produces: </span>
            {s.out}
          </p>
        </Card>
      </Section>

      <Section tint eyebrow="The question" title="What this project actually tests">
        <div className="grid md:grid-cols-3 gap-4">
          <Card>
            <h3 className="font-medium text-zinc-900">Premise</h3>
            <p className="text-sm text-zinc-600 mt-2 leading-relaxed">Stereo disparity gives evidence that a single image cannot, so adding it to the reliability gate should detect more failures.</p>
          </Card>
          <Card>
            <h3 className="font-medium text-zinc-900">Method</h3>
            <p className="text-sm text-zinc-600 mt-2 leading-relaxed">Compare gates with and without stereo indicators, then compare corrections chosen selectively against corrections applied blindly to every frame.</p>
          </Card>
          <Card>
            <h3 className="font-medium text-zinc-900">Rule</h3>
            <p className="text-sm text-zinc-600 mt-2 leading-relaxed">No tuning to flatter a number. A measurement that disappoints is reported as the result.</p>
          </Card>
        </div>
        <div className="mt-8">
          <Callout tone="info" title="Where this leads">
            Stereo is geometrically usable here, but it did not improve failure detection (see Stereo). The surviving contribution is a cost one: selective correction reaches the same accuracy as blind fusion at roughly a third of the compute (see Results).
          </Callout>
        </div>
      </Section>

      <Section eyebrow="Setup" title="Data and constraints">
        <div className="grid sm:grid-cols-2 gap-4 max-w-4xl">
          {FACTS.map(([k, v]) => (
            <Card key={k} className="!p-4">
              <p className="text-xs uppercase tracking-wide text-zinc-400">{k}</p>
              <p className="text-zinc-900 mt-1">{v}</p>
            </Card>
          ))}
        </div>
        <div className="mt-8 flex gap-3 flex-wrap">
          <LinkButton href="/stereo/" variant="dark">
            Check the stereo evidence
          </LinkButton>
          <LinkButton href="/selector/" variant="dark">
            See the selector
          </LinkButton>
        </div>
      </Section>
    </main>
  );
}

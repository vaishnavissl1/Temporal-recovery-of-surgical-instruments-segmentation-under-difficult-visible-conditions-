"use client";

import { useState } from "react";
import { Callout, CopyBlock, LinkButton, PageHero, Section, Toggle } from "@/components/ui";

const PY = `C:\\Users\\<you>\\anaconda3\\envs\\sitr\\python.exe`;

const STAGES = {
  env: {
    label: "Environment",
    text: "Every command must run inside the sitr conda environment. The path must contain \\envs\\sitr and CUDA must print True.",
    code: `${PY} -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.cuda.is_available())"

# one-off installs used by this project
${PY} -m pip install scipy scikit-learn streamlit`,
  },
  stereo: {
    label: "Stereo + evidence",
    text: "Task 2 checks rectification and matching; the caches hold disparity evidence per frame.",
    code: `${PY} -m tools.check_stereo --root <dataset> --out results/task2
${PY} -m tools.cache_stereo --root <dataset> --out work_v2/stereo --sgbm-json results/task2/stereo_check.json
${PY} -m tools.compare_stereo --oof <oof-cache-dir> --folds work_v2/folds.json --test <test-cache-dir> --out results/task4

# every tool lists its options with --help`,
  },
  train: {
    label: "Train + gate",
    text: "Cross-fitting folds, then the final model and the reliability gate. On a 4 GB GPU use gradient checkpointing and the memory cap (Windows silently spills to system RAM instead of failing).",
    code: `${PY} -m tools.run_v2 --root <dataset> --out work_v2 --folds 3 --grad-checkpoint --gpu-mem-cap-gb 3.3
# see tools/run_v2.py --help for epochs, batch size and accumulation`,
  },
  actions: {
    label: "Actions + selector",
    text: "Cache the right-view predictions, build candidate-action data, then train the selector with cross-fitting.",
    code: `${PY} -m tools.cache_right_view --work work_v2 --root <dataset> --out work_v2/right_view
${PY} -m tools.make_action_data --release work_v2/release --work work_v2 --root <dataset> ^
    --stereo work_v2/stereo --right-view work_v2/right_view --out work_v2/actions
${PY} -m tools.train_selector --actions work_v2/actions --folds work_v2/folds.json --release work_v2/release --out results/task5`,
  },
  final: {
    label: "Final evaluation",
    text: "M1 to M6 at full resolution with the official metric and a cross-check against the official run.py.",
    code: `${PY} -m tools.final_eval --actions work_v2/actions --choices results/task5/choices_test.npz ^
    --root <dataset> --release work_v2/release --out results/task6 --crosscheck`,
  },
  site: {
    label: "This website",
    text: "The site is a static export built only from aggregate numbers in results/.",
    code: `${PY} -m tools.export_web_data --results results --out web/data/results.json
cd web
npm install
npm run dev          # local preview at http://localhost:3000
npm run build        # static site in web/out (deploy this folder)
npx vercel --prod    # first time: vercel login`,
  },
} as const;
type Stage = keyof typeof STAGES;

export default function RunPage() {
  const [stage, setStage] = useState<Stage>("env");
  const s = STAGES[stage];
  return (
    <main>
      <PageHero eyebrow="Reproduce" title="Run it yourself" lead="The commands for each stage, in order. Use the buttons to switch stage and copy a block.">
        <LinkButton href="/limits/">Before you do: the caveats</LinkButton>
      </PageHero>

      <Section eyebrow="Commands" title="Pick a stage">
        <Toggle options={(Object.keys(STAGES) as Stage[]).map((k) => ({ id: k, label: STAGES[k].label }))} value={stage} onChange={setStage} label="Stage" />
        <p className="text-sm text-zinc-600 mt-6 max-w-2xl">{s.text}</p>
        <div className="mt-4 max-w-4xl"><CopyBlock code={s.code} /></div>
      </Section>

      <Section tint eyebrow="Repository" title="Where things are">
        <div className="grid md:grid-cols-2 gap-3 max-w-4xl text-sm">
          {[
            ["sitr/", "library: stereo, difficulty, actions, selector, metrics"],
            ["tools/", "one command-line entry point per task"],
            ["tests/", "unit checks; run each as python -m tests.<name>"],
            ["results/", "all measured numbers (JSON, logs, reports)"],
            ["results/FINDINGS.md", "the plain-language write-up"],
            ["RUNLOG.md", "every error, fix and decision, in order"],
            ["DATASET_FACTS.md", "what the dataset really contains"],
            ["web/", "this website (Next.js static export)"],
          ].map(([a, b]) => (
            <div key={a} className="bg-white border border-zinc-200 rounded-lg p-4">
              <p className="font-mono text-zinc-900">{a}</p>
              <p className="text-zinc-500 mt-1">{b}</p>
            </div>
          ))}
        </div>
        <div className="mt-8 max-w-3xl">
          <Callout tone="info" title="What is not in the repository or on this site">
            The EndoVis dataset, model weights and any patient frames. You need to obtain the dataset yourself and train the models to reproduce the numbers.
          </Callout>
        </div>
      </Section>
    </main>
  );
}

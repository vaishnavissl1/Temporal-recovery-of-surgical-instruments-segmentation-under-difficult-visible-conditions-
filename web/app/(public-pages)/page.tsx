"use client";

import Link from "next/link";
import { ArrowRight, BarChart3, Boxes, FlaskConical, Layers, ShieldAlert, Terminal } from "lucide-react";
import { Callout, Eyebrow, LinkButton, Reveal, Section, Stat } from "@/components/ui";
import { R, model, pct } from "@/lib/data";

const CARDS = [
  { href: "/problem/", icon: Layers, title: "The problem", text: "What goes wrong in surgical segmentation, and the pipeline we built to fix only the frames that need it." },
  { href: "/results/", icon: BarChart3, title: "Results", text: "Six models on the four official test sequences. Switch metrics, sequences and classes." },
  { href: "/stereo/", icon: FlaskConical, title: "Did stereo help?", text: "Calibration quality, and whether stereo evidence improves failure detection. It did not." },
  { href: "/selector/", icon: Boxes, title: "The selector", text: "Candidate corrections, the benefit selector with abstention, and how it compares with simple baselines." },
  { href: "/limits/", icon: ShieldAlert, title: "Claims and limits", text: "What the evidence supports, what it does not, and sentences you can and cannot defend." },
  { href: "/run/", icon: Terminal, title: "Run it yourself", text: "Environment, commands for each stage, and how this site is rebuilt from the result files." },
];

export default function Home() {
  const m1 = model("M1");
  const m5 = model("M5");
  const m6 = model("M6");
  const m4 = model("M4");
  const speedup = (m4.latency_mean_ms! / m6.latency_mean_ms!).toFixed(1);
  return (
    <main>
      <section className="relative flex flex-col items-center min-h-screen justify-center bg-zinc-950 px-4 w-full overflow-hidden">
        <div className="absolute inset-0 opacity-70 bg-[radial-gradient(60%_60%_at_75%_10%,rgba(56,189,248,0.30),transparent),radial-gradient(55%_60%_at_15%_90%,rgba(52,211,153,0.22),transparent)]" />
        <div className="relative flex flex-col items-center">
          <Reveal>
            <div className="bg-white/10 backdrop-blur text-sm text-white pl-3 pr-4 py-1 rounded-full flex items-center gap-2 border border-white/20">
              <span className="size-2 rounded-full bg-emerald-400" />
              EndoVis 2018 &middot; 12-class scene segmentation &middot; stereo pairs
            </div>
          </Reveal>
          <Reveal delay={0.05}>
            <h1 className="text-4xl md:text-[64px] text-zinc-50 font-medium max-w-4xl text-center mt-5 leading-tight">
              Fix a surgical segmenter only where it fails
            </h1>
          </Reveal>
          <Reveal delay={0.1}>
            <p className="text-zinc-200 max-w-2xl text-center mt-4">
              Stereo-gated selective correction, tested on {R.meta.n_test_frames} frames from the {R.meta.n_test_sequences} official test sequences. This site shows what held up and
              what did not, in plain numbers.
            </p>
          </Reveal>
          <Reveal delay={0.15} className="flex flex-wrap justify-center items-center gap-4 mt-8">
            <LinkButton href="/results/">See the results</LinkButton>
            <LinkButton href="/problem/" variant="outline">
              How it works
            </LinkButton>
            <LinkButton href="/limits/" variant="outline">
              What we can claim
            </LinkButton>
          </Reveal>
        </div>
      </section>

      <Section eyebrow="Headline numbers" title="The honest summary" lead="Official metric, full 1280x1024 resolution, four official test sequences. Our scorer reproduces the official script to 1e-16.">
        <div className="flex max-lg:flex-col max-lg:gap-10 justify-between max-w-5xl">
          <Stat value={m1.mIoU.toFixed(3)} label="Base mIoU (M1)" note="segmentation only, no correction" />
          <div className="max-lg:hidden h-20 w-px bg-zinc-200" />
          <Stat value={m5.mIoU.toFixed(3)} label="Best corrected mIoU" note={`M5, +${(m5.mIoU - m1.mIoU).toFixed(4)} over base`} />
          <div className="max-lg:hidden h-20 w-px bg-zinc-200" />
          <Stat value={pct(m6.activation_pct)} label="Frames our selector touches" note="blind fusion touches 99.9%" />
          <div className="max-lg:hidden h-20 w-px bg-zinc-200" />
          <Stat value={`${speedup}x`} label="Cheaper than blind fusion" note={`${m6.latency_mean_ms!.toFixed(0)} ms vs ${m4.latency_mean_ms!.toFixed(0)} ms per frame`} />
        </div>
        <div className="grid md:grid-cols-3 gap-4 mt-12">
          <Callout tone="good" title="What held up">
            The pipeline and metric are exact. Calibrated stereo is usable. Selective correction matches blind fusion&apos;s mIoU at about a third of the cost.
          </Callout>
          <Callout tone="bad" title="What did not">
            Stereo evidence did not improve failure detection, and no correction raised mIoU by more than {(m5.mIoU - m1.mIoU).toFixed(4)}. Temporal cues, not stereo, carried the gate.
          </Callout>
          <Callout tone="warn" title="Read with care">
            Only four test sequences. Sequence-to-sequence spread (0.41 to 0.81) is far larger than any difference between methods.
          </Callout>
        </div>
      </Section>

      <Section tint eyebrow="Explore" title="Pick a page" lead="Each page answers one question and uses buttons to switch between views.">
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {CARDS.map((c, i) => (
            <Reveal key={c.href} delay={i * 0.05}>
              <Link href={c.href} className="group block h-full bg-white border border-zinc-200 rounded-xl p-6 hover:border-zinc-900 transition">
                <c.icon className="size-6 text-zinc-700" />
                <h3 className="text-xl text-zinc-900 mt-4 font-medium">{c.title}</h3>
                <p className="text-sm text-zinc-500 mt-2 leading-relaxed">{c.text}</p>
                <span className="inline-flex items-center gap-1.5 text-sm text-zinc-900 mt-5">
                  Open <ArrowRight className="size-4 transition-transform group-hover:translate-x-1" />
                </span>
              </Link>
            </Reveal>
          ))}
        </div>
      </Section>

      <Section eyebrow="Why it matters" title="The use case">
        <div className="max-w-3xl text-zinc-600 leading-relaxed flex flex-col gap-4">
          <p>
            Robotic surgery systems that segment instruments and tissue can fail when the view is smoky, bloody, glared or occluded. Fixing every frame is expensive. If a system could tell
            <em> which</em> frames are unreliable and <em>choose the cheapest fix that helps</em>, it could save compute on the frames that were already fine.
          </p>
          <p>
            We built that, measured it honestly, and report that the saving is real while the accuracy gain is not. That is a useful result for anyone deciding whether
            to add stereo hardware processing to a segmentation pipeline: on this data, it does not pay for itself.
          </p>
          <Eyebrow>Next</Eyebrow>
          <div className="flex gap-3 flex-wrap">
            <LinkButton href="/problem/" variant="dark">
              Read how it works
            </LinkButton>
            <LinkButton href="/results/" variant="dark">
              Go to the results
            </LinkButton>
          </div>
        </div>
      </Section>
    </main>
  );
}

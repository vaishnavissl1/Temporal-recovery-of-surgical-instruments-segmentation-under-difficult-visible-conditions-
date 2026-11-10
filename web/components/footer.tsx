import Link from "next/link";
import { NAV } from "@/lib/nav";
import { Logo } from "@/components/ui";

export function Footer() {
  return (
    <footer className="bg-black text-white pt-16 pb-8 mt-24 px-4 md:px-16 lg:px-24 xl:px-32 w-full">
      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-10 pb-12">
        <div className="lg:col-span-7 flex flex-col items-start gap-5">
          <Logo />
          <p className="text-zinc-300 text-sm/6 max-w-lg">
            Measured results for stereo-gated selective correction of robotic surgical scene segmentation, on the four official EndoVis 2018 test sequences.
            Negative results are reported as results.
          </p>
        </div>
        <div className="lg:col-span-5 flex gap-16 flex-wrap">
          <div className="flex flex-col gap-3 text-sm">
            <span className="text-white">Pages</span>
            {NAV.map((n) => (
              <Link key={n.href} href={n.href} className="text-zinc-300 hover:text-white transition-colors">
                {n.label}
              </Link>
            ))}
          </div>
          <div className="flex flex-col gap-3 text-sm max-w-56">
            <span className="text-white">About this page</span>
            <span className="text-zinc-300 text-xs/5">
              Aggregate numbers only: no patient frames, no model weights, no analytics or cookies. Every figure comes from a file in the repository&apos;s results folder.
            </span>
          </div>
        </div>
      </div>
      <div className="max-w-7xl mx-auto border-t border-white/10 pt-6 text-xs text-zinc-400">
        Not compared to published EndoVis numbers: protocols differ. Research prototype, not for clinical use.
      </div>
    </footer>
  );
}

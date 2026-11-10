"use client";

import Link from "next/link";
import React, { useState } from "react";
import { motion } from "framer-motion";
import { Check, Copy } from "lucide-react";

export function Logo({ dark = false }: { dark?: boolean }) {
  const c = dark ? "#18181b" : "#ffffff";
  return (
    <span className="flex items-center gap-2 select-none" style={{ color: c }}>
      <svg width="28" height="28" viewBox="0 0 32 32" fill="none" aria-hidden>
        <rect x="2" y="6" width="13" height="20" rx="3" stroke={c} strokeWidth="2" />
        <rect x="17" y="6" width="13" height="20" rx="3" stroke={c} strokeWidth="2" />
        <circle cx="8.5" cy="16" r="2.5" fill={c} />
        <circle cx="23.5" cy="16" r="2.5" fill={c} />
      </svg>
      <span className="font-[family-name:var(--font-urbanist)] text-lg font-semibold tracking-tight">Stereo-Gated</span>
    </span>
  );
}

export function Reveal({ children, delay = 0, className = "" }: { children: React.ReactNode; delay?: number; className?: string }) {
  return (
    <motion.div
      className={className}
      initial={{ y: 28, opacity: 0 }}
      whileInView={{ y: 0, opacity: 1 }}
      viewport={{ once: true }}
      transition={{ delay, type: "spring", stiffness: 240, damping: 60, mass: 1 }}
    >
      {children}
    </motion.div>
  );
}

export function Eyebrow({ children, light = false }: { children: React.ReactNode; light?: boolean }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className={`size-1.5 ${light ? "bg-white" : "bg-zinc-900"}`} />
      <span className={`text-sm uppercase tracking-wide ${light ? "text-white/90" : "text-zinc-900"}`}>{children}</span>
    </div>
  );
}

/** Dark band at the top of every inner page (the navbar floats over it). */
export function PageHero({ eyebrow, title, lead, children }: { eyebrow: string; title: string; lead: string; children?: React.ReactNode }) {
  return (
    <section className="relative bg-zinc-950 text-white pt-36 pb-20 px-4 md:px-16 lg:px-24 xl:px-32 overflow-hidden">
      <div className="absolute inset-0 opacity-60 bg-[radial-gradient(60%_80%_at_80%_0%,rgba(56,189,248,0.25),transparent),radial-gradient(50%_70%_at_10%_100%,rgba(52,211,153,0.18),transparent)]" />
      <div className="relative max-w-5xl">
        <Reveal>
          <Eyebrow light>{eyebrow}</Eyebrow>
        </Reveal>
        <Reveal delay={0.05}>
          <h1 className="text-4xl md:text-6xl font-medium leading-tight mt-6">{title}</h1>
        </Reveal>
        <Reveal delay={0.1}>
          <p className="text-zinc-300 mt-5 max-w-2xl text-base md:text-lg">{lead}</p>
        </Reveal>
        {children && <div className="mt-8 flex flex-wrap gap-3">{children}</div>}
      </div>
    </section>
  );
}

export function Section({ eyebrow, title, lead, children, tint = false, id }: { eyebrow?: string; title: string; lead?: string; children: React.ReactNode; tint?: boolean; id?: string }) {
  return (
    <section id={id} className={`py-14 md:py-20 px-4 md:px-16 lg:px-24 xl:px-32 w-full ${tint ? "bg-gray-50" : ""}`}>
      <div className="max-w-6xl mx-auto">
        <Reveal>
          {eyebrow && <Eyebrow>{eyebrow}</Eyebrow>}
          <h2 className="text-3xl md:text-[40px]/12 text-zinc-900 mt-4 font-medium max-w-3xl">{title}</h2>
          {lead && <p className="text-zinc-500 mt-3 max-w-2xl text-sm md:text-base">{lead}</p>}
        </Reveal>
        <div className="mt-10">{children}</div>
      </div>
    </section>
  );
}

export function LinkButton({ href, children, variant = "light" }: { href: string; children: React.ReactNode; variant?: "light" | "outline" | "dark" }) {
  const cls = {
    light: "bg-zinc-50 hover:bg-zinc-200 text-zinc-800",
    outline: "border border-white/40 text-white hover:bg-white/10",
    dark: "bg-zinc-950 hover:bg-zinc-800 text-white",
  }[variant];
  return (
    <Link href={href} className={`px-6 py-2.5 rounded-md text-sm font-medium transition ${cls}`}>
      {children}
    </Link>
  );
}

export function Stat({ value, label, note }: { value: string; label: string; note?: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-4xl md:text-5xl text-zinc-900 font-[family-name:var(--font-urbanist)]">{value}</span>
      <span className="text-xs tracking-wide uppercase text-zinc-600 mt-3">{label}</span>
      {note && <span className="text-xs text-zinc-400 mt-1 max-w-48">{note}</span>}
    </div>
  );
}

export function Card({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return <div className={`bg-white border border-zinc-200 rounded-xl p-5 md:p-6 ${className}`}>{children}</div>;
}

const TONES = {
  good: "border-emerald-200 bg-emerald-50 text-emerald-900",
  bad: "border-rose-200 bg-rose-50 text-rose-900",
  warn: "border-amber-200 bg-amber-50 text-amber-900",
  info: "border-sky-200 bg-sky-50 text-sky-900",
};
export function Callout({ tone = "info", title, children }: { tone?: keyof typeof TONES; title: string; children: React.ReactNode }) {
  return (
    <div className={`rounded-xl border p-4 md:p-5 ${TONES[tone]}`}>
      <p className="font-medium text-sm md:text-base">{title}</p>
      <div className="text-sm mt-1.5 leading-relaxed opacity-90">{children}</div>
    </div>
  );
}

/** Row of buttons that switches a view. */
export function Toggle<T extends string>({ options, value, onChange, label }: { options: { id: T; label: string }[]; value: T; onChange: (v: T) => void; label?: string }) {
  return (
    <div className="flex flex-wrap items-center gap-2" role="tablist" aria-label={label}>
      {options.map((o) => (
        <button
          key={o.id}
          role="tab"
          aria-selected={o.id === value}
          onClick={() => onChange(o.id)}
          className={`px-4 py-2 rounded-full text-sm transition cursor-pointer border ${
            o.id === value ? "bg-zinc-900 text-white border-zinc-900" : "bg-white text-zinc-700 border-zinc-200 hover:border-zinc-400"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export type Bar = { label: string; value: number | null; sub?: string; color?: string };

/** Horizontal bar chart in plain SVG-free HTML. `domain` fixes the scale so toggled views stay honest. */
export function HBars({ bars, domain, fmt = (v: number) => v.toFixed(3), unit = "" }: { bars: Bar[]; domain?: [number, number]; fmt?: (v: number) => string; unit?: string }) {
  const vals = bars.map((b) => b.value).filter((v): v is number => v !== null);
  const lo = domain ? domain[0] : Math.min(0, ...vals);
  const hi = domain ? domain[1] : Math.max(...vals, 1e-9);
  const pos = (v: number) => Math.max(0, Math.min(100, ((v - lo) / (hi - lo || 1)) * 100));
  const zero = pos(0);
  return (
    <div className="flex flex-col gap-3">
      {bars.map((b) => (
        <div key={b.label} className="grid grid-cols-[minmax(7rem,16rem)_1fr_auto] items-center gap-3 text-sm">
          <div className="text-zinc-700 truncate" title={b.label}>
            {b.label}
            {b.sub && <span className="block text-xs text-zinc-400 truncate">{b.sub}</span>}
          </div>
          <div className="relative h-6 bg-zinc-100 rounded">
            {b.value !== null && (
              <div
                className="absolute top-0 h-full rounded transition-all duration-500"
                style={{
                  left: `${Math.min(zero, pos(b.value))}%`,
                  width: `${Math.abs(pos(b.value) - zero)}%`,
                  background: b.color ?? "#18181b",
                }}
              />
            )}
            {lo < 0 && <div className="absolute top-0 h-full w-px bg-zinc-400" style={{ left: `${zero}%` }} />}
          </div>
          <div className="tabular-nums text-zinc-900 w-20 text-right">{b.value === null ? "n/a" : fmt(b.value) + unit}</div>
        </div>
      ))}
    </div>
  );
}

export function DataTable({ head, rows, highlight, align }: { head: string[]; rows: (string | number | React.ReactNode)[][]; highlight?: number[]; align?: ("l" | "r")[] }) {
  return (
    <div className="overflow-x-auto border border-zinc-200 rounded-xl bg-white">
      <table className="w-full text-sm">
        <thead className="bg-zinc-50 text-zinc-600">
          <tr>
            {head.map((h, i) => (
              <th key={h} className={`px-4 py-3 font-medium whitespace-nowrap ${(align?.[i] ?? (i === 0 ? "l" : "r")) === "l" ? "text-left" : "text-right"}`}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, ri) => (
            <tr key={ri} className={`border-t border-zinc-100 ${highlight?.includes(ri) ? "bg-sky-50/60 font-medium" : ""}`}>
              {r.map((c, ci) => (
                <td key={ci} className={`px-4 py-2.5 tabular-nums ${(align?.[ci] ?? (ci === 0 ? "l" : "r")) === "l" ? "text-left" : "text-right"}`}>
                  {c}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function CopyBlock({ title, code }: { title?: string; code: string }) {
  const [done, setDone] = useState(false);
  return (
    <div className="rounded-xl overflow-hidden border border-zinc-800 bg-zinc-950 text-zinc-100">
      <div className="flex items-center justify-between px-4 py-2 border-b border-zinc-800 text-xs text-zinc-400">
        <span>{title ?? "PowerShell"}</span>
        <button
          className="flex items-center gap-1.5 hover:text-white cursor-pointer"
          onClick={() => {
            navigator.clipboard?.writeText(code).then(() => {
              setDone(true);
              setTimeout(() => setDone(false), 1500);
            });
          }}
        >
          {done ? <Check size={14} /> : <Copy size={14} />}
          {done ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="p-4 text-xs md:text-[13px] leading-relaxed overflow-x-auto font-[family-name:var(--font-geist-mono)]">{code}</pre>
    </div>
  );
}

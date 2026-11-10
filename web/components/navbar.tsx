"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import React, { useEffect, useState } from "react";
import { Menu, X } from "lucide-react";
import { Logo } from "@/components/ui";
import { NAV } from "@/lib/nav";

export function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const path = usePathname() || "/";

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 50);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const active = (href: string) => (href === "/" ? path === "/" : path.startsWith(href));
  const linkCls = (href: string) =>
    `transition-colors duration-300 ${scrolled ? "text-zinc-800 hover:text-zinc-500" : "text-white/80 hover:text-white"} ${
      active(href) ? (scrolled ? "font-semibold text-zinc-950" : "font-semibold text-white") : ""
    }`;

  return (
    <>
      <nav
        className={`fixed z-50 flex items-center justify-between left-1/2 -translate-x-1/2 transition-all duration-500 p-4 ${
          scrolled
            ? "lg:w-5xl w-[calc(100vw-14px)] bg-white/70 backdrop-blur-2xl rounded-full mt-4 pl-6 shadow"
            : "md:px-16 lg:px-24 xl:px-32 w-full"
        }`}
      >
        <Link href="/" aria-label="Home">
          <Logo dark={scrolled} />
        </Link>
        <div className="hidden md:flex items-center gap-5 lg:gap-8 text-sm">
          {NAV.map((n) => (
            <Link key={n.href} href={n.href} className={linkCls(n.href)}>
              {n.label}
            </Link>
          ))}
        </div>
        <Link
          href="/results/"
          className={`hidden md:block px-6 py-2.5 text-sm font-medium transition-all duration-500 ${
            scrolled ? "bg-zinc-900 text-white hover:bg-zinc-800 rounded-full" : "bg-zinc-50 text-zinc-800 hover:bg-zinc-200 rounded-[6px]"
          }`}
        >
          See the numbers
        </Link>
        <button
          onClick={() => setMobileOpen(true)}
          aria-label="Open menu"
          className={`md:hidden p-2 rounded-md aspect-square cursor-pointer ${scrolled ? "text-zinc-800" : "text-white"}`}
        >
          <Menu size={24} />
        </button>
      </nav>

      <div
        className={`${
          mobileOpen ? "w-full" : "w-0"
        } md:hidden fixed top-0 z-50 left-0 transition-all duration-300 overflow-hidden h-full bg-zinc-950/95 backdrop-blur flex flex-col items-center justify-center gap-6 text-base`}
      >
        {NAV.map((n) => (
          <Link key={n.href} href={n.href} onClick={() => setMobileOpen(false)} className="text-white/90 hover:text-white">
            {n.label}
          </Link>
        ))}
        <button onClick={() => setMobileOpen(false)} aria-label="Close menu" className="bg-white text-zinc-800 p-2 rounded-md aspect-square cursor-pointer">
          <X size={22} />
        </button>
      </div>
    </>
  );
}

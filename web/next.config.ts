import type { NextConfig } from "next";

// Static export: the site is plain HTML/JS built from aggregate numbers only (no server, no analytics).
const nextConfig: NextConfig = {
  output: "export",
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;

import type { Metadata } from "next";
// Geist Mono only. Geist Sans used to be loaded and stamped on <html> too, but no token ever
// referenced --font-geist-sans: --font-sans is Jakarta and --font-serif is Lora, so it was a whole
// webfont family downloaded by every visitor and used by nothing.
import { GeistMono } from "geist/font/mono";
import { Lora, Plus_Jakarta_Sans } from "next/font/google";
import { Providers } from "@/components/providers";
import { PROVIDER_IDS } from "@/config/providers";
import "./globals.css";

const lora = Lora({ subsets: ["latin"], variable: "--font-lora", display: "swap" });
const jakarta = Plus_Jakarta_Sans({ subsets: ["latin"], variable: "--font-jakarta", display: "swap" });

export const metadata: Metadata = {
  title: "Gratis - Free LLM Market",
  // Counted from the registry, not typed out. It said "7 providers" through two provider additions.
  description: `Real-time market intelligence for free LLM models across ${PROVIDER_IDS.length} providers.`,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // suppressHydrationWarning is required: next-themes stamps data-theme on <html> before React
    // hydrates, so the server and client markup legitimately differ on that attribute.
    <html lang="en" suppressHydrationWarning className={`${GeistMono.variable} ${lora.variable} ${jakarta.variable}`}>
      <body><Providers>{children}</Providers></body>
    </html>
  );
}

# DESIGN.md - Gratis

The design contract, transcribed from what actually ships. Every token below is read from
`src/app/globals.css`; every dependency from `package.json`. If the two disagree, the code is right
and this file is a bug.

> The previous version of this document described a cool indigo-violet Linear palette (hue 265),
> shadcn/ui and SWR. None of that is in the app: the palette was replaced by a warm clay-on-ivory
> system, the components are Base UI, and the data layer is TanStack Query. A design doc that
> contradicts the product is worse than no design doc, because people build against it.

---

## 1. Identity

**Product:** real-time market intelligence for free LLM models across every provider in
`src/config/providers.ts`, plus an OpenAI-compatible proxy and a chat client for them. The count is
derived from the registry wherever it is shown - it was hardcoded as "7 providers" and stayed that
way through two additions.

**Personality:** warm-editorial density. Information-first, but not cold - a serif display face over
a dense data table, on ivory rather than clinical grey.

**Emotional target:** technical · precise · trustworthy · unhurried.

**Two surfaces, one system.** The market and the chat used to run different design systems - a cool
dark dashboard and a warm ivory `.theme-editorial` that hardcoded light values and ignored the theme
toggle entirely. They are now the same tokens, and both honour light and dark.

**Honesty is a design rule here, not a nicety.** This product's whole claim is that its numbers mean
something, so the UI must never render an inference as if it were a measurement:
- an unknown parameter count shows nothing, never `1B`
- a lens ranked on a provider-class prior says so, inline (see `TaskLens`)
- an estimated token count is prefixed `~`
- a contested score carries a confidence dot with its bench count and cross-benchmark agreement

---

## 2. Color

OKLCH throughout. **Light is the base** (warm ivory); dark (espresso) is the override on
`:root[data-theme="dark"]`. `next-themes` stamps `data-theme` on `<html>` before paint, so there is
no flash and no `mounted` guard anywhere.

### Semantic tokens

| Token | Light | Dark |
|---|---|---|
| `--color-bg` | `oklch(0.985 0.010 78)` | `oklch(0.16 0.010 55)` |
| `--color-fg` | `oklch(0.21 0.012 55)` | `oklch(0.94 0.010 75)` |
| `--color-fg-muted` | `oklch(0.36 0.014 55)` | `oklch(0.80 0.012 70)` |
| `--color-fg-subtle` | `oklch(0.47 0.015 60)` | `oklch(0.70 0.014 65)` |
| `--color-fg-disabled` | `oklch(0.68 0.014 65)` | `oklch(0.45 0.012 55)` |
| `--color-surface-1` | `oklch(0.978 0.011 76)` | `oklch(0.21 0.012 55)` |
| `--color-surface-2` | `oklch(0.94 0.012 74)` | `oklch(0.26 0.012 55)` |
| `--color-surface-3` | `oklch(0.90 0.014 72)` | `oklch(0.30 0.012 55)` |
| `--color-border` | `oklch(0.89 0.014 70)` | `oklch(0.32 0.014 55)` |
| `--color-border-strong` | `oklch(0.64 0.016 65)` | `oklch(0.53 0.014 55)` |

### Accent - clay, hue 45

| Token | Light | Dark |
|---|---|---|
| `--color-accent` | `oklch(0.56 0.14 45)` | `oklch(0.70 0.15 45)` |
| `--color-accent-hover` | `oklch(0.52 0.135 45)` | `oklch(0.75 0.155 45)` |
| `--color-accent-fg` | `oklch(0.99 0.008 78)` | `oklch(0.14 0.020 50)` |
| `--color-accent-soft` | `accent / 0.12` | `accent / 0.16` |

Single accent system. `--color-accent-hover` exists because the primary button's hover was once a
hardcoded hue-265 blue left behind by the palette that no longer exists.

### Status - each solid + soft

| Role | Hue | Light | Dark |
|---|---:|---|---|
| success (moss) | 150 | `oklch(0.53 0.125 150)` | `oklch(0.72 0.13 150)` |
| warning (amber) | 75 / 80 | `oklch(0.72 0.15 75)` | `oklch(0.83 0.14 80)` |
| danger (rust) | 28 | `oklch(0.55 0.19 28)` | `oklch(0.72 0.15 28)` |
| info (the one cool hue) | 235 | `oklch(0.52 0.11 235)` | `oklch(0.74 0.10 235)` |

### Provider tokens - avatars and charts only, never UI accents

`openrouter` 35 · `ollama` 200 · `aistudio` 90 · `groq` 25 · `cerebras` 290 · `cloudflare` 60 ·
`local` 145 · `nvidia` 130.

`local` is your own machine. Its green is deliberately the quietest of the set: it is the only tier
that is free without qualification, and it should read as calm, not as a status badge. `nvidia` sits
next to it in hue on purpose - NVIDIA's own brand green - but at much higher chroma, so the two
never read as the same provider at a glance.

---

## 3. Typography

Two families plus a mono. `--font-sans` **Plus Jakarta Sans** · `--font-serif` **Lora** (the `.serif`
class, used on display headings only) · `--font-mono` **Geist Mono** (all quantitative data).

| Token | Size | Line height |
|---|---:|---:|
| `--text-2xs` | 11px | 1.35 |
| `--text-xs` | 12px | 1.4 |
| `--text-sm` | 14px | 1.45 |
| `--text-base` | 15px | 1.55 |
| `--text-lg` | 17px | 1.4 |
| `--text-xl` | 20px | 1.3 |
| `--text-2xl` | 24px | 1.25 |
| `--text-3xl` | 28px | 1.15 |
| `--text-4xl` | 32px | 1.1 |

**11px is a hard floor.** The scale before this one was defined and then ignored: 126 call sites
wrote arbitrary values like `text-[8px]`, which is why the whole app read as microscopic. Nothing
smaller than `--text-2xs` ships.

Body is 15px with `-0.011em` tracking, set once in `@layer base`. Quantitative data is always
monospace, right-aligned, `tabular-nums`.

---

## 4. Spacing, radius, layout

4px grid. Radius is **sharp** on purpose - this is a data terminal, and round corners fight a grid:
`--radius-sm 2 · md 4 · lg 6 · xl 8 · 2xl 12`. Four tokens govern every card in the app.

| Container | Value | Use |
|---|---:|---|
| `--width-market` | 1280px | data surfaces, sized to the table's real content |
| `--width-prose` | 1100px | chat, archive, settings |
| `--container-measure` | 65ch | assistant replies |
| `--container-drawer` | 440px | detail drawer |
| `--container-dialog` | 640px | command palette, model picker |
| `--container-popover` | 400px | help sheet, confirm dialogs |

Shell: sidebar 240px (resizable 200-400, persisted), header 48px sticky, content padding 24px.

**Market table column budget** lives in tokens so it can be reasoned about together:
`rank 40 · model 416 · score 140 · signals (flex) · cost 96 · action 48`. Model is fixed and wide
enough that names stop truncating; **signals is the column that absorbs slack**, and its meters fill
it. Getting this backwards is what produced ~680px of dead space between the last bar and the cost
column.

---

## 5. Motion

| Token | Value | Use |
|---|---:|---|
| `DURATION.fast` | 120ms | hover, colour, opacity |
| `DURATION.base` | 180ms | enter |
| `DURATION.slow` | 250ms | drawer, sheet |
| `--animate-shimmer` | 1.4s linear infinite | skeletons - the only linear easing allowed |

Animate `transform` and `opacity` only. `prefers-reduced-motion: reduce` zeroes every duration with
`!important` in `@layer base`, and overlays additionally carry `motion-reduce:transition-none`.

---

## 6. Elevation and z-index

Dark mode uses surface progression, not shadows. Only two shadows exist, both for floating layers:
`--shadow-popover` and `--shadow-drawer`.

```
tooltip           1070   InfoTips float above everything, including the palette
command           1065   ⌘K must clear any open modal
command-backdrop  1060
modal             1050   drawer · sheet · dialog panels
modal-backdrop    1040
sticky            1020   header
```

**No ties.** A tie makes stacking depend on DOM order, which breaks silently. The old code mixed two
scales - header at 1020, drawer backdrop at 70 - so opening the drawer dimmed the page and left the
header blazing above it.

---

## 7. Component contracts

| Component | Contract |
|---|---|
| Model table | 2 views: **Decide** (rank · model · lens score · signal meters · cost) and **Audit** (every score dimension). Rows are buttons: click and Enter/Space open the drawer; the nested chat action stops propagation. Sortable headers in **both** views |
| Sort headers | A header that cannot sort must not look sortable - no pointer cursor, no hover, and `aria-sort` is always set |
| Task lens | 6 lenses drive the primary sort and the headline score. A lens ranked on inferred data renders an inline warning, not a tooltip |
| Detail drawer | 440px right slide, tabs Overview · Code · Metrics, copyable cURL + Python, "Open in chat" |
| Command palette | ⌘K · Actions · Navigation · Recent chats · Models · Chat with… |
| Chat | Shell owns the chrome; the conversation owns its toolbar (model picker + context meter). Switching model mid-thread **forks** into a new thread |
| Vault indicator | Permanent header chrome on every surface, with inline unlock. The vault is in-memory by design, so a reload locks it - that must be visible before you send a message, not discovered from a 401 |
| Every container | has an empty state, a loading state (skeletons, not spinners), and a distinct error state. "Backend unreachable" and "filters match nothing" are different screens |
| Every surface | is wrapped in an `ErrorBoundary` scoped to the content, so a crash costs you the surface, not the app |

---

## 8. Accessibility

- Focus ring on every interactive element: `2px solid var(--color-accent)`, offset 2px, set once in
  `@layer base` via `:focus-visible`.
- Interactive targets are ≥ 28px tall with padding to a comfortable hit area; table rows are 40px.
- `aria-sort` on sortable headers, `aria-pressed` on toggles and chips, `role="switch"` +
  `aria-checked` on filter switches, `role="alert"` on error states, `aria-label` on every icon-only
  button.
- Contrast: body text is `fg` on `bg` in both themes; `fg-subtle` is reserved for non-essential
  labels that are never the only carrier of meaning.
- Icons are Lucide. Never emoji.

---

## 9. Responsive

| Breakpoint | Behaviour |
|---|---|
| `< 640` | KPI strip 2-up, lens grid 2-up, table scrolls horizontally in its own container |
| `640-1023` | KPI 3-up, lens 3-up |
| `≥ 1024` | KPI 5-up, lens 6-up, full table |

The table scrolls inside `overflow-x-auto`; the page body never scrolls sideways.

---

## 10. Stack

| Layer | Choice |
|---|---|
| Framework | Next.js 16.2.10 · React 19.2.4 |
| Styling | Tailwind v4, CSS-first `@theme` |
| Components | **Base UI** (`@base-ui-components/react`, `@base-ui/react`) + `cmdk` |
| Table | TanStack Table v8 |
| Server state | **TanStack Query v5** |
| Client state | Zustand v5 with `persist` |
| Icons | Lucide React |
| Fonts | Plus Jakarta Sans · Lora · Geist Mono |
| Theme | `next-themes`, `attribute="data-theme"` |
| Chat | Vercel AI SDK v6 (`ai`, `@ai-sdk/react`, `@ai-sdk/openai-compatible`) |
| Storage | IndexedDB via `idb` |
| Motion | CSS transitions only |
| Deploy | OpenNext → Cloudflare Workers |

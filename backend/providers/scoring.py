"""Composite model scoring.

Turns the raw market + intelligence signals (AA intelligence/coding/math index, throughput, tool-call,
reasoning, price, context, uptime) into normalized 0..100 per-dimension scores, task-fit presets, an
overall rank, and an archetype label. Pure over its input list and fully fail-open: a model missing
intel / tps / price still scores on what it has, so the ranking never crashes on a sparse row.

Design: one number can't capture "cheap but dumb" vs "smart but slow" vs "great tool-caller". So every
model gets a vector of dimension scores; a consumer sorts by whichever dimension its use-case needs
(fit_code for a coding agent, fit_budget for bulk classification, fit_agent for tool use, ...).
"""

from __future__ import annotations

import math
import statistics
from typing import Any

Model = dict[str, Any]

# The AA intelligence/coding/math indices are a calibrated ABSOLUTE scale (~0..60), so they're scored
# against a fixed ceiling, not min-max within the pool — a 50-intel model is "50-smart" no matter what
# else is in the list. Only pool-relative concepts (speed, value, affordability) use min-max.
INTEL_CEIL = 65.0

# When throughput isn't measured (groq/cerebras/cloudflare report no tps), fall back to a provider-CLASS
# speed prior instead of a flat neutral-low default. groq (LPU) and cerebras (wafer-scale) are
# speed-first by architecture — ranking them neutral-low is the one number a zero-cost/fast-model tool
# must not get wrong. A genuinely unknown provider stays at true-neutral 50.
_PROVIDER_SPEED_PRIOR: dict[str, float] = {
    "groq": 88.0,
    "cerebras": 88.0,
    "cloudflare workers ai": 62.0,
}


def _speed_prior(m: "Model") -> float:
    return _PROVIDER_SPEED_PRIOR.get(str(m.get("provider") or "").lower(), 50.0)


# Which community benchmarks (0..1, from AA's per-model evals) measure each dimension. A dimension is
# the MEAN of the benches present for it — fail-open, so a model missing a bench just drops it from the
# mean. Grounds coding/math/reasoning/tool-use in multiple verified benches (LiveCodeBench, SciCode,
# Terminal-Bench, tau-bench, GPQA, HLE, AIME, MMLU-Pro, IFBench) instead of one aggregate index.
_DIM_BENCHES: dict[str, list[str]] = {
    "coding": ["livecodebench", "scicode", "terminalbench_hard"],
    "math": ["math_500", "aime_25", "aime"],
    "reasoning": ["gpqa", "hle", "lcr"],
    "agentic": ["tau2", "tau_banking", "terminalbench_hard"],
    "instruction": ["ifbench"],
    "knowledge": ["mmlu_pro"],
}


# Benchmark governance — how contamination-resistant / honest each bench is. Dynamic/live benches
# (fresh problems harvested AFTER training cutoffs) are the trustworthy signal and carry full weight;
# static benches contaminate and saturate over time and are down-weighted; MMLU is near-saturated
# (~92-94% at the frontier, variance within noise) so it barely counts. Basis: LiveBench/LiveCodeBench
# refresh monthly; MMLU saturation is a documented 2026 finding.
_BENCH_GOVERNANCE: dict[str, float] = {
    # dynamic / contamination-resistant (harvested post-cutoff, rotates) — full weight
    "livecodebench": 1.0, "terminalbench_hard": 1.0, "terminalbench_v2_1": 1.0,
    "aime_25": 1.0, "tau2": 1.0, "tau_banking": 1.0, "scicode": 0.9, "lcr": 0.9,
    # GPQA-Diamond (#4 cleanest, most discriminative model-vs-model in 2026, holds up vs web search) and
    # HLE (#3 cleanest) are top-tier honest signals — weighted near dynamic.
    "gpqa": 0.9, "hle": 0.9, "ifbench": 0.8, "math_500": 0.7, "aime": 0.6,
    # saturated / low-signal at the frontier
    "mmlu_pro": 0.4,
}
# The 0..1 quality benches used for the cross-benchmark consensus (agreement) signal.
_QUALITY_BENCHES = tuple(_BENCH_GOVERNANCE)


def _bench_weight(key: str) -> float:
    return _BENCH_GOVERNANCE.get(key, 0.7)


def _bench_blend(benches: dict, keys: list[str]) -> float | None:
    """Governance-WEIGHTED mean of the available 0..1 benchmarks (→0..100). Contamination-resistant
    benches count more; saturated ones (mmlu_pro) barely move the score. None when none present."""
    num = den = 0.0
    for k in keys:
        v = benches.get(k)
        if v is not None:
            w = _bench_weight(k)
            num += _clamp(v * 100.0) * w
            den += w
    return num / den if den else None


def _consensus(benches: dict) -> float | None:
    """Cross-benchmark agreement 0..100. Low spread across a model's benches = the composite is
    trustworthy; wide disagreement (great on one, weak on another) = a contested score you should not
    take at face value. The honest move is to SURFACE the disagreement, not average it into silence.
    None when too few benches to judge."""
    vals = [_clamp(benches[k] * 100.0) for k in _QUALITY_BENCHES if benches.get(k) is not None]
    if len(vals) < 3:
        return None
    return round(_clamp(100.0 - statistics.pstdev(vals) * 2.2), 1)


def _confidence(bench_count: int, consensus: float | None, intel_est: bool) -> str:
    """How much to trust this row: grounded in many agreeing benches → high; thin or estimated → low."""
    if intel_est or bench_count < 3:
        return "low"
    if bench_count >= 8 and (consensus is None or consensus >= 55):
        return "high"
    return "medium"


def _clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def _log10p(x: float | None) -> float:
    """log10(1 + max(x, 0)) — safe against the junk the upstream APIs sometimes send (negative ctx, a
    -1 BYOK price sentinel, None) that would otherwise blow up math.log10 with a domain error."""
    return math.log10(max(x or 0.0, 0.0) + 1.0)


def _minmax(vals: list[float | None]) -> tuple[float, float]:
    present = [v for v in vals if v is not None]
    if not present:
        return 0.0, 1.0
    return min(present), max(present)


def _default(x: float | None, fallback: float) -> float:
    """`x` unless it is genuinely absent. Distinct from `x or fallback`, which also swallows 0.0 -
    and 0.0 is a real score here (worst in the pool), not a missing one."""
    return fallback if x is None else x


def _norm(x: float | None, lo: float, hi: float) -> float | None:
    """Min-max to 0..100, or None when x is absent. hi==lo → mid (50) so a degenerate range is neutral."""
    if x is None:
        return None
    if hi <= lo:
        return 50.0
    return _clamp((x - lo) / (hi - lo) * 100.0)


def _blended_price_per_mtok(m: Model) -> float | None:
    """Blended $/1M tokens (3:1 input:output), or None when the model carries no usable price. Some
    providers use a negative sentinel (e.g. -1 for BYOK/variable) — treat non-positive as unknown so the
    downstream log10 never hits a domain error."""
    pin, pout = m.get("price_in"), m.get("price_out")
    if pin is None and pout is None:
        return None
    p = ((pin or 0.0) * 0.75 + (pout or 0.0) * 0.25) * 1_000_000
    return p if p > 0 else None


def _raw_intel(m: Model) -> float | None:
    """Real AA intelligence when present, else the capability heuristic as a stand-in (flagged elsewhere
    by intel_est). Kept on separate scales but each is normalized within its own population."""
    if m.get("intel") is not None:
        return m["intel"]
    return m.get("capability")


def score_models(models: list[Model]) -> list[Model]:
    """Attach `scores` (dimension + fit + overall), `archetype`, and `badges` to each model; return the
    list sorted by overall desc. Non-destructive-ish: copies each model dict."""
    if not models:
        return []

    # ── pass 1: raw per-model signals (pre-normalization) ──
    raws: list[dict[str, float | None]] = []
    for m in models:
        tps = m.get("tps")
        price = _blended_price_per_mtok(m)
        intel = _raw_intel(m)
        # value = intelligence per dollar; free lanes get the max raw value, paid = intel / $/Mtok
        if m.get("is_free") or price in (None, 0):
            r_value = None  # marks "free/unknown" → resolved to top affordability below
        else:
            r_value = (intel or 1.0) / price
        raws.append({
            "intel": intel,
            "speed": _log10p(tps) if tps else None,
            "coding": m.get("intel_coding"),
            "math": m.get("intel_math"),
            "ctx": _log10p(m.get("ctx")),
            "value": r_value,
            "price": price,
        })

    # ── ranges across the set (pool-relative signals only) ──
    cap_lo, cap_hi = _minmax([m.get("capability") for m in models])
    speed_lo, speed_hi = _minmax([r["speed"] for r in raws])
    ctx_lo, ctx_hi = _minmax([r["ctx"] for r in raws])
    val_lo, val_hi = _minmax([_log10p(r["value"]) for r in raws if r["value"] is not None])
    # affordability: invert price (cheaper = higher); free handled as top
    inv_prices = [(-math.log10(r["price"] + 0.001)) for r in raws if r["price"]]
    aff_lo, aff_hi = _minmax(inv_prices)

    out: list[Model] = []
    for m, r in zip(models, raws):
        # intelligence: real AA index on an absolute ceiling; else a capability-derived estimate held to
        # a conservative 0..60 band so a heuristic guess never outranks a measured flagship.
        ri = m.get("intel")
        if ri is not None:
            intelligence = _clamp(ri / INTEL_CEIL * 100.0)
        else:
            intelligence = 0.6 * _default(_norm(m.get("capability"), cap_lo, cap_hi), 0.0)
        # Dimensions grounded in the real benches; fall back to AA's aggregate index (already 0..100),
        # then to general intelligence. Coverage: coding/reasoning/agentic have strong bench coverage.
        benches = m.get("benches") or {}
        coding = _bench_blend(benches, _DIM_BENCHES["coding"])
        if coding is None and m.get("intel_coding") is not None:
            coding = _clamp(m["intel_coding"])
        coding = intelligence if coding is None else coding
        maths = _bench_blend(benches, _DIM_BENCHES["math"])
        if maths is None and m.get("intel_math") is not None:
            maths = _clamp(m["intel_math"])
        maths = intelligence if maths is None else maths
        reasoning_bench = _bench_blend(benches, _DIM_BENCHES["reasoning"])
        agentic_bench = _bench_blend(benches, _DIM_BENCHES["agentic"])
        instruction = _bench_blend(benches, _DIM_BENCHES["instruction"])
        knowledge = _bench_blend(benches, _DIM_BENCHES["knowledge"])
        speed = _norm(r["speed"], speed_lo, speed_hi)
        # `speed_est` marks the difference between MEASURED throughput and an inferred one, the same
        # way `intel_est` already does for intelligence. 244 of 289 live models fall back to a prior,
        # and the UI was rendering every one of them as if it were a measurement.
        speed_est = speed is None
        if speed_est:  # throughput not measured → provider-class prior (fast free lanes not punished)
            speed = _speed_prior(m)
        context = _default(_norm(r["ctx"], ctx_lo, ctx_hi), 0.0)

        if m.get("is_free"):
            affordability = 100.0
            value = _clamp(0.5 * 100.0 + 0.5 * intelligence)  # free + smart tops value; free + dumb mid
        elif r["price"]:
            # `_default(x, 50.0)`, not `x or 50.0`. The most expensive model in the pool normalizes to
            # exactly 0.0 - which is falsy - so `or` silently promoted the worst affordability (and
            # the worst value) in the market to the NEUTRAL 50, the same score meaning "price
            # unknown". Only a genuinely absent value may fall back.
            affordability = _default(_norm(-math.log10(r["price"] + 0.001), aff_lo, aff_hi), 50.0)
            value = _default(_norm(_log10p(r["value"]), val_lo, val_hi), 50.0)
        else:
            affordability = 60.0  # unknown price → cautiously neutral
            value = 50.0

        # tool_use: the MEASURED agentic score (tau-bench / terminal-bench) when available — a real
        # number, not a boolean guess; else fall back to the tools flag gated by intelligence.
        if agentic_bench is not None:
            tool_use = agentic_bench
        elif m.get("tools"):
            tool_use = intelligence
        else:
            tool_use = intelligence * 0.25
        # reasoning: grounded in GPQA/HLE/long-context when measured (reasoning models score high there),
        # else the math dimension as a proxy.
        reasoning = reasoning_bench if reasoning_bench is not None else maths
        instruction = intelligence if instruction is None else instruction
        knowledge = intelligence if knowledge is None else knowledge
        # uptime arrives either as a fraction (0..1) or a percentage (0..100) depending on the provider —
        # normalize both, then clamp, so a 99% uptime never explodes the composite.
        up = m.get("uptime")
        reliability = 70.0 if up is None else _clamp(up if up > 1 else up * 100.0)

        fit_chat = 0.50 * intelligence + 0.20 * speed + 0.20 * affordability + 0.10 * context
        fit_code = 0.55 * coding + 0.25 * intelligence + 0.20 * affordability
        fit_math = 0.55 * maths + 0.25 * reasoning + 0.20 * intelligence
        fit_agent = 0.50 * tool_use + 0.30 * intelligence + 0.20 * speed
        fit_budget = 0.60 * value + 0.40 * intelligence
        fit_fast = 0.60 * speed + 0.40 * intelligence
        # overall is a QUALITY ranking (validated to track human-preference Elo far better than the old
        # cost-heavy blend, which correlated ~0.12). Cost/speed live in fit_budget / fit_fast, not here —
        # a free model and a paid one of equal quality rank equal on overall.
        overall = _clamp(0.45 * intelligence + 0.15 * coding + 0.12 * reasoning
                         + 0.12 * tool_use + 0.10 * speed + 0.06 * reliability)

        scores = {
            "intelligence": round(intelligence, 1), "speed": round(speed, 1),
            "coding": round(coding, 1), "math": round(maths, 1), "value": round(value, 1),
            "affordability": round(affordability, 1), "tool_use": round(tool_use, 1),
            "reasoning": round(reasoning, 1), "instruction": round(instruction, 1),
            "knowledge": round(knowledge, 1), "context": round(context, 1),
            "reliability": round(reliability, 1),
            "fit_chat": round(fit_chat, 1), "fit_code": round(fit_code, 1),
            "fit_math": round(fit_math, 1), "fit_agent": round(fit_agent, 1),
            "fit_budget": round(fit_budget, 1), "fit_fast": round(fit_fast, 1),
            "overall": round(overall, 1),
        }
        consensus = _consensus(benches)
        out.append({**m, "scores": scores, "archetype": _archetype(m, scores),
                    "badges": _badges(m, scores), "bench_count": len(benches),
                    "consensus": consensus,
                    "speed_est": speed_est,
                    "confidence": _confidence(len(benches), consensus, bool(m.get("intel_est")))})

    out.sort(key=lambda x: x["scores"]["overall"], reverse=True)
    return out


def _archetype(m: Model, s: dict[str, float]) -> str:
    """Single standout label. Order matters: the most distinctive claim wins."""
    if s["intelligence"] >= 85:
        return "flagship"
    if s["coding"] - s["intelligence"] >= 12 and s["coding"] >= 65:
        return "code-specialist"
    if m.get("brain") and s["math"] >= 65:
        return "reasoner"
    if m.get("tools") and s["intelligence"] >= 60:
        return "agent"
    if s["speed"] >= 80 and s["intelligence"] >= 40:
        return "speedster"
    if s["value"] >= 80:
        return "budget"
    return "workhorse"


def _badges(m: Model, s: dict[str, float]) -> list[str]:
    b: list[str] = []
    if m.get("is_free"):
        b.append("free")
    if m.get("tools"):
        b.append("tools")
    if m.get("brain"):
        b.append("reasoning")
    if s["speed"] >= 75:
        b.append("fast")
    if s["value"] >= 75:
        b.append("great-value")
    if s["intelligence"] >= 80:
        b.append("high-intelligence")
    return b

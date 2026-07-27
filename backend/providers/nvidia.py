"""NVIDIA NIM (build.nvidia.com) as a provider.

NIM is OpenAI-compatible and routes through litellm's native `nvidia_nim/` provider, whose default
base is https://integrate.api.nvidia.com/v1.

Three things make this catalogue different from the other providers':

  the prefix is `nim/`, not     NVIDIA is an AUTHOR on OpenRouter, which serves ids like
  `nvidia/`                    `nvidia/nemotron-3-super-120b-a12b`. A `nvidia/` route prefix would
                               have been indistinguishable from those: the market would list them
                               twice, the provider filter would claim OpenRouter's rows as NIM's,
                               and resolve_model would send an OpenRouter model to NIM, where it
                               does not exist. `nim/` collides with none of OpenRouter's 57 authors
                               and matches litellm's own provider id.

  ids carry an author segment   `meta/llama-3.1-8b-instruct`, exactly like OpenRouter. Prefixed for
                               the market that becomes `nim/meta/llama-3.1-8b-instruct`: the first
                               segment is the ROUTE, the second is the AUTHOR, and they are
                               genuinely different facts.

  the catalogue is mixed        /v1/models lists embedders, rerankers, OCR, speech and image models
                               alongside chat ones. Those are not chat-completable, so shipping them
                               would fill the market with rows that can only ever fail. They are
                               filtered out here rather than left for the availability prober to
                               discover one wasted call at a time.

Unlike every other provider here, the catalogue needs no key - so the models are listed even on a
deployment that cannot call them. Inference still requires NVIDIA_NIM_API_KEY, and without one the
provider is skipped, matching the "providers without a key are skipped" contract.
"""

import math
import os
import re

import httpx

from providers.authors import author_from_model_id
from providers.params import capability_params, parse_params

NVIDIA_MODEL_PREFIX = "nim/"
NVIDIA_PROVIDER_NAME = "NVIDIA NIM"
NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_TIMEOUT_S = 10.0

# NIM does not publish a per-model context window anywhere in /v1/models. This is a conservative
# FLOOR used only so the capability heuristic has a number to work with - deliberately not the 128k
# most of these models actually support, because claiming a window we have not verified is the kind
# of confident-looking fiction this codebase keeps having to delete.
NVIDIA_DEFAULT_CTX = 32_768

# Families that are not chat-completable. Matched against the full id, so `nvidia/embed-qa-4` and
# `baai/bge-m3` both go, while `meta/llama-3.1-8b-instruct` stays.
_NON_CHAT_PATTERN = re.compile(
    r"embed|bge|rerank|retriever|ocr|paddle|whisper|riva|asr|tts|speech|vila|clip|molmo|esm"
    r"|diffusion|stable-|sdxl|flux|video|cosmos|genmol|protein|fold|dna|audio|parakeet|canary"
    r"|table-structure|graphic-elements|page-elements|chart|deplot|florence",
    re.I,
)

_REASONING_PATTERN = re.compile(
    r"deepseek-r1|reasoning|thinking|qwen3|gpt-oss|nemotron-.*-reason|magistral|kimi", re.I
)


def get_nvidia_api_key() -> str:
    """The name litellm itself reads, so a key set for one is set for both."""
    return os.getenv("NVIDIA_NIM_API_KEY", "")


def get_nvidia_base_url() -> str:
    return os.getenv("NVIDIA_NIM_API_BASE", NVIDIA_DEFAULT_BASE_URL).rstrip("/")


def is_nvidia_model(model_id: str) -> bool:
    return isinstance(model_id, str) and model_id.startswith(NVIDIA_MODEL_PREFIX)


def strip_nvidia_prefix(model_id: str) -> str:
    return model_id[len(NVIDIA_MODEL_PREFIX):] if is_nvidia_model(model_id) else model_id


def is_chat_model(model_id: str) -> bool:
    return not _NON_CHAT_PATTERN.search(model_id)


def _is_reasoning(model_id: str) -> bool:
    return bool(_REASONING_PATTERN.search(model_id))


async def fetch_nvidia_models() -> list[dict]:
    """The chat-capable catalogue, or [] when no key is configured.

    The listing endpoint is public, but a deployment with no key cannot actually CALL any of these -
    so it must not advertise them. Same contract as every other provider: no key, no rows.
    """
    if not get_nvidia_api_key():
        return []
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{get_nvidia_base_url()}/models",
                headers={"Authorization": f"Bearer {get_nvidia_api_key()}"},
                timeout=NVIDIA_TIMEOUT_S,
            )
            r.raise_for_status()
            rows = r.json().get("data", [])
    except Exception as e:
        print(f"NVIDIA NIM models error: {e}")
        return []

    return [m for m in rows if m.get("id") and is_chat_model(m["id"])]


def build_nvidia_market_stats(raw: list[dict]) -> list[dict]:
    result = []
    for m in raw:
        slug = m.get("id") or ""
        if not slug:
            continue
        params = parse_params(slug)
        ctx = NVIDIA_DEFAULT_CTX
        capability = capability_params(params) * math.log10(ctx + 1)
        result.append({
            "id": f"{NVIDIA_MODEL_PREFIX}{slug}",
            "name": slug,
            "params": params,
            "ctx": ctx,
            # Free by allocation, like Cloudflare Workers AI: a personal NVIDIA account comes with
            # free credits, and the endpoints are callable until those run out. Exhausted credits
            # surface as a rate-limit/quota error, which availability.py classifies as TRANSIENT -
            # so running out hides the models for a while rather than deleting them permanently.
            "is_free": True,
            "capability": capability,
            "brain": _is_reasoning(slug),
            # NIM publishes no per-model tool-calling flag, and the market's other key-based
            # providers (Groq, Cerebras) make the same optimistic assumption. Kept consistent rather
            # than quietly different; the agentic BENCH scores are what actually drive fit_agent.
            "tools": True,
            "open": True,
            # NVIDIA publishes no throughput figures, so this stays absent rather than estimated.
            "tps": None,
            "uptime": None,
            "provider": NVIDIA_PROVIDER_NAME,
            # The id's first segment is the author, exactly as on OpenRouter - so the row can say
            # who BUILT the model instead of only who serves it.
            "author": author_from_model_id(slug),
            "balanced": 0.0,
            "value": 0.0,
        })
    return result

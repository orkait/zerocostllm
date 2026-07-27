import math
import os
import re

import httpx

from providers.params import capability_params, parse_params

OLLAMA_CLOUD_BASE = "https://ollama.com/v1"
AISTUDIO_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
TIMEOUT_S = 10.0

OLLAMA_PROVIDER = "Ollama"
AISTUDIO_PROVIDER = "Google AI Studio"
OPENROUTER_PROVIDER = "OpenRouter"

DEFAULT_CTX_OLLAMA = 131_072
DEFAULT_CTX_AISTUDIO = 32_768



def _is_reasoning(model_id: str) -> bool:
    return bool(re.search(r"gpt-oss|deepseek-v[34]|deepseek-r1|qwen3|kimi-k2", model_id.lower()))


def _aistudio_params(slug: str) -> float | None:
    """Gemini's size, or None. Google does not publish parameter counts for Gemini.

    This used to invent them: flash-lite -> 4B, flash -> 8B, pro -> 70B. Those numbers came from
    nowhere and were rendered in the UI as if measured. If Google ever states a size in the model id
    it is used; otherwise the honest answer is that we do not know.
    """
    return parse_params(slug)


def _aistudio_ctx(slug: str) -> int:
    lower = slug.lower()
    if "gemini-2.5" in lower or "gemini-2.0" in lower:
        return 1_048_576
    if "gemma" in lower:
        return 8_192
    return DEFAULT_CTX_AISTUDIO


async def _fetch_ollama_models(client: httpx.AsyncClient) -> list[dict]:
    api_key = os.getenv("OLLAMA_API_KEY", "")
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        r = await client.get(f"{OLLAMA_CLOUD_BASE}/models", headers=headers, timeout=TIMEOUT_S)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        print(f"Ollama models error: {e}")
        return []


async def _fetch_aistudio_models(client: httpx.AsyncClient) -> list[dict]:
    api_key = os.getenv("GOOGLE_AISTUDIO_API_KEY", "")
    if not api_key:
        return []
    try:
        r = await client.get(
            f"{AISTUDIO_BASE}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=TIMEOUT_S,
        )
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        print(f"AI Studio models error: {e}")
        return []


async def _fetch_openrouter_models(client: httpx.AsyncClient) -> list[dict]:
    try:
        r = await client.get(OPENROUTER_MODELS_URL, timeout=TIMEOUT_S)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        print(f"OpenRouter models error: {e}")
        return []


def _normalize_aistudio_id(raw_id: str) -> str:
    return raw_id[len("models/"):] if raw_id.startswith("models/") else raw_id


def build_market_stats_from_ollama(raw: list[dict]) -> list[dict]:
    result = []
    for m in raw:
        slug = m["id"]
        params = parse_params(slug)
        ctx = DEFAULT_CTX_OLLAMA
        capability = capability_params(params) * math.log10(ctx + 1)
        brain = _is_reasoning(slug)
        result.append({
            "id": f"ollama/{slug}",
            "name": slug,
            "params": params,
            "ctx": ctx,
            "is_free": False,
            "capability": capability,
            "brain": brain,
            "tools": True,
            "open": True,
            "tps": None,
            "uptime": None,
            "provider": OLLAMA_PROVIDER,
            "balanced": 0.0,
            "value": 0.0,
        })
    return result


def build_market_stats_from_aistudio(raw: list[dict]) -> list[dict]:
    result = []
    for m in raw:
        slug = _normalize_aistudio_id(m["id"])
        lower = slug.lower()
        if "gemini" not in lower and "gemma" not in lower:
            continue
        params = _aistudio_params(slug)
        ctx = _aistudio_ctx(slug)
        capability = capability_params(params) * math.log10(ctx + 1)
        brain = "pro" in lower or "thinking" in lower
        # Pro-tier gemini models have no free quota (429 limit:0 on free keys,
        # billed on paid keys). Only flash / flash-lite / gemma are free-callable.
        is_free = "pro" not in lower
        result.append({
            "id": f"aistudio/{slug}",
            "name": slug,
            "params": params,
            "ctx": ctx,
            "is_free": is_free,
            "capability": capability,
            "brain": brain,
            "tools": True,
            "open": "gemma" in lower,
            "tps": None,
            "uptime": None,
            "provider": AISTUDIO_PROVIDER,
            "balanced": 0.0,
            "value": 0.0,
        })
    return result


async def fetch_unified_models() -> dict:
    import asyncio
    from .groq import fetch_groq_models
    from .cerebras import fetch_cerebras_models
    from .cloudflare import fetch_cloudflare_models
    from .local import fetch_local_models

    async with httpx.AsyncClient() as client:
        ollama_task = _fetch_ollama_models(client)
        aistudio_task = _fetch_aistudio_models(client)
        openrouter_task = _fetch_openrouter_models(client)
        ollama_raw, aistudio_raw, openrouter_raw, groq_raw, cerebras_raw, cloudflare_raw, local_raw = await asyncio.gather(
            ollama_task,
            aistudio_task,
            openrouter_task,
            fetch_groq_models(),
            fetch_cerebras_models(),
            fetch_cloudflare_models(),
            fetch_local_models(),
        )

    data = []
    for m in ollama_raw:
        slug = m["id"]
        data.append({"id": f"ollama/{slug}", "object": "model", "created": m.get("created"), "owned_by": "ollama"})
    for m in aistudio_raw:
        slug = _normalize_aistudio_id(m["id"])
        data.append({"id": f"aistudio/{slug}", "object": "model", "created": m.get("created"), "owned_by": "google"})
    for m in groq_raw:
        slug = m["id"]
        data.append({"id": f"groq/{slug}", "object": "model", "created": m.get("created"), "owned_by": "groq"})
    for m in cerebras_raw:
        slug = m["id"]
        data.append({"id": f"cerebras/{slug}", "object": "model", "created": m.get("created"), "owned_by": "cerebras"})
    for m in cloudflare_raw:
        slug = m.get("name", "")
        if not slug:
            continue
        data.append({"id": f"cloudflare/{slug}", "object": "model", "created": None, "owned_by": "cloudflare"})
    for m in local_raw:
        slug = m.get("id", "")
        if not slug:
            continue
        data.append({"id": f"local/{slug}", "object": "model", "created": m.get("created"), "owned_by": "local"})
    for m in openrouter_raw:
        data.append({"id": m["id"], "object": "model", "created": m.get("created"), "owned_by": "openrouter"})

    return {"object": "list", "data": data}


async def fetch_unified_market_stats() -> tuple[list[dict], list[dict]]:
    import asyncio

    async with httpx.AsyncClient() as client:
        ollama_raw, aistudio_raw = await asyncio.gather(
            _fetch_ollama_models(client),
            _fetch_aistudio_models(client),
        )

    return build_market_stats_from_ollama(ollama_raw), build_market_stats_from_aistudio(aistudio_raw)

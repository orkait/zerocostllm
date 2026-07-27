"""Local llama.cpp-compatible server as a provider.

The one genuinely zero-cost, zero-rate-limit, fully private tier: a model running on
your own GPU. llama-server already speaks the OpenAI API, so it routes through litellm
as `openai/<slug>` with an `api_base` - the same shape as the Cloudflare/Ollama paths.

Discovery is the health check: if nothing is listening on LOCAL_LLM_BASE_URL the fetch
returns [] and the provider simply disappears from the market, matching the "providers
without a key are skipped" contract. No key is needed at all - presence is the auth.
"""

import math
import os
import re

import httpx

from providers.params import capability_params, parse_params

LOCAL_MODEL_PREFIX = "local/"
LOCAL_PROVIDER_NAME = "Local"
# Short: a local box that is up answers instantly, and one that is down must not stall
# the market aggregation behind a network timeout.
LOCAL_TIMEOUT_S = 2.0
LOCAL_DEFAULT_CTX = 65_536


def get_local_base_url() -> str:
    """OpenAI-compatible base of the local server (llama-server, vLLM, LM Studio, ...)."""
    return os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8081/v1").rstrip("/")


def get_local_api_key() -> str:
    """llama-server ignores the key unless started with --api-key; send a placeholder so
    clients that require a non-empty Authorization header still work."""
    return os.getenv("LOCAL_LLM_API_KEY", "local")


def get_local_ctx() -> int:
    """Context the server was actually started with. Reported as-is: claiming a bigger
    window than llama-server was launched with would make the market lie."""
    raw = os.getenv("LOCAL_LLM_CTX", "")
    try:
        return int(raw) if raw else LOCAL_DEFAULT_CTX
    except ValueError:
        return LOCAL_DEFAULT_CTX


def get_local_tps() -> float | None:
    """Measured decode tok/s, if the operator recorded one. Unlike hosted providers this
    is ground truth for THIS machine, so it is worth surfacing when known."""
    raw = os.getenv("LOCAL_LLM_TPS", "")
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def is_local_model(model_id: str) -> bool:
    return isinstance(model_id, str) and model_id.startswith(LOCAL_MODEL_PREFIX)


def strip_local_prefix(model_id: str) -> str:
    return model_id[len(LOCAL_MODEL_PREFIX):] if is_local_model(model_id) else model_id


def _is_reasoning(model_id: str) -> bool:
    return bool(re.search(r"deepseek-r1|reasoning|thinking|qwen3|qwen3\.|bonsai|gpt-oss", model_id.lower()))


def _normalize_models(payload: dict) -> list[dict]:
    """Local servers disagree on the /v1/models shape. vLLM and LM Studio return the
    OpenAI form {"data": [{"id": ...}]}; llama-server returns an Ollama-flavoured
    {"models": [{"name": ..., "model": ...}]}. Accept both and emit {"id": ...} rows,
    so swapping the local backend does not silently empty the market.
    """
    rows = payload.get("data") or payload.get("models") or []
    out = []
    for m in rows:
        if not isinstance(m, dict):
            continue
        slug = m.get("id") or m.get("model") or m.get("name")
        if slug:
            out.append({"id": slug, "created": m.get("created")})
    return out


async def fetch_local_models() -> list[dict]:
    base = get_local_base_url()
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {get_local_api_key()}"},
                timeout=LOCAL_TIMEOUT_S,
            )
            r.raise_for_status()
            return _normalize_models(r.json())
    except Exception:
        # Server not running is the normal case, not an error worth logging on every
        # market refresh - the provider just does not appear.
        return []


# A reasoning model whose template opens a <think> block spends its first tokens on
# reasoning that llama-server splits into `reasoning_content`. If the caller's budget
# runs out before the block closes, `content` comes back EMPTY and the request looks
# broken - and the cliff is much higher than it looks. Measured on Ternary Bonsai 27B
# with a one-line coding task: max_tokens 24 and 700 both returned EMPTY content
# (finish=length), while 4096 answered correctly using 1281 tokens. This floor covers
# that worst case with headroom.
LOCAL_THINKING_MIN_TOKENS = 2048

# Output tokens on a local model are FREE and reasoning is the point of this model, so
# the DEFAULT is to leave the caller's token budget completely alone. Rewriting someone
# else's max_tokens is a surprise that belongs behind an explicit opt-in. Policies:
#   off     - (default) change nothing; the context window is the only real limit
#   raise   - lift a too-small cap to LOCAL_THINKING_MIN_TOKENS so the think block closes
#   disable - keep the caller's cap and turn thinking off (strict cap adherence)
#
# If a caller does send a tiny max_tokens they will get empty content, because the budget
# is spent inside <think>. That is the model behaving correctly, not a bug to paper over:
# the fix belongs in the caller's budget, or in LOCAL_THINKING_POLICY=raise.
LOCAL_THINKING_POLICY = "off"

# Left unchecked this model reasons at length: 3839 characters of thinking to write an
# iterative fibonacci. A brevity directive cut that to 1298 chars - 1281 -> 564 completion
# tokens and 47.9s -> 20.7s wall - with an identical, correct answer. Reasoning stays ON;
# it is simply told not to ramble.
LOCAL_CONCISE_DIRECTIVE = (
    "Reason privately and BRIEFLY: a few short steps, no restating the question, "
    "no exploring alternatives you will not use. Then answer directly."
)


def get_local_thinking_policy() -> str:
    policy = os.getenv("LOCAL_THINKING_POLICY", LOCAL_THINKING_POLICY).strip().lower()
    return policy if policy in {"raise", "disable", "off"} else LOCAL_THINKING_POLICY


def hide_local_reasoning() -> bool:
    """Whether to drop `reasoning_content` from local responses before returning.

    The model still reasons - that is where its quality comes from - the chain of
    thought simply is not shipped to the client. Clients that render the field (agent
    CLIs show it as a "Thinking..." block) otherwise bury the answer in scratch work.
    Set LOCAL_HIDE_REASONING=0 to pass it through.
    """
    return os.getenv("LOCAL_HIDE_REASONING", "1").strip().lower() not in {"0", "false", "no"}


def strip_reasoning(payload: dict) -> dict:
    """Remove reasoning fields from a non-streamed completion payload."""
    if not hide_local_reasoning() or not isinstance(payload, dict):
        return payload
    for choice in payload.get("choices") or []:
        msg = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(msg, dict):
            msg.pop("reasoning_content", None)
            msg.pop("reasoning", None)
    return payload


def strip_reasoning_chunk(chunk: dict) -> dict:
    """Same, for a streamed chunk (fields live on `delta`, not `message`)."""
    if not hide_local_reasoning() or not isinstance(chunk, dict):
        return chunk
    for choice in chunk.get("choices") or []:
        delta = choice.get("delta") if isinstance(choice, dict) else None
        if isinstance(delta, dict):
            delta.pop("reasoning_content", None)
            delta.pop("reasoning", None)
    return chunk


def get_local_reasoning_style() -> str:
    """`concise` (default) trims verbose chain-of-thought; `off` leaves prompts alone."""
    return os.getenv("LOCAL_REASONING_STYLE", "concise").strip().lower()


def apply_local_reasoning_prompt(messages: list) -> list:
    """Add the brevity directive without stomping a caller's own system prompt.

    An agent like OpenClaude ships a large, carefully built system prompt; replacing it
    would break the harness. So the directive is appended to an existing system message,
    and only becomes a standalone message when the caller sent none.
    """
    if get_local_reasoning_style() != "concise" or not isinstance(messages, list):
        return messages

    for i, msg in enumerate(messages):
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content")
            if isinstance(content, str):
                if LOCAL_CONCISE_DIRECTIVE in content:
                    return messages
                out = list(messages)
                out[i] = {**msg, "content": f"{content}\n\n{LOCAL_CONCISE_DIRECTIVE}"}
                return out
            return messages  # structured/multimodal system content: leave it alone
    return [{"role": "system", "content": LOCAL_CONCISE_DIRECTIVE}, *messages]


def apply_local_request_defaults(body: dict) -> dict:
    """Keep a local reasoning model useful when a caller sends a tiny token budget.

    Mutates nothing: returns the kwargs to merge into the upstream request. Callers who
    set `enable_thinking` explicitly are always respected.
    """
    kwargs: dict = {}
    policy = get_local_thinking_policy()
    if policy == "off":
        return kwargs

    template_kwargs = body.get("chat_template_kwargs")
    if isinstance(template_kwargs, dict) and "enable_thinking" in template_kwargs:
        return kwargs  # caller was explicit; respect it

    budget = body.get("max_tokens") or body.get("max_completion_tokens")
    try:
        budget = int(budget) if budget is not None else None
    except (TypeError, ValueError):
        budget = None

    # No cap at all is the healthy case on a local model: thinking runs, and the context
    # window is the only limit.
    if budget is None or budget >= LOCAL_THINKING_MIN_TOKENS:
        return kwargs

    if policy == "raise":
        # Tokens cost nothing here, so buy the reasoning room instead of losing the answer.
        key = "max_completion_tokens" if body.get("max_completion_tokens") else "max_tokens"
        kwargs[key] = LOCAL_THINKING_MIN_TOKENS
    else:  # disable
        merged = dict(template_kwargs) if isinstance(template_kwargs, dict) else {}
        merged["enable_thinking"] = False
        kwargs["chat_template_kwargs"] = merged
    return kwargs


def build_local_market_stats(raw: list[dict]) -> list[dict]:
    ctx = get_local_ctx()
    tps = get_local_tps()
    result = []
    for m in raw:
        slug = m.get("id") or ""
        if not slug:
            continue
        params = parse_params(slug)
        capability = capability_params(params) * math.log10(ctx + 1)
        result.append({
            "id": f"{LOCAL_MODEL_PREFIX}{slug}",
            "name": slug,
            "params": params,
            "ctx": int(ctx),
            "is_free": True,
            "capability": capability,
            "brain": _is_reasoning(slug),
            "tools": True,
            "open": True,
            "tps": tps,
            # A local box is up exactly when you are using it; there is no meaningful
            # historical uptime to report.
            "uptime": None,
            "provider": LOCAL_PROVIDER_NAME,
            "balanced": 0.0,
            "value": 0.0,
        })
    return result

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

import litellm
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from providers.models import fetch_unified_models, fetch_unified_market_stats
from providers.openrouter import fetch_market_data
from providers.groq import (
    fetch_groq_models,
    build_groq_market_stats,
    is_groq_model,
    strip_groq_prefix,
    get_groq_api_key,
)
from providers.cerebras import (
    fetch_cerebras_models,
    build_cerebras_market_stats,
    is_cerebras_model,
    strip_cerebras_prefix,
    get_cerebras_api_key,
)
from providers.local import (
    fetch_local_models,
    build_local_market_stats,
    is_local_model,
    strip_local_prefix,
    get_local_base_url,
    get_local_api_key,
    apply_local_request_defaults,
    apply_local_reasoning_prompt,
    strip_reasoning,
    strip_reasoning_chunk,
)
from providers.nvidia import (
    fetch_nvidia_models,
    build_nvidia_market_stats,
    is_nvidia_model,
    strip_nvidia_prefix,
    get_nvidia_api_key,
    get_nvidia_base_url,
    NVIDIA_DEFAULT_BASE_URL,
)
from providers.cloudflare import (
    fetch_cloudflare_models,
    build_cloudflare_market_stats,
    is_cloudflare_model,
    strip_cloudflare_prefix,
    get_cloudflare_credentials,
    cloudflare_openai_base,
)
from providers.intelligence import enrich_with_intelligence
from providers.scoring import score_models
from providers.arena import fetch_arena_elo, attach_arena
from providers.ttl_cache import AsyncTTLCache
from providers.capabilities import degradations, warn_on_startup
from providers import db, api_keys, usage, market_store
from providers.api_keys import ApiKey, Scope
from providers.availability_pg import PostgresAvailabilityStore
from providers.availability import (
    PROBE_INTERVAL,
    AvailabilityStore,
    filter_available,
    pick_unverified,
    probe_models,
    record_call_failure,
)

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")
GOOGLE_AISTUDIO_API_KEY = os.getenv("GOOGLE_AISTUDIO_API_KEY", "")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY", "")
# Set on a public deploy: inference then requires the caller's own provider key (BYOK). The market
# still renders from the server's keys, so a first-time visitor sees a populated page.
REQUIRE_USER_KEY = os.getenv("REQUIRE_USER_KEY", "0") not in {"0", "false", "False", ""}
# Mints and revokes consumer keys. Separate from LOCAL_API_KEY on purpose: the key that can create
# credentials must not be the same key you hand to a consumer.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
FALLBACK_MODEL = "groq/llama-3.3-70b-versatile"
OLLAMA_CLOUD_BASE = "https://ollama.com/v1"

# "Pick the best free model for me". `zero-cost-intelligent` is the pre-Gratis name and is kept
# permanently: it is a public API id, and it also sits in users' persisted client state.
POOL_MODEL_ID = "gratis-auto"
POOL_MODEL_IDS = {POOL_MODEL_ID, "zero-cost-intelligent"}

# Model alias map: clients sending OpenAI/Anthropic names get routed to free equivalents.
# Override with MODEL_ALIASES env (JSON object).
DEFAULT_ALIASES: dict[str, str] = {
    "gpt-4": "groq/llama-3.3-70b-versatile",
    "gpt-4-turbo": "groq/llama-3.3-70b-versatile",
    "gpt-4o": "groq/llama-3.3-70b-versatile",
    "gpt-4o-mini": "groq/llama-3.1-8b-instant",
    "gpt-3.5-turbo": "groq/llama-3.1-8b-instant",
    "claude-3-opus": "openrouter/anthropic/claude-3-opus",
    "claude-3-sonnet": "groq/llama-3.3-70b-versatile",
    "claude-3-haiku": "groq/llama-3.1-8b-instant",
    "claude-3-5-sonnet": "groq/llama-3.3-70b-versatile",
    "claude-3-5-haiku": "groq/llama-3.1-8b-instant",
    "claude-sonnet-4": "groq/llama-3.3-70b-versatile",
    "claude-haiku-4": "groq/llama-3.1-8b-instant",
}


def load_aliases() -> dict[str, str]:
    raw = os.getenv("MODEL_ALIASES", "")
    if not raw:
        return DEFAULT_ALIASES
    try:
        import json
        custom = json.loads(raw)
        if isinstance(custom, dict):
            return {**DEFAULT_ALIASES, **custom}
    except Exception as e:
        print(f"MODEL_ALIASES parse error: {e}")
    return DEFAULT_ALIASES


ALIASES = load_aliases()

litellm.drop_params = True
litellm.set_verbose = False


# Postgres when there is one, SQLite otherwise. The SQLite path is what local dev and self-hosting
# use; it is also what production used to use, on an ephemeral disk, which is why an hour of
# learning was destroyed on every deploy.
availability = PostgresAvailabilityStore() if db.is_configured() else AvailabilityStore()


async def _probe_one(model_id: str) -> None:
    """Cheapest call that still proves the model answers. Raises on failure - the caller classifies."""
    litellm_model, extra = resolve_model(model_id)
    await litellm.acompletion(
        model=litellm_model,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
        stream=False,
        **extra,
    )


async def _probe_sweep() -> None:
    """Verify free models nobody has exercised yet, so the market converges without user pain.

    Never blocks a request: the market is served from cache while this runs behind it. Verdicts land
    in the store and take effect on the next read.
    """
    while True:
        try:
            # The SAME compute the request path uses. It used to call _compute_rankings directly, so
            # a cycle the sweep happened to populate wrote no snapshot at all - the cache was warm
            # and the request that would have persisted it never ran the compute.
            models, _ = await _rankings_cache.get_or_compute(_compute_and_persist)
            targets = pick_unverified(models, availability)
            if targets:
                verdicts = await probe_models(targets, _probe_one, availability)
                gone = [m for m, v in verdicts.items() if v == "unavailable"]
                print(f"availability: probed {len(targets)}, quarantined {len(gone)}")
            # Reload the mirror from Postgres. Writes already update it in memory, so this is about
            # OTHER instances: without it a horizontally scaled deploy never learns what its peers
            # quarantined until it restarts.
            if db.is_configured():
                await availability.refresh()
        except Exception as e:  # a broken sweep must never take the API down
            print(f"availability sweep error: {e}")
        await asyncio.sleep(PROBE_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Say it out loud at boot. Production ran for hours with no AA key and looked fine.
    warn_on_startup()

    await db.connect()
    if db.is_configured():
        # The quarantine mirror must be warm BEFORE the first request, or a fresh instance serves a
        # market full of models it already knows are dead.
        await availability.refresh()
        # Answer from the last persisted market instead of refetching ~10 upstreams first. Seeded on
        # a short TTL, so the first request is instant AND a real refresh still lands shortly after.
        snapshot = await market_store.latest_snapshot()
        if snapshot:
            _rankings_cache.prime(snapshot, SNAPSHOT_PRIME_TTL)
            print(f"market: primed from snapshot ({len(snapshot)} models)", flush=True)

    sweep = asyncio.create_task(_probe_sweep())
    try:
        yield
    finally:
        sweep.cancel()
        await db.disconnect()


app = FastAPI(title="Gratis", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== OpenAI-compat error envelope =====

def openai_error(message: str, status: int, err_type: str = "api_error", code: str | None = None) -> JSONResponse:
    body = {
        "error": {
            "message": message,
            "type": err_type,
            "code": code,
            "param": None,
        },
    }
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(HTTPException)
async def http_exc_handler(_: Request, exc: HTTPException):
    err_type = "invalid_request_error" if exc.status_code == 400 else "api_error"
    if exc.status_code == 401:
        err_type = "authentication_error"
    elif exc.status_code == 403:
        err_type = "permission_error"
    elif exc.status_code == 404:
        err_type = "not_found_error"
    elif exc.status_code == 429:
        err_type = "rate_limit_error"
    msg = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return openai_error(msg, exc.status_code, err_type)


@app.exception_handler(RequestValidationError)
async def validation_exc_handler(_: Request, exc: RequestValidationError):
    return openai_error(str(exc.errors()), 400, "invalid_request_error")


# ===== Auth (optional bearer) =====

def require_user_key(user_key: str | None) -> None:
    """Public deploys must not lend their own provider keys for inference - a stranger would drain
    the deployer's quota. Off by default so local dev and self-hosting keep working from .env.
    """
    if REQUIRE_USER_KEY and not user_key:
        raise HTTPException(
            status_code=401,
            detail="This deployment requires your own provider API key. Add one in Settings.",
        )


def _bearer(authorization: str | None) -> str | None:
    if not authorization or not authorization.startswith("Bearer "):
        return None
    return authorization[len("Bearer "):].strip() or None


async def resolve_caller(authorization: str | None = Header(default=None)) -> ApiKey | None:
    """Who is calling. None means "unauthenticated", which is only allowed in open mode.

    Three eras of auth coexist deliberately:
      gr_live_...     a per-consumer key from Postgres: scoped, revocable, metered
      LOCAL_API_KEY   the single shared secret. Still honoured so existing deployments and the UI
                      do not break the moment this ships.
      no key          open mode, when neither is configured (local dev, self-hosting)
    """
    token = _bearer(authorization)

    if token and token.startswith(api_keys.KEY_PREFIX) and db.is_configured():
        key = await api_keys.authenticate(token)
        if key is None:
            # Unknown and revoked must be indistinguishable, or this becomes an oracle for which
            # keys exist.
            raise HTTPException(status_code=401, detail="Invalid API key")
        if not await usage.within_rate_limit(key.id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        return key

    if LOCAL_API_KEY:
        if token != LOCAL_API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return None  # the shared secret has no identity, so it cannot be metered or scoped

    if token:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return None  # open mode


def require_scope(scope: str):
    """A market-only consumer must not be able to spend inference. Keys minted before scopes
    existed, and the shared secret, are unscoped and therefore unrestricted - that is what
    backward compatibility costs."""

    async def dependency(caller: ApiKey | None = Depends(resolve_caller)) -> ApiKey | None:
        if caller is not None and caller.scopes and not caller.has(scope):
            raise HTTPException(status_code=403, detail=f"Key lacks the '{scope}' scope")
        return caller

    return dependency


async def require_admin(authorization: str | None = Header(default=None)) -> None:
    """Minting credentials is not something an ordinary key may do."""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=503, detail="ADMIN_TOKEN is not configured")
    if _bearer(authorization) != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")


# ===== Model resolution =====

def resolve_model(
    model_id: str,
    user_key: str | None = None,
    cf_account_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Returns (litellm_model_string, extra_kwargs).

    `user_key` is a bring-your-own key supplied per request. It is used in place of the server's key
    for whichever provider owns the model, and is never stored or logged.
    """
    if model_id in ALIASES:
        model_id = ALIASES[model_id]

    if model_id.startswith("aistudio/"):
        slug = model_id[len("aistudio/"):]
        return f"gemini/{slug}", {"api_key": user_key or GOOGLE_AISTUDIO_API_KEY}

    if is_groq_model(model_id):
        slug = strip_groq_prefix(model_id)
        return f"groq/{slug}", {"api_key": user_key or get_groq_api_key()}

    if is_cerebras_model(model_id):
        slug = strip_cerebras_prefix(model_id)
        return f"cerebras/{slug}", {"api_key": user_key or get_cerebras_api_key()}

    if is_cloudflare_model(model_id):
        slug = strip_cloudflare_prefix(model_id)
        env_key, env_account = get_cloudflare_credentials()
        return f"openai/{slug}", {
            "api_base": cloudflare_openai_base(cf_account_id or env_account),
            "api_key": user_key or env_key,
        }

    if is_nvidia_model(model_id):
        # litellm has a native nvidia_nim provider, so the base URL is its concern - api_base is
        # passed only when the operator has overridden it.
        slug = strip_nvidia_prefix(model_id)
        nvidia_extra: dict[str, Any] = {"api_key": user_key or get_nvidia_api_key()}
        base = get_nvidia_base_url()
        if base != NVIDIA_DEFAULT_BASE_URL:
            nvidia_extra["api_base"] = base
        return f"nvidia_nim/{slug}", nvidia_extra

    if is_local_model(model_id):
        # Local server speaks the OpenAI API, so it rides the generic openai/ path with
        # an api_base. No BYOK: the endpoint is the user's own machine.
        slug = strip_local_prefix(model_id)
        return f"openai/{slug}", {
            "api_base": get_local_base_url(),
            "api_key": get_local_api_key(),
        }

    if model_id.startswith("ollama/"):
        slug = model_id[len("ollama/"):]
        return f"openai/{slug}", {
            "api_base": OLLAMA_CLOUD_BASE,
            "api_key": user_key or OLLAMA_API_KEY or "ollama",
        }

    if model_id.startswith("openrouter/"):
        return model_id, {"api_key": user_key or OPENROUTER_API_KEY}

    return f"openrouter/{model_id}", {"api_key": user_key or OPENROUTER_API_KEY}


def user_credentials(request: Request) -> tuple[str | None, str | None]:
    """Per-request BYOK credentials. Never persisted, never logged."""
    key = (request.headers.get("X-Provider-Key") or "").strip() or None
    account = (request.headers.get("X-CF-Account-Id") or "").strip() or None
    return key, account


async def pick_best_free_model() -> str:
    """Most capable free model from providers that work without account config.

    Avoids OpenRouter for the default pool: its free models require an account
    data-policy opt-in (openrouter.ai/settings/privacy) and 404 otherwise.
    """
    try:
        groq_free = [m for m in build_groq_market_stats(await fetch_groq_models()) if m["is_free"]]
        if groq_free:
            return max(groq_free, key=lambda m: m["capability"])["id"]
    except Exception:
        pass
    return FALLBACK_MODEL


# ===== Chat completions =====

def _meter(caller: ApiKey | None, endpoint: str, status: int, model_id: str | None, usage_block: dict | None = None) -> None:
    """Fire-and-forget. A metering write must never fail the request it is describing."""
    if caller is None:
        return  # the shared secret has no identity, so there is nothing to meter it against
    tokens = usage_block or {}
    usage.record_background(
        key_id=caller.id,
        endpoint=endpoint,
        status=status,
        model_id=model_id,
        prompt_tokens=tokens.get("prompt_tokens"),
        completion_tokens=tokens.get("completion_tokens"),
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, caller: ApiKey | None = Depends(require_scope(Scope.INFERENCE))) -> Any:
    body = await request.json()
    model_id: str = body.get("model", "")
    is_pool = not model_id or model_id in POOL_MODEL_IDS

    user_key, cf_account_id = user_credentials(request)
    require_user_key(user_key)

    if is_pool:
        model_id = await pick_best_free_model()
    litellm_model, extra = resolve_model(model_id, user_key, cf_account_id)

    messages = body.get("messages", [])
    stream: bool = body.get("stream", False)

    # A local reasoning model returns empty content when a small token budget is spent
    # inside its <think> block; fold in the guard before passthrough is assembled, and
    # keep its chain-of-thought brief (measured 56% fewer completion tokens, 2.3x faster,
    # same answer) rather than turning reasoning off.
    if is_local_model(model_id):
        body = {**body, **apply_local_request_defaults(body)}
        messages = apply_local_reasoning_prompt(messages)

    passthrough = {
        k: v for k, v in body.items()
        if k not in {"model", "messages", "stream"}
    }

    try:
        if stream:
            response = await litellm.acompletion(
                model=litellm_model,
                messages=messages,
                stream=True,
                **extra,
                **passthrough,
            )

            hide_reasoning = is_local_model(model_id)

            async def event_stream():
                # A streamed request used to be metered NOWHERE: only the non-stream path recorded a
                # usage row. Since the rate limiter counts those same rows, and the UI always
                # streams, streaming was both unbilled and unlimited - a free hole straight through
                # the quota. The verdict is recorded in `finally`, so a client that disconnects
                # mid-reply is still accounted for.
                usage_block: dict | None = None
                status_code = 200
                try:
                    async for chunk in response:
                        chunk_usage = getattr(chunk, "usage", None)
                        if chunk_usage is not None:
                            usage_block = (
                                chunk_usage if isinstance(chunk_usage, dict) else chunk_usage.model_dump()
                            )
                        if hide_reasoning:
                            # Re-serialize only when we actually have to edit the chunk.
                            import json as _json
                            payload = _json.dumps(strip_reasoning_chunk(chunk.model_dump()))
                        else:
                            payload = chunk.model_dump_json()
                        yield f"data: {payload}\n\n"
                except Exception as e:
                    status_code = getattr(e, "status_code", 500) or 500
                    # Same rule as the non-stream path: only a failure on OUR key says anything about
                    # the model, so a user's revoked or out-of-quota key cannot quarantine it.
                    if user_key is None:
                        record_call_failure(availability, model_id, status_code, str(e))
                    err = {"error": {"message": str(e), "type": "api_error", "code": getattr(e, "status_code", None)}}
                    import json as _json
                    yield f"data: {_json.dumps(err)}\n\n"
                finally:
                    _meter(caller, "/v1/chat/completions", status_code, model_id, usage_block)
                    yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache, no-transform",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        response = await litellm.acompletion(
            model=litellm_model,
            messages=messages,
            stream=False,
            **extra,
            **passthrough,
        )
        payload = response.model_dump()
        _meter(caller, "/v1/chat/completions", 200, model_id, payload.get("usage"))
        if is_local_model(model_id):
            payload = strip_reasoning(payload)
        return JSONResponse(content=payload)

    except HTTPException:
        raise
    except Exception as e:
        status = getattr(e, "status_code", 500) or 500
        # Only a failure on OUR key says anything about the model. A user's key can be revoked,
        # out of quota or plain wrong - quarantining on that would let one bad key delete a working
        # model for everyone, and would persist their key's error text to disk.
        if user_key is None:
            record_call_failure(availability, model_id, status, str(e))
        _meter(caller, "/v1/chat/completions", status, model_id)
        raise HTTPException(status_code=status, detail=str(e))


# ===== Embeddings =====

# Scoped, not merely authenticated. This used to sit behind require_auth, which checks that a key is
# VALID and never that it is ALLOWED - so a market-only key could spend inference here, through the
# one inference endpoint that was not scope-checked.
@app.post("/v1/embeddings")
async def embeddings(request: Request, caller: ApiKey | None = Depends(require_scope(Scope.INFERENCE))) -> Any:
    body = await request.json()
    model_id: str = body.get("model", "")
    if not model_id:
        raise HTTPException(status_code=400, detail="model field required")
    inputs = body.get("input")
    if inputs is None:
        raise HTTPException(status_code=400, detail="input field required")

    user_key, cf_account_id = user_credentials(request)
    require_user_key(user_key)

    litellm_model, extra = resolve_model(model_id, user_key, cf_account_id)
    passthrough = {k: v for k, v in body.items() if k not in {"model", "input"}}

    try:
        response = await litellm.aembedding(
            model=litellm_model,
            input=inputs,
            **extra,
            **passthrough,
        )
        payload = response.model_dump()
        _meter(caller, "/v1/embeddings", 200, model_id, payload.get("usage"))
        return JSONResponse(content=payload)
    except HTTPException:
        raise
    except Exception as e:
        status = getattr(e, "status_code", 500) or 500
        if user_key is None:
            record_call_failure(availability, model_id, status, str(e))
        _meter(caller, "/v1/embeddings", status, model_id)
        raise HTTPException(status_code=status, detail=str(e))


# ===== Models =====

# The catalogue is 7 upstream calls, and it was uncached while /v1/rankings - which fetches strictly
# more - was cached for 30 minutes. Same 1800s default; both describe the same underlying data.
MODELS_CACHE_TTL = float(os.getenv("MODELS_CACHE_TTL", "1800"))
_models_cache: AsyncTTLCache[dict] = AsyncTTLCache(ttl=MODELS_CACHE_TTL)


@app.get("/v1/models")
async def models(caller: ApiKey | None = Depends(require_scope(Scope.MARKET))):
    listing, _ = await _models_cache.get_or_compute(fetch_unified_models)
    _meter(caller, "/v1/models", 200, None)
    # A NEW dict every time. Assigning the filtered list back into `listing` would write through to
    # the cached object, so one quarantined model would stay missing for the rest of the TTL even
    # after its quarantine expired.
    return JSONResponse(
        content={**listing, "data": filter_available(listing.get("data", []), availability)}
    )


# MUST be declared before /v1/models/{model_id:path}: that route is greedy (`:path` matches slashes)
# and would otherwise swallow "<id>/history" and return a 404 for a model literally called that.
@app.get("/v1/models/{model_id:path}/history")
async def model_history(model_id: str, days: int = 30, caller: ApiKey | None = Depends(require_scope(Scope.MARKET))):
    _meter(caller, "/v1/models/history", 200, model_id)
    """Price and score over time. The market could never answer 'did this get cheaper?' because it
    only ever knew about right now."""
    if not db.is_configured():
        raise HTTPException(status_code=503, detail="A database is required for history")
    return {"model_id": model_id, "days": days, "data": await market_store.history(model_id, days)}


@app.get("/v1/models/{model_id:path}")
async def model_get(model_id: str, _: ApiKey | None = Depends(require_scope(Scope.MARKET))):
    listing, _hit = await _models_cache.get_or_compute(fetch_unified_models)
    found = next((m for m in listing.get("data", []) if m.get("id") == model_id), None)
    if not found:
        raise HTTPException(status_code=404, detail=f"Model '{model_id}' not found")
    return JSONResponse(content=found)


# ===== Service: keys, usage, history =====

@app.post("/admin/keys", dependencies=[Depends(require_admin)])
async def create_api_key(request: Request):
    """Mint a consumer key. The raw value is returned ONCE and is not recoverable - only a SHA-256
    digest is stored, so a database dump hands an attacker nothing usable."""
    if not db.is_configured():
        raise HTTPException(status_code=503, detail="A database is required to issue API keys")

    body = await request.json()
    tenant = (body.get("tenant") or "").strip()
    name = (body.get("name") or "").strip()
    if not tenant or not name:
        raise HTTPException(status_code=400, detail="tenant and name are required")

    scopes = tuple(body.get("scopes") or api_keys.DEFAULT_SCOPES)
    raw, key = await api_keys.create(tenant, name, scopes)

    return JSONResponse(
        status_code=201,
        content={
            "id": key.id,
            "tenant": key.tenant,
            "name": key.name,
            "scopes": list(key.scopes),
            "key": raw,
            "warning": "This is the only time the key is shown. Store it now.",
        },
    )


@app.get("/admin/keys", dependencies=[Depends(require_admin)])
async def list_api_keys(tenant: str | None = None):
    keys = await api_keys.list_keys(tenant)
    return {
        "data": [
            {
                "id": k.id,
                "tenant": k.tenant,
                "name": k.name,
                "key_prefix": k.key_prefix,
                "scopes": list(k.scopes),
                "created_at": k.created_at.isoformat(),
                "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                "revoked_at": k.revoked_at.isoformat() if k.revoked_at else None,
            }
            for k in keys
        ]
    }


@app.delete("/admin/keys/{key_id}", dependencies=[Depends(require_admin)])
async def revoke_api_key(key_id: str):
    if not await api_keys.revoke(key_id):
        raise HTTPException(status_code=404, detail="Key not found or already revoked")
    return {"revoked": key_id}


@app.get("/v1/usage")
async def get_usage(hours: int = 24, caller: ApiKey | None = Depends(resolve_caller)):
    """What THIS key has spent. A shared secret has no identity, so it has no usage to report."""
    if caller is None:
        raise HTTPException(status_code=401, detail="A per-consumer API key is required")
    s = await usage.summary(caller.id, hours)
    return {
        "tenant": caller.tenant,
        "window_hours": hours,
        "requests": s.requests,
        "prompt_tokens": s.prompt_tokens,
        "completion_tokens": s.completion_tokens,
        "errors": s.errors,
    }



# ===== Rankings (extension) =====

async def _compute_rankings() -> list[dict]:
    (
        openrouter_stats,
        unified_extra,
        groq_raw,
        cerebras_raw,
        cloudflare_raw,
        local_raw,
        nvidia_raw,
    ) = await asyncio.gather(
        fetch_market_data(),
        fetch_unified_market_stats(),
        fetch_groq_models(),
        fetch_cerebras_models(),
        fetch_cloudflare_models(),
        fetch_local_models(),
        fetch_nvidia_models(),
    )

    ollama_stats, aistudio_stats = unified_extra
    groq_stats = build_groq_market_stats(groq_raw)
    cerebras_stats = build_cerebras_market_stats(cerebras_raw)
    cloudflare_stats = build_cloudflare_market_stats(cloudflare_raw)
    local_stats = build_local_market_stats(local_raw)
    nvidia_stats = build_nvidia_market_stats(nvidia_raw)

    extras = (
        ollama_stats + aistudio_stats + groq_stats + cerebras_stats
        + cloudflare_stats + local_stats + nvidia_stats
    )
    all_stats = openrouter_stats + extras
    if not all_stats:
        return []

    max_cap = max((m["capability"] for m in all_stats), default=1)

    def rescore(m: dict) -> dict:
        return {**m, "balanced": (m["capability"] / max_cap) * 60 + (20 if m["brain"] else 0) + 20}

    rescored_extra = [rescore(m) for m in extras]
    combined = sorted(openrouter_stats + rescored_extra, key=lambda m: m["balanced"], reverse=True)
    # Attach real AA intelligence FIRST, then the composite scorer — so intelligence actually drives the
    # ranking (previously it was only decorated on AFTER the crude balanced sort). score_models returns
    # the list sorted by overall, with per-dimension + task-fit scores + archetype for downstream sorts.
    combined = await enrich_with_intelligence(combined)
    combined = score_models(combined)
    # Overlay the human-preference axis (LMArena Elo) + the benchmark-vs-humans divergence flag — the
    # honest triangulation the benchmark composite alone can't give. Fail-open (no Elo → no overlay).
    combined = attach_arena(combined, await fetch_arena_elo())
    return combined


# The full ranking is ~10 external calls (5 provider markets + AA intelligence + LMArena Elo) plus
# scoring. Cache the assembled result so repeat loads within the TTL are instant and don't re-hammer
# upstreams. Empty results (all providers down) are not cached, so recovery isn't pinned out. TTL is
# env-driven; default 1800s (30 min).
RANKINGS_CACHE_TTL = float(os.getenv("RANKINGS_CACHE_TTL", "1800"))
_rankings_cache: AsyncTTLCache[list[dict]] = AsyncTTLCache(ttl=RANKINGS_CACHE_TTL)

# How long a snapshot seeded at startup is trusted. Deliberately much shorter than the full TTL: it
# exists to make the FIRST request instant, not to serve half-hour-old data for half an hour.
SNAPSHOT_PRIME_TTL = float(os.getenv("SNAPSHOT_PRIME_TTL", "120"))


async def _compute_and_persist() -> list[dict]:
    """Compute the market, then persist it: the snapshot is what lets the NEXT cold instance answer
    immediately instead of refetching ~10 upstreams first."""
    models = await _compute_rankings()
    await market_store.save_snapshot(models)
    return models


@app.get("/v1/rankings")
async def rankings(caller: ApiKey | None = Depends(require_scope(Scope.MARKET))):
    # Metered, not just authorised. The rate limiter counts the SAME rows the usage report bills
    # from, so an endpoint that is not metered is also not rate-limited - it would be a free hole
    # straight through the quota. Metered AFTER the work: recording 200 before doing anything books
    # a success for a request that may be about to fail.
    data, hit = await _rankings_cache.get_or_compute(_compute_and_persist)
    _meter(caller, "/v1/rankings", 200, None)
    # Filtered on read, not inside the cached compute: a quarantine then takes effect on the next
    # request instead of waiting out the 30-min TTL, and the prober still sees the full catalogue.
    return JSONResponse(
        content=filter_available(data, availability),
        headers={"X-Cache": "HIT" if hit else "MISS"},
    )


# ===== Health + Root =====

@app.get("/health")
async def health():
    """`degraded` is non-empty when a missing key has quietly broken a feature. A health check that
    reports "ok" while the market is ranking on a size heuristic is not a health check."""
    degraded = degradations()
    return {
        "status": "ok" if not degraded else "degraded",
        "degraded": degraded,
    }


@app.get("/")
async def root():
    return {
        "name": "Gratis",
        "version": "1.0.0",
        "compat": "openai",
        "auth": "required" if LOCAL_API_KEY else "none",
        "endpoints": [
            "/v1/chat/completions",
            "/v1/embeddings",
            "/v1/models",
            "/v1/models/{id}",
            "/v1/rankings",
            "/health",
        ],
        "aliases_count": len(ALIASES),
    }

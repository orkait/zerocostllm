"""NVIDIA NIM: an OpenAI-compatible catalogue that is public, mixed-modality, and author-prefixed.

The mixed catalogue is the interesting part. /v1/models lists embedders, rerankers, OCR and speech
models next to the chat ones, and none of those can serve a chat completion - so shipping them whole
would fill the market with rows whose only possible outcome is a failed call.
"""

import asyncio

from providers.nvidia import (
    NVIDIA_DEFAULT_CTX,
    NVIDIA_MODEL_PREFIX,
    NVIDIA_PROVIDER_NAME,
    build_nvidia_market_stats,
    fetch_nvidia_models,
    get_nvidia_base_url,
    is_chat_model,
    is_nvidia_model,
    strip_nvidia_prefix,
)


# ---- id handling -------------------------------------------------------------------------------

def test_prefix_round_trip():
    assert is_nvidia_model("nim/meta/llama-3.1-8b-instruct")
    assert strip_nvidia_prefix("nim/meta/llama-3.1-8b-instruct") == "meta/llama-3.1-8b-instruct"


def test_nvidias_own_models_keep_their_author_segment():
    """`nim/nvidia/nemotron-...` - route first, author second. Stripping removes only the ROUTE."""
    assert strip_nvidia_prefix("nim/nvidia/nemotron-4-340b-instruct") == "nvidia/nemotron-4-340b-instruct"


def test_an_openrouter_model_authored_by_nvidia_is_not_claimed_as_a_nim_model():
    """The reason the prefix is `nim/` and not `nvidia/`.

    NVIDIA is an author on OpenRouter, which really serves `nvidia/nemotron-3-super-120b-a12b`. With
    a `nvidia/` route prefix those rows were indistinguishable from NIM's: the live catalogue came
    back with 90 "NVIDIA" models when NIM only had 80, because ten OpenRouter rows had been absorbed
    into the wrong provider.
    """
    for openrouter_id in [
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "nvidia/nemotron-3-ultra-550b-a55b",
    ]:
        assert not is_nvidia_model(openrouter_id), openrouter_id


def test_a_non_nvidia_id_is_left_alone():
    assert not is_nvidia_model("groq/llama-3.3-70b")
    assert strip_nvidia_prefix("groq/llama-3.3-70b") == "groq/llama-3.3-70b"


def test_a_non_string_id_does_not_explode():
    assert is_nvidia_model(None) is False
    assert is_nvidia_model(123) is False


# ---- modality filtering -------------------------------------------------------------------------

def test_chat_models_are_kept():
    for model_id in [
        "meta/llama-3.1-8b-instruct",
        "deepseek-ai/deepseek-v4-pro",
        "mistralai/mistral-nemotron",
        "google/gemma-4-31b-it",
        "openai/gpt-oss-120b",
    ]:
        assert is_chat_model(model_id), model_id


def test_models_that_cannot_serve_a_chat_completion_are_dropped():
    """Real ids from the live catalogue. Each of these would be a permanent failure on first call."""
    for model_id in [
        "baai/bge-m3",                                  # embedding
        "nvidia/embed-qa-4",                            # embedding
        "nvidia/llama-3.2-nv-embedqa-1b-v1",            # embedding
        "nvidia/nemoretriever-parse",                   # retrieval/parse
        "nvidia/nemotron-3-embed-1b",                   # embedding
        "google/deplot",                                # chart-to-text
        "nvidia/ai-synthetic-video-detector",           # video
        "google/diffusiongemma-26b-a4b-it",             # diffusion
    ]:
        assert not is_chat_model(model_id), model_id


def test_the_filter_is_applied_to_the_fetched_catalogue(monkeypatch):
    monkeypatch.setenv("NVIDIA_NIM_API_KEY", "nvapi-test")

    class _Response:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {"data": [{"id": "meta/llama-3.1-8b-instruct"}, {"id": "baai/bge-m3"}, {"id": ""}]}

    class _Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, *a, **kw):
            return _Response()

    monkeypatch.setattr("providers.nvidia.httpx.AsyncClient", lambda *a, **kw: _Client())
    assert asyncio.run(fetch_nvidia_models()) == [{"id": "meta/llama-3.1-8b-instruct"}]


# ---- the no-key contract -------------------------------------------------------------------------

def test_no_key_means_no_models(monkeypatch):
    """NIM's catalogue is public, unlike every other provider's - so this is the one place the
    "no key, no rows" contract could have been broken by accident. A deployment that cannot CALL
    these models must not advertise them."""
    monkeypatch.delenv("NVIDIA_NIM_API_KEY", raising=False)
    assert asyncio.run(fetch_nvidia_models()) == []


def test_the_base_url_is_overridable(monkeypatch):
    """Self-hosted NIM containers are the whole point of the product; they are not on
    integrate.api.nvidia.com."""
    monkeypatch.setenv("NVIDIA_NIM_API_BASE", "http://nim.internal:8000/v1/")
    assert get_nvidia_base_url() == "http://nim.internal:8000/v1"


def test_the_default_base_is_nvidias_hosted_endpoint(monkeypatch):
    monkeypatch.delenv("NVIDIA_NIM_API_BASE", raising=False)
    assert get_nvidia_base_url() == "https://integrate.api.nvidia.com/v1"


# ---- market rows ---------------------------------------------------------------------------------

def test_market_row_shape():
    [row] = build_nvidia_market_stats([{"id": "meta/llama-3.1-405b-instruct"}])

    assert row["id"] == f"{NVIDIA_MODEL_PREFIX}meta/llama-3.1-405b-instruct"
    assert row["provider"] == NVIDIA_PROVIDER_NAME
    assert row["is_free"] is True
    assert row["ctx"] == NVIDIA_DEFAULT_CTX
    assert row["params"] == 405.0
    # Throughput is not published, so it stays absent rather than being invented.
    assert row["tps"] is None


def test_the_author_is_read_off_the_id():
    """NIM ids name who BUILT the model. Dropping that would repeat the bug that labelled
    claude-fable-5 "Google" - showing the route where the author belongs."""
    [row] = build_nvidia_market_stats([{"id": "meta/llama-3.1-8b-instruct"}])
    assert row["author"] == "Meta"
    assert row["provider"] == NVIDIA_PROVIDER_NAME


def test_a_size_that_is_not_stated_is_none_not_one():
    [row] = build_nvidia_market_stats([{"id": "nvidia/nemotron-mini-instruct"}])
    assert row["params"] is None


def test_reasoning_models_are_flagged():
    [row] = build_nvidia_market_stats([{"id": "deepseek-ai/deepseek-r1"}])
    assert row["brain"] is True


def test_a_plain_instruct_model_is_not_flagged_as_reasoning():
    [row] = build_nvidia_market_stats([{"id": "meta/llama-3.1-8b-instruct"}])
    assert row["brain"] is False


def test_rows_without_a_slug_are_skipped():
    assert build_nvidia_market_stats([{"id": ""}, {}]) == []

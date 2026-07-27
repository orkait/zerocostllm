"""The local provider: a model on your own GPU, discovered by being reachable.

Every other provider shipped with a test and this one did not, which is how it reached the market
carrying behaviour nothing verified - the /v1/models shape sniffing, the reasoning-prompt injection
that must not stomp a caller's system prompt, and the thinking-budget policy.
"""

import asyncio

from providers.local import (
    LOCAL_CONCISE_DIRECTIVE,
    LOCAL_DEFAULT_CTX,
    LOCAL_THINKING_MIN_TOKENS,
    apply_local_reasoning_prompt,
    apply_local_request_defaults,
    build_local_market_stats,
    fetch_local_models,
    get_local_ctx,
    get_local_tps,
    is_local_model,
    strip_local_prefix,
    strip_reasoning,
    strip_reasoning_chunk,
    _normalize_models,
)


# ---- id handling -------------------------------------------------------------------------------

def test_prefix_round_trip():
    assert is_local_model("local/bonsai-27b")
    assert strip_local_prefix("local/bonsai-27b") == "bonsai-27b"


def test_a_non_local_id_is_left_alone():
    assert not is_local_model("groq/llama-3.3-70b")
    assert strip_local_prefix("groq/llama-3.3-70b") == "groq/llama-3.3-70b"


def test_a_non_string_id_does_not_explode():
    """Ids arrive from request bodies, so they are whatever the caller sent."""
    assert is_local_model(None) is False
    assert is_local_model(123) is False


# ---- /v1/models shape sniffing -----------------------------------------------------------------

def test_openai_shape_is_normalized():
    """vLLM and LM Studio answer with the OpenAI form."""
    rows = _normalize_models({"data": [{"id": "qwen3-8b", "created": 1}]})
    assert rows == [{"id": "qwen3-8b", "created": 1}]


def test_llama_server_ollama_shape_is_normalized():
    """llama-server answers with an Ollama-flavoured body. Accepting only the OpenAI form would
    silently empty the market for the DEFAULT local backend."""
    rows = _normalize_models({"models": [{"name": "bonsai-27b", "model": "bonsai-27b"}]})
    assert rows == [{"id": "bonsai-27b", "created": None}]


def test_rows_without_any_usable_name_are_dropped():
    rows = _normalize_models({"data": [{"id": "ok"}, {"nothing": "useful"}, "not-a-dict"]})
    assert rows == [{"id": "ok", "created": None}]


def test_an_unrecognised_body_yields_nothing_rather_than_raising():
    assert _normalize_models({}) == []


def test_a_server_that_is_not_running_is_not_an_error(monkeypatch):
    """Nothing listening is the NORMAL case: the provider disappears, exactly like a provider with
    no key. It must not raise and take the whole market aggregation down with it."""
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:9/v1")  # discard port
    assert asyncio.run(fetch_local_models()) == []


# ---- reported context / throughput -------------------------------------------------------------

def test_ctx_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_CTX", "131072")
    assert get_local_ctx() == 131_072


def test_a_junk_ctx_falls_back_rather_than_crashing(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_CTX", "sixty-five-thousand")
    assert get_local_ctx() == LOCAL_DEFAULT_CTX


def test_tps_is_none_when_unmeasured(monkeypatch):
    monkeypatch.delenv("LOCAL_LLM_TPS", raising=False)
    assert get_local_tps() is None


def test_a_junk_tps_is_none_not_zero(monkeypatch):
    """Zero would render as a measurement of "no throughput". Absent is the honest answer."""
    monkeypatch.setenv("LOCAL_LLM_TPS", "fast")
    assert get_local_tps() is None


# ---- market rows ---------------------------------------------------------------------------------

def test_market_row_is_free_and_carries_the_configured_ctx(monkeypatch):
    monkeypatch.setenv("LOCAL_LLM_CTX", "65536")
    monkeypatch.setenv("LOCAL_LLM_TPS", "29.4")
    [row] = build_local_market_stats([{"id": "Ternary-Bonsai-27B"}])

    assert row["id"] == "local/Ternary-Bonsai-27B"
    assert row["provider"] == "Local"
    assert row["is_free"] is True
    assert row["ctx"] == 65_536
    assert row["tps"] == 29.4
    # 27B is stated in the slug, so the size is known rather than invented.
    assert row["params"] == 27.0


def test_a_size_that_is_not_stated_is_none_not_one(monkeypatch):
    """The repo-wide truth rule: an unparseable size is None. It used to default to 1.0, which
    rendered Claude Opus as a 1B model."""
    monkeypatch.delenv("LOCAL_LLM_TPS", raising=False)
    [row] = build_local_market_stats([{"id": "my-custom-merge"}])
    assert row["params"] is None


def test_rows_without_a_slug_are_skipped():
    assert build_local_market_stats([{"id": ""}, {}]) == []


def test_a_reasoning_slug_is_flagged():
    [row] = build_local_market_stats([{"id": "deepseek-r1-distill"}])
    assert row["brain"] is True


# ---- reasoning prompt ---------------------------------------------------------------------------

def test_the_directive_is_appended_to_an_existing_system_prompt(monkeypatch):
    """An agent ships a large, carefully built system prompt. REPLACING it would break the harness,
    so the directive is appended."""
    monkeypatch.delenv("LOCAL_REASONING_STYLE", raising=False)
    messages = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "hi"}]
    out = apply_local_reasoning_prompt(messages)

    assert out[0]["content"].startswith("You are a coding agent.")
    assert LOCAL_CONCISE_DIRECTIVE in out[0]["content"]
    assert out[1] == {"role": "user", "content": "hi"}
    # The caller's list is not mutated.
    assert messages[0]["content"] == "You are a coding agent."


def test_a_standalone_directive_is_added_when_there_is_no_system_prompt(monkeypatch):
    monkeypatch.delenv("LOCAL_REASONING_STYLE", raising=False)
    out = apply_local_reasoning_prompt([{"role": "user", "content": "hi"}])
    assert out[0] == {"role": "system", "content": LOCAL_CONCISE_DIRECTIVE}


def test_the_directive_is_not_applied_twice(monkeypatch):
    """Retries and multi-turn conversations re-send the same messages."""
    monkeypatch.delenv("LOCAL_REASONING_STYLE", raising=False)
    once = apply_local_reasoning_prompt([{"role": "user", "content": "hi"}])
    twice = apply_local_reasoning_prompt(once)
    assert twice[0]["content"].count(LOCAL_CONCISE_DIRECTIVE) == 1


def test_structured_system_content_is_left_alone(monkeypatch):
    """Multimodal/structured content is not a string; appending to it would corrupt the request."""
    monkeypatch.delenv("LOCAL_REASONING_STYLE", raising=False)
    messages = [{"role": "system", "content": [{"type": "text", "text": "x"}]}]
    assert apply_local_reasoning_prompt(messages) == messages


def test_style_off_leaves_prompts_untouched(monkeypatch):
    monkeypatch.setenv("LOCAL_REASONING_STYLE", "off")
    messages = [{"role": "user", "content": "hi"}]
    assert apply_local_reasoning_prompt(messages) == messages


# ---- thinking budget ----------------------------------------------------------------------------

def test_default_policy_changes_nothing(monkeypatch):
    """Output tokens are free on a local model, so rewriting someone else's max_tokens is a surprise
    that belongs behind an explicit opt-in."""
    monkeypatch.delenv("LOCAL_THINKING_POLICY", raising=False)
    assert apply_local_request_defaults({"max_tokens": 24}) == {}


def test_raise_policy_buys_room_for_the_think_block(monkeypatch):
    monkeypatch.setenv("LOCAL_THINKING_POLICY", "raise")
    assert apply_local_request_defaults({"max_tokens": 24}) == {"max_tokens": LOCAL_THINKING_MIN_TOKENS}


def test_raise_policy_respects_which_budget_field_the_caller_used(monkeypatch):
    monkeypatch.setenv("LOCAL_THINKING_POLICY", "raise")
    out = apply_local_request_defaults({"max_completion_tokens": 24})
    assert out == {"max_completion_tokens": LOCAL_THINKING_MIN_TOKENS}


def test_a_generous_budget_is_left_alone(monkeypatch):
    monkeypatch.setenv("LOCAL_THINKING_POLICY", "raise")
    assert apply_local_request_defaults({"max_tokens": 4096}) == {}


def test_no_budget_at_all_is_the_healthy_case(monkeypatch):
    monkeypatch.setenv("LOCAL_THINKING_POLICY", "raise")
    assert apply_local_request_defaults({}) == {}


def test_disable_policy_turns_thinking_off_instead_of_raising_the_cap(monkeypatch):
    monkeypatch.setenv("LOCAL_THINKING_POLICY", "disable")
    out = apply_local_request_defaults({"max_tokens": 24})
    assert out == {"chat_template_kwargs": {"enable_thinking": False}}


def test_an_explicit_caller_choice_is_always_respected(monkeypatch):
    monkeypatch.setenv("LOCAL_THINKING_POLICY", "raise")
    body = {"max_tokens": 24, "chat_template_kwargs": {"enable_thinking": True}}
    assert apply_local_request_defaults(body) == {}


def test_an_unknown_policy_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("LOCAL_THINKING_POLICY", "yolo")
    assert apply_local_request_defaults({"max_tokens": 24}) == {}


def test_a_junk_budget_is_treated_as_absent(monkeypatch):
    monkeypatch.setenv("LOCAL_THINKING_POLICY", "raise")
    assert apply_local_request_defaults({"max_tokens": "lots"}) == {}


# ---- reasoning stripping ------------------------------------------------------------------------

def test_reasoning_is_dropped_from_a_completion(monkeypatch):
    monkeypatch.delenv("LOCAL_HIDE_REASONING", raising=False)
    payload = {"choices": [{"message": {"content": "42", "reasoning_content": "thinking..."}}]}
    out = strip_reasoning(payload)
    assert out["choices"][0]["message"] == {"content": "42"}


def test_reasoning_is_dropped_from_a_streamed_chunk(monkeypatch):
    """The fields live on `delta`, not `message` - stripping only the message shape would leak the
    entire chain of thought to every streaming client."""
    monkeypatch.delenv("LOCAL_HIDE_REASONING", raising=False)
    chunk = {"choices": [{"delta": {"content": "4", "reasoning": "thinking..."}}]}
    assert strip_reasoning_chunk(chunk)["choices"][0]["delta"] == {"content": "4"}


def test_reasoning_passes_through_when_the_operator_asks_for_it(monkeypatch):
    monkeypatch.setenv("LOCAL_HIDE_REASONING", "0")
    payload = {"choices": [{"message": {"content": "42", "reasoning_content": "thinking..."}}]}
    assert strip_reasoning(payload)["choices"][0]["message"]["reasoning_content"] == "thinking..."


def test_stripping_survives_a_payload_with_no_choices(monkeypatch):
    monkeypatch.delenv("LOCAL_HIDE_REASONING", raising=False)
    assert strip_reasoning({}) == {}
    assert strip_reasoning_chunk({"choices": None}) == {"choices": None}

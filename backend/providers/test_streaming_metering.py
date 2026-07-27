"""A streamed completion must be metered, exactly like a non-streamed one.

Metering was wired into the non-stream branch only. The rate limiter counts the SAME usage_events
rows the usage report bills from, so an unmetered endpoint is also an unlimited one - and since the
UI always streams, the entire product's real traffic went through the one path that recorded
nothing. It was neither billed nor rate-limited.
"""

import json
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("GROQ_API_KEY", "server-groq")

import main  # noqa: E402


class _Chunk:
    """Enough of a litellm chunk for the stream handler: it serializes, and it may carry usage."""

    def __init__(self, text: str, usage: dict | None = None):
        self._text = text
        self.usage = usage

    def model_dump_json(self) -> str:
        return json.dumps(self._as_dict())

    def model_dump(self) -> dict:
        return self._as_dict()

    def _as_dict(self) -> dict:
        body: dict = {"choices": [{"delta": {"content": self._text}}]}
        if self.usage is not None:
            body["usage"] = self.usage
        return body


@pytest.fixture
def metered(monkeypatch):
    """Records every _meter() call instead of writing to a database."""
    calls: list[tuple] = []
    monkeypatch.setattr(main, "_meter", lambda *a, **kw: calls.append((a, kw)))
    monkeypatch.setattr(main, "REQUIRE_USER_KEY", False)
    return calls


def _stream_of(*chunks, fail_with: Exception | None = None):
    async def fake_acompletion(**kwargs):
        async def gen():
            for chunk in chunks:
                yield chunk
            if fail_with is not None:
                raise fail_with

        return gen()

    return fake_acompletion


def _post_stream(body: dict) -> str:
    with TestClient(main.app) as client:
        return client.post("/v1/chat/completions", json=body).text


REQUEST = {"model": "groq/llama-3.3-70b-versatile", "messages": [{"role": "user", "content": "hi"}], "stream": True}


def test_a_streamed_completion_is_metered(metered, monkeypatch):
    monkeypatch.setattr(main.litellm, "acompletion", _stream_of(_Chunk("hi")))

    body = _post_stream(REQUEST)

    assert "data: [DONE]" in body
    endpoints = [args[1] for args, _ in metered]
    assert "/v1/chat/completions" in endpoints, "a streamed request recorded no usage at all"


def test_the_streamed_token_counts_are_recorded(metered, monkeypatch):
    """Without the usage block the row exists but bills nothing, so the report would under-count
    every streamed request to zero tokens."""
    usage = {"prompt_tokens": 11, "completion_tokens": 22}
    monkeypatch.setattr(main.litellm, "acompletion", _stream_of(_Chunk("hi"), _Chunk("", usage=usage)))

    _post_stream(REQUEST)

    [(args, _)] = [c for c in metered if c[0][1] == "/v1/chat/completions"]
    assert args[2] == 200
    assert args[4] == usage


def test_a_stream_that_fails_midway_is_metered_with_the_failure_status(metered, monkeypatch):
    error = RuntimeError("upstream exploded")
    error.status_code = 503
    monkeypatch.setattr(main.litellm, "acompletion", _stream_of(_Chunk("par"), fail_with=error))

    body = _post_stream(REQUEST)

    assert "upstream exploded" in body
    [(args, _)] = [c for c in metered if c[0][1] == "/v1/chat/completions"]
    assert args[2] == 503, "a failed stream must not be booked as a success"


def test_a_non_streamed_completion_is_still_metered(metered, monkeypatch):
    """The path that already worked, pinned so the refactor did not trade one hole for another."""

    class _Response:
        def model_dump(self):
            return {"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 3}}

    async def fake_acompletion(**kwargs):
        return _Response()

    monkeypatch.setattr(main.litellm, "acompletion", fake_acompletion)

    with TestClient(main.app) as client:
        client.post("/v1/chat/completions", json={**REQUEST, "stream": False})

    [(args, _)] = [c for c in metered if c[0][1] == "/v1/chat/completions"]
    assert args[2] == 200
    assert args[4] == {"prompt_tokens": 3}

"""The prefix contract is enforced across BOTH codebases, not just described in a comment.

The frontend re-declares every model-id prefix in src/config/providers.ts because it has to resolve
the same routing client-side (which key to attach, which provider filter a row belongs to). Two
hand-maintained copies of one contract drift, and the last drift shipped: `local/` existed only on
the backend, so local models were attached to the OpenRouter provider - wrong filter, wrong label,
and the user's OpenRouter key sent along with a request to their own machine.
"""

import pathlib
import re

import pytest

from main import resolve_model
from providers.registry import FALLBACK_PREFIX, ROUTED_PREFIXES

TS_REGISTRY = pathlib.Path(__file__).resolve().parents[2] / "src" / "config" / "providers.ts"

# `prefix: "groq/"` and `backendLabel: "Groq"` / `backendLabel: null` out of the PROVIDERS table.
_PREFIX_RE = re.compile(r'prefix:\s*"([^"]+)"')
_BACKEND_LABEL_RE = re.compile(r'backendLabel:\s*(?:"([^"]+)"|null)')


def _ts_source() -> str:
    assert TS_REGISTRY.exists(), f"frontend provider registry not found at {TS_REGISTRY}"
    return TS_REGISTRY.read_text(encoding="utf-8")


def test_the_frontend_declares_exactly_the_prefixes_the_backend_routes():
    declared = set(_PREFIX_RE.findall(_ts_source()))
    assert declared == set(ROUTED_PREFIXES), (
        "src/config/providers.ts and providers/registry.py disagree about which prefixes exist. "
        f"backend only: {sorted(set(ROUTED_PREFIXES) - declared)}; "
        f"frontend only: {sorted(declared - set(ROUTED_PREFIXES))}"
    )


def test_the_frontend_uses_the_provider_names_the_market_actually_emits():
    """`backendLabel` is matched against ModelStats["provider"], so a mismatch silently breaks the
    provider filter for that provider - the rows simply never match any chip."""
    labels = {m.group(1) for m in _BACKEND_LABEL_RE.finditer(_ts_source()) if m.group(1)}
    # OpenRouter is `null` on purpose: as an aggregator it is identified by ABSENCE from this map.
    expected = {name for prefix, name in ROUTED_PREFIXES.items() if prefix != FALLBACK_PREFIX}
    assert labels == expected, (
        f"backendLabel values in providers.ts do not match the market's provider names. "
        f"missing: {sorted(expected - labels)}; unexpected: {sorted(labels - expected)}"
    )


@pytest.mark.parametrize("prefix", [p for p in ROUTED_PREFIXES if p != FALLBACK_PREFIX])
def test_every_registered_prefix_actually_routes(prefix):
    """Proves registry.py describes reality: a prefix that resolve_model() does not recognise falls
    through to the OpenRouter fallback, which is exactly the failure this guards against."""
    model_id = f"{prefix}some-model"
    routed, _extra = resolve_model(model_id)
    assert routed != f"openrouter/{model_id}", f"{prefix} fell through to the OpenRouter fallback"


def test_the_fallback_is_still_openrouter():
    """The frontend's FALLBACK_PROVIDER mirrors this: an unprefixed id is an OpenRouter id."""
    routed, _extra = resolve_model("vendor/unprefixed-model")
    assert routed == "openrouter/vendor/unprefixed-model"


def test_a_local_model_is_not_routed_as_openrouter():
    """The exact drift that shipped, pinned."""
    routed, extra = resolve_model("local/Ternary-Bonsai-27B")
    assert routed == "openai/Ternary-Bonsai-27B"
    assert "api_base" in extra


def test_a_local_model_never_carries_the_callers_provider_key():
    """BYOK is meaningless for a server on your own machine, and forwarding someone's OpenRouter
    key to 127.0.0.1 is a credential going somewhere it has no business being."""
    _routed, extra = resolve_model("local/x", user_key="sk-or-v1-the-users-openrouter-key")
    assert extra["api_key"] != "sk-or-v1-the-users-openrouter-key"

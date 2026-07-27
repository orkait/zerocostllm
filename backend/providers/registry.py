"""The model-id prefix contract, written down once.

A model id's prefix decides which provider serves it, and that decision is made in TWO codebases:
`resolve_model()` here, and `PROVIDERS` in src/config/providers.ts. They were mirrored by hand and
they drifted - the frontend had no `local/` entry, so a model running on the user's own machine was
routed as OpenRouter: the market filed it under the OpenRouter filter, and the chat client attached
the user's OpenRouter key to a request going to 127.0.0.1.

This module is the canonical list. test_registry.py holds both sides to it: every prefix here must
actually route in resolve_model(), and the TypeScript registry must declare exactly these prefixes
with exactly these provider labels. Adding a provider on one side alone is now a test failure
instead of a silent mislabelling.
"""

from .cerebras import CEREBRAS_MODEL_PREFIX, CEREBRAS_PROVIDER_NAME
from .cloudflare import CLOUDFLARE_MODEL_PREFIX, CLOUDFLARE_PROVIDER_NAME
from .groq import GROQ_MODEL_PREFIX, GROQ_PROVIDER_NAME
from .local import LOCAL_MODEL_PREFIX, LOCAL_PROVIDER_NAME
from .models import AISTUDIO_PROVIDER, OLLAMA_PROVIDER, OPENROUTER_PROVIDER

AISTUDIO_MODEL_PREFIX = "aistudio/"
OLLAMA_MODEL_PREFIX = "ollama/"
OPENROUTER_MODEL_PREFIX = "openrouter/"

#: model-id prefix -> the provider name the market reports for it (ModelStats["provider"]).
ROUTED_PREFIXES: dict[str, str] = {
    GROQ_MODEL_PREFIX: GROQ_PROVIDER_NAME,
    CEREBRAS_MODEL_PREFIX: CEREBRAS_PROVIDER_NAME,
    CLOUDFLARE_MODEL_PREFIX: CLOUDFLARE_PROVIDER_NAME,
    LOCAL_MODEL_PREFIX: LOCAL_PROVIDER_NAME,
    AISTUDIO_MODEL_PREFIX: AISTUDIO_PROVIDER,
    OLLAMA_MODEL_PREFIX: OLLAMA_PROVIDER,
    OPENROUTER_MODEL_PREFIX: OPENROUTER_PROVIDER,
}

#: OpenRouter is the fallback: an unprefixed id routes there, so it is the one prefix that cannot be
#: detected by "does it route somewhere other than the fallback".
FALLBACK_PREFIX = OPENROUTER_MODEL_PREFIX

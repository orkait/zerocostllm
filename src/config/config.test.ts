import { describe, it, expect } from "vitest";
import { PROVIDERS, PROVIDER_IDS, providerForModel, FALLBACK_PROVIDER, isProviderId, PROVIDER_BY_BACKEND_LABEL, isDirectlyRoutedProvider } from "./providers";
import { POOL_MODEL_ID, LEGACY_POOL_MODEL_ID, isPoolModel } from "./models";
import { AUTH_HEADERS } from "./api";
import { DATABASES } from "./storage";

describe("provider registry", () => {
  it("every id has a spec whose id matches its key", () => {
    for (const id of PROVIDER_IDS) expect(PROVIDERS[id].id).toBe(id);
  });

  it("routes a model id to the provider that owns its prefix", () => {
    // These prefixes are a contract with the backend's resolve_model(). If they drift, inference
    // is sent with the wrong provider's key.
    expect(providerForModel("groq/llama-3.3-70b-versatile")).toBe("groq");
    expect(providerForModel("cloudflare/@cf/openai/gpt-oss-120b")).toBe("cloudflare");
    expect(providerForModel("aistudio/gemini-2.5-flash")).toBe("aistudio");
    expect(providerForModel("cerebras/gemma-4-31b")).toBe("cerebras");
    expect(providerForModel("ollama/qwen3")).toBe("ollama");
    expect(providerForModel("openrouter/anthropic/claude-3-opus")).toBe("openrouter");
    expect(providerForModel("local/Ternary-Bonsai-27B")).toBe("local");
  });

  it("a local model is NOT mistaken for an OpenRouter one", () => {
    // The whole point of registering `local`. Without it providerForModel fell through to the
    // OpenRouter fallback, which sent the user's OpenRouter key in X-Provider-Key for a model
    // running on their own machine, and listed local models under the OpenRouter filter.
    expect(providerForModel("local/x")).not.toBe(FALLBACK_PROVIDER);
    expect(isDirectlyRoutedProvider("Local")).toBe(true);
    expect(PROVIDER_BY_BACKEND_LABEL["Local"]).toBe("local");
  });

  it("only the local provider takes no credential", () => {
    const keyless = PROVIDER_IDS.filter((id) => !PROVIDERS[id].needsKey);
    expect(keyless).toEqual(["local"]);
  });

  it("an unprefixed model falls back to OpenRouter, as the backend does", () => {
    expect(providerForModel("some/unprefixed-model")).toBe(FALLBACK_PROVIDER);
    expect(FALLBACK_PROVIDER).toBe("openrouter");
  });

  it("only Cloudflare needs a second credential", () => {
    const needy = PROVIDER_IDS.filter((id) => PROVIDERS[id].needsAccountId);
    expect(needy).toEqual(["cloudflare"]);
  });

  it("narrows unknown strings", () => {
    expect(isProviderId("groq")).toBe(true);
    expect(isProviderId("nope")).toBe(false);
  });
});

describe("pool model ids", () => {
  it("accepts both the current and the legacy id", () => {
    expect(isPoolModel(POOL_MODEL_ID)).toBe(true);
    expect(isPoolModel(LEGACY_POOL_MODEL_ID)).toBe(true);
  });

  it("keeps the legacy id alive - it is in users' persisted state", () => {
    expect(LEGACY_POOL_MODEL_ID).toBe("zero-cost-intelligent");
  });

  it("does not treat a real model as the pool", () => {
    expect(isPoolModel("groq/llama-3.3-70b-versatile")).toBe(false);
  });
});

describe("wire contracts", () => {
  it("BYOK header names match the backend exactly", () => {
    expect(AUTH_HEADERS.providerKey).toBe("X-Provider-Key");
    expect(AUTH_HEADERS.cloudflareAccountId).toBe("X-CF-Account-Id");
  });

  it("the chat database keeps its pre-rename name", () => {
    // Renaming this orphans every user's chat history.
    expect(DATABASES.chat.name).toBe("zerocostllm-chat");
  });
});

describe("backend provider labels", () => {
  it("maps the backend's names, which are NOT our UI labels", () => {
    // The backend says "Ollama"; we display "Ollama Cloud". Collapsing the two silently breaks the
    // provider filter, so the mapping is explicit and pinned here.
    expect(PROVIDER_BY_BACKEND_LABEL["Ollama"]).toBe("ollama");
    expect(PROVIDERS.ollama.label).toBe("Ollama Cloud");
    expect(PROVIDER_BY_BACKEND_LABEL["Google AI Studio"]).toBe("aistudio");
    expect(PROVIDER_BY_BACKEND_LABEL["Cloudflare Workers AI"]).toBe("cloudflare");
    expect(PROVIDER_BY_BACKEND_LABEL["Groq"]).toBe("groq");
    expect(PROVIDER_BY_BACKEND_LABEL["Cerebras"]).toBe("cerebras");
  });

  it("OpenRouter is an aggregator, identified by absence", () => {
    expect(PROVIDERS.openrouter.backendLabel).toBeNull();
    expect(isDirectlyRoutedProvider("Groq")).toBe(true);
    // An upstream that reached us via OpenRouter:
    expect(isDirectlyRoutedProvider("Anthropic")).toBe(false);
    expect(isDirectlyRoutedProvider("xAI")).toBe(false);
  });
});

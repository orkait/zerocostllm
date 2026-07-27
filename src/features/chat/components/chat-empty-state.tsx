"use client";
import { memo, useCallback, useMemo } from "react";
import Link from "next/link";
import { Sparkles } from "lucide-react";
import { ROUTES } from "@/config/routes";
import { useChatSessionStore } from "@/stores/chat-session-store";
import type { ModelStats } from "@/types/model";
import { cn } from "@/lib/utils";
import { CHAT_SURFACE_CLASS, EMPTY_STATE_MODEL_COUNT } from "../lib/chat-config";

type PickModel = (modelId: string) => void;

/** No model chosen yet - the chat has nothing to be. Offers the top free models as one-click starts,
 *  and the market for everything else. */
export function ChatEmptyState({ models }: { models: ModelStats[] }) {
  const startNewChat = useChatSessionStore((state) => state.startNewChat);

  const free = useMemo(() => models.filter((model) => model.is_free), [models]);
  const suggested = useMemo(() => free.slice(0, EMPTY_STATE_MODEL_COUNT), [free]);
  const providerCount = useMemo(
    () => new Set(free.map((model) => model.provider)).size,
    [free],
  );

  const handlePick = useCallback((modelId: string) => startNewChat(modelId), [startNewChat]);

  return (
    <div className={cn(CHAT_SURFACE_CLASS, "items-center justify-center p-6 text-center")}>
      <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-(--color-accent-soft) text-(--color-accent) mb-4">
        <Sparkles className="w-5 h-5" />
      </div>
      <h2 className="serif text-4xl font-semibold">Pick a model to start</h2>
      {/* Counted from the market that is actually loaded. It read "300+ free models across 7
          providers" as a hardcoded string, which was a claim nothing checked and which went stale
          the moment a provider was added. */}
      <p className="text-sm text-(--color-fg-muted) mt-1 mb-6">
        {free.length > 0
          ? `${free.length} free models across ${providerCount} providers.`
          : "Free models across every provider you have a key for."}
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-dialog w-full">
        {suggested.map((model) => (
          <ModelChoice key={model.id} model={model} onPick={handlePick} />
        ))}
      </div>
      <div className="mt-6 text-sm text-(--color-fg-subtle)">
        Or browse the full{" "}
        <Link href={ROUTES.market} className="text-(--color-accent) underline underline-offset-2">
          model market
        </Link>
        .
      </div>
    </div>
  );
}

const ModelChoice = memo(function ModelChoice({
  model,
  onPick,
}: {
  model: ModelStats;
  onPick: PickModel;
}) {
  const handleClick = useCallback(() => onPick(model.id), [onPick, model.id]);

  return (
    <button
      type="button"
      onClick={handleClick}
      className="text-left text-sm bg-(--color-surface-1) hover:bg-(--color-surface-2) border border-(--color-border) rounded-lg px-4 py-3 cursor-pointer transition-colors duration-120"
    >
      <div className="font-mono text-(--color-fg) truncate">{model.id}</div>
      <div className="text-xs text-(--color-fg-subtle) mt-0.5">
        {model.provider} · score {model.balanced.toFixed(1)}
      </div>
    </button>
  );
});

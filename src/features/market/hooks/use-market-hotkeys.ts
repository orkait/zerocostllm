"use client";
import { useCallback, useMemo } from "react";
import { useUIStore } from "@/stores/ui-store";
import { useGlobalHotkeys } from "@/lib/use-global-hotkeys";
import type { ModelStats } from "@/types/model";

const SEARCH_INPUT_SELECTOR = 'aside input[placeholder="Model id..."]';

type MarketHotkeyArgs = {
  filtered: readonly ModelStats[];
  drawerModelId: string | null;
  helpOpen: boolean;
  setHelpOpen: (next: boolean) => void;
};

/** Keyboard navigation for the market. Lifted out of the page so the page stays a composition of
 * components, and so the step logic can be reasoned about (and tested) on its own. */
export function useMarketHotkeys({ filtered, drawerModelId, helpOpen, setHelpOpen }: MarketHotkeyArgs): void {
  const { openDrawer, closeDrawer, setCmdk, cmdkOpen } = useUIStore();

  const stepDrawer = useCallback(
    (direction: 1 | -1) => {
      const current = drawerModelId ? filtered.findIndex((m) => m.id === drawerModelId) : -1;
      const next = filtered[clamp(current + direction, 0, filtered.length - 1)];
      if (next) openDrawer(next.id);
    },
    [drawerModelId, filtered, openDrawer],
  );

  const toggleCmdk = useCallback(() => setCmdk(!cmdkOpen), [cmdkOpen, setCmdk]);
  const focusSearch = useCallback(() => {
    document.querySelector<HTMLInputElement>(SEARCH_INPUT_SELECTOR)?.focus();
  }, []);
  const nextModel = useCallback(() => stepDrawer(1), [stepDrawer]);
  const prevModel = useCallback(() => stepDrawer(-1), [stepDrawer]);
  const toggleHelp = useCallback(() => setHelpOpen(!helpOpen), [helpOpen, setHelpOpen]);

  /** Escape closes exactly one layer, outermost first. */
  const closeTopOverlay = useCallback(() => {
    if (cmdkOpen) return setCmdk(false);
    if (drawerModelId) return closeDrawer();
    if (helpOpen) return setHelpOpen(false);
  }, [cmdkOpen, drawerModelId, helpOpen, setCmdk, closeDrawer, setHelpOpen]);

  // Memoized, like useShellOverlays does. A fresh object literal here is a new dependency every
  // render, so useGlobalHotkeys tore down and re-attached its window listener on every keystroke
  // that changed a filter.
  const handlers = useMemo(
    () => ({
      "cmdk.toggle": toggleCmdk,
      "search.focus": focusSearch,
      "drawer.next": nextModel,
      "drawer.prev": prevModel,
      "overlay.close": closeTopOverlay,
      "help.toggle": toggleHelp,
    }),
    [toggleCmdk, focusSearch, nextModel, prevModel, closeTopOverlay, toggleHelp],
  );

  useGlobalHotkeys(handlers);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

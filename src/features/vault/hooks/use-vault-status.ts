"use client";
import { useEffect } from "react";
import { useVaultStore, type VaultStatus } from "@/stores/vault-store";

/** Initialises the vault wherever the shell is mounted - i.e. everywhere.
 *
 * `init()` used to be called ONLY by the settings page, so on every other surface the store sat at
 * "loading" forever and nothing in the UI knew whether keys were available. The vault is in-memory
 * by design (persisting it would write decrypted keys to localStorage), which means a page reload
 * silently locks it. Combined, that produced the trap: reload, go to chat, type a message, send it,
 * and only THEN discover from a backend 401 that you needed to unlock.
 */
export function useVaultStatus(): VaultStatus {
  const status = useVaultStore((s) => s.status);
  const init = useVaultStore((s) => s.init);

  useEffect(() => {
    if (status === "loading") void init();
  }, [status, init]);

  return status;
}

import { PROVIDER_LIST } from "@/config/providers";
import { accountIdKey, type SecretKey } from "./vault";

/** The API-keys form, derived from the provider registry rather than re-declared beside it. Adding
 *  a provider to @/config/providers adds its field here; flipping `needsAccountId` adds the second
 *  credential. Neither one needs this file edited.
 *
 *  Providers with `needsKey: false` are skipped: a local server authenticates by being reachable,
 *  so a field for it would collect a credential nothing ever reads. */

export type VaultField = {
  key: SecretKey;
  label: string;
  /** Empty when the provider has no recognisable key shape to hint at. */
  placeholder: string;
};

export const VAULT_FIELDS: readonly VaultField[] = PROVIDER_LIST.flatMap((provider) => {
  if (!provider.needsKey) return [];
  const keyField: VaultField = {
    key: provider.id,
    label: provider.label,
    placeholder: provider.keyHint,
  };
  if (!provider.needsAccountId) return [keyField];
  return [
    keyField,
    {
      key: accountIdKey(provider.id),
      label: `${provider.label} account ID`,
      placeholder: `required with the ${provider.label} key`,
    },
  ];
});

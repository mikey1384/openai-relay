export const STAGE5_RELAY_BILLING_SERVICES = {
  TRANSCRIPTION: "transcription",
  TRANSLATION: "translation",
  TTS: "tts",
} as const;

export const STAGE5_RELAY_BILLING_ENDPOINTS = {
  AUTHORIZE: "/auth/authorize",
  CONFIRM: "/auth/confirm",
  RESERVE: "/auth/reserve",
  FINALIZE: "/auth/finalize",
  PERSIST: "/auth/persist",
  RELEASE: "/auth/release",
  REPLAY_STORE: "/auth/replay-store",
  REPLAY_LOAD: "/auth/replay-load",
  REPLAY_DELETE: "/auth/replay-delete",
} as const;

function normalizeBaseUrl(baseUrl: string): string {
  return String(baseUrl || "").trim().replace(/\/+$/, "");
}

export function buildStage5RelayBillingUrl(
  baseUrl: string,
  endpoint: (typeof STAGE5_RELAY_BILLING_ENDPOINTS)[keyof typeof STAGE5_RELAY_BILLING_ENDPOINTS]
): string {
  return `${normalizeBaseUrl(baseUrl)}${endpoint}`;
}

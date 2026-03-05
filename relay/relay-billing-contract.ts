export const STAGE5_RELAY_BILLING_SERVICES = {
  TRANSCRIPTION: "transcription",
  TRANSLATION: "translation",
  TTS: "tts",
} as const;

export const STAGE5_RELAY_BILLING_ENDPOINTS = {
  AUTHORIZE: "/auth/authorize",
  DEDUCT: "/auth/deduct",
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

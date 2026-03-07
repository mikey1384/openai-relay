function canonicalizeModelId(model?: string): string {
  return String(model || "").trim().toLowerCase();
}

export const DEFAULT_STAGE5_TRANSLATION_MODEL = "gpt-5.1";
export const STAGE5_REVIEW_TRANSLATION_MODEL = "gpt-5.4";
export const STAGE5_CLAUDE_OPUS_MODEL = "claude-opus-4-6";

export const STAGE5_TRANSLATION_MODEL_ALIASES = {
  "claude-opus-4.6": STAGE5_CLAUDE_OPUS_MODEL,
} as const;

export const STAGE5_ALLOWED_TRANSLATION_MODELS = [
  DEFAULT_STAGE5_TRANSLATION_MODEL,
  STAGE5_REVIEW_TRANSLATION_MODEL,
  STAGE5_CLAUDE_OPUS_MODEL,
] as const;

const STAGE5_ALLOWED_TRANSLATION_MODEL_SET = new Set<string>(
  STAGE5_ALLOWED_TRANSLATION_MODELS
);

export function normalizeStage5TranslationModel(model?: string): string {
  const canonical = canonicalizeModelId(model);
  if (!canonical) return DEFAULT_STAGE5_TRANSLATION_MODEL;
  return (
    STAGE5_TRANSLATION_MODEL_ALIASES[
      canonical as keyof typeof STAGE5_TRANSLATION_MODEL_ALIASES
    ] || canonical
  );
}

export function isAllowedStage5TranslationModelId(model?: string): boolean {
  return STAGE5_ALLOWED_TRANSLATION_MODEL_SET.has(
    normalizeStage5TranslationModel(model)
  );
}

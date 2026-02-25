export const DEFAULT_TRANSLATION_MODEL = "gpt-5.1";
export const CLAUDE_OPUS_MODEL = "claude-opus-4-6";
export const ALLOWED_STAGE5_TRANSLATION_MODELS = [
  DEFAULT_TRANSLATION_MODEL,
  CLAUDE_OPUS_MODEL,
] as const;

const MODEL_ALIASES: Record<string, string> = {
  "claude-opus-4.6": CLAUDE_OPUS_MODEL,
};

const ALLOWED_STAGE5_TRANSLATION_MODEL_SET = new Set<string>(
  ALLOWED_STAGE5_TRANSLATION_MODELS
);

function canonicalizeModelId(model: string): string {
  return (model || "").trim().toLowerCase();
}

// File upload limits
export const MAX_FILE_SIZE = 200 * 1024 * 1024; // 200MB

// Model provider detection
export function isClaudeModel(model: string): boolean {
  return canonicalizeModelId(model).startsWith("claude-");
}

export function normalizeModelId(model: string): string {
  const canonical = canonicalizeModelId(model);
  if (!canonical) return DEFAULT_TRANSLATION_MODEL;
  return MODEL_ALIASES[canonical] || canonical;
}

export function isAllowedStage5TranslationModel(model: string): boolean {
  return ALLOWED_STAGE5_TRANSLATION_MODEL_SET.has(normalizeModelId(model));
}

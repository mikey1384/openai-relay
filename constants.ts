import {
  DEFAULT_STAGE5_TRANSLATION_MODEL,
  STAGE5_CLAUDE_OPUS_MODEL,
  STAGE5_ALLOWED_TRANSLATION_MODELS,
  isAllowedStage5TranslationModelId,
  normalizeStage5TranslationModel,
} from "./model-catalog.js";

export const DEFAULT_TRANSLATION_MODEL = DEFAULT_STAGE5_TRANSLATION_MODEL;
export const CLAUDE_OPUS_MODEL = STAGE5_CLAUDE_OPUS_MODEL;
export const ALLOWED_STAGE5_TRANSLATION_MODELS = STAGE5_ALLOWED_TRANSLATION_MODELS;
export const DEFAULT_CF_API_BASE = "https://api.stage5.tools";

function normalizeBaseUrl(url: string): string {
  return url.replace(/\/+$/, "");
}

export const CF_API_BASE = normalizeBaseUrl(
  (process.env.CF_API_BASE || DEFAULT_CF_API_BASE).trim()
);

// File upload limits
export const MAX_FILE_SIZE = 200 * 1024 * 1024; // 200MB

// Model provider detection
export function isClaudeModel(model: string): boolean {
  return String(model || "")
    .trim()
    .toLowerCase()
    .startsWith("claude-");
}

export function normalizeModelId(model: string): string {
  return normalizeStage5TranslationModel(model);
}

export function isAllowedStage5TranslationModel(model: string): boolean {
  return isAllowedStage5TranslationModelId(model);
}

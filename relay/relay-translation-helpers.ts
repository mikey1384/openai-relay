import { claudeModelAlwaysThinks } from "../anthropic-config.js";
import {
  STAGE5_CLAUDE_HAIKU_5_5_MODEL,
  STAGE5_CLAUDE_OPUS_5_5_MODEL,
} from "../model-catalog.js";
import {
  DEFAULT_TRANSLATION_MODEL,
  isClaudeModel,
  normalizeModelId,
  STAGE5_REVIEW_MODEL,
} from "../constants.js";

export type TranslationModelFamily = "gpt" | "claude" | "auto";

export const DEFAULT_TRANSLATION_MAX_COMPLETION_TOKENS = 16_000;
export const EXTENDED_TRANSLATION_MAX_COMPLETION_TOKENS = 32_000;

export function isLikelySubtitleReviewMessages(
  messages: Array<{ role: string; content: string }> | undefined
): boolean {
  if (!Array.isArray(messages) || messages.length === 0) return false;
  const systemText = messages
    .filter((m) => String(m?.role ?? "").toLowerCase() === "system")
    .map((m) => String(m?.content ?? ""))
    .join("\n")
    .toLowerCase();

  if (!systemText) return false;

  return (
    systemText.includes("subtitle reviewer.") &&
    systemText.includes("output exactly") &&
    systemText.includes("@@sub_line@@") &&
    systemText.includes("no commentary.")
  );
}

export function isLikelySubtitleDraftMessages(
  messages: Array<{ role: string; content: string }> | undefined
): boolean {
  if (!Array.isArray(messages) || messages.length === 0) return false;
  const systemText = messages
    .filter((m) => String(m?.role ?? "").toLowerCase() === "system")
    .map((m) => String(m?.content ?? ""))
    .join("\n")
    .toLowerCase();

  if (!systemText) return false;

  return (
    systemText.includes("subtitle translator.") &&
    systemText.includes("output exactly") &&
    systemText.includes("@@sub_line@@")
  );
}

export function parseTranslationPhase(
  raw: unknown
): "draft" | "review" | undefined {
  const phase =
    typeof raw === "string" ? raw.trim().toLowerCase() : undefined;
  if (phase === "draft" || phase === "review") {
    return phase;
  }
  return undefined;
}

export function parseTranslationModelFamily(
  raw: unknown
): TranslationModelFamily | undefined {
  const family =
    typeof raw === "string" ? raw.trim().toLowerCase() : undefined;
  if (family === "gpt" || family === "claude" || family === "auto") {
    return family;
  }
  if (family === "openai") return "gpt";
  if (family === "anthropic") return "claude";
  return undefined;
}

export function parseBooleanLike(raw: unknown): boolean | undefined {
  if (typeof raw === "boolean") return raw;
  if (typeof raw !== "string") return undefined;
  const normalized = raw.trim().toLowerCase();
  if (!normalized) return undefined;
  if (["1", "true", "yes", "on", "high", "quality"].includes(normalized)) {
    return true;
  }
  if (["0", "false", "no", "off", "low", "standard"].includes(normalized)) {
    return false;
  }
  return undefined;
}

export function resolveTranslationModel({
  rawModel,
  modelFamily,
  messages,
  canUseAnthropic,
  translationPhase,
  qualityMode,
}: {
  rawModel?: string;
  modelFamily?: TranslationModelFamily;
  messages?: Array<{ role: string; content: string }>;
  canUseAnthropic: boolean;
  translationPhase?: "draft" | "review";
  qualityMode?: boolean;
}): string {
  const normalized = normalizeModelId(rawModel || DEFAULT_TRANSLATION_MODEL);
  const reviewByHeuristic = isLikelySubtitleReviewMessages(messages);
  const draftByHeuristic = isLikelySubtitleDraftMessages(messages);
  const isSubtitleWorkflow =
    translationPhase === "review" ||
    translationPhase === "draft" ||
    reviewByHeuristic ||
    draftByHeuristic;

  if (!isSubtitleWorkflow) {
    return normalized;
  }

  const effectivePhase =
    translationPhase === "review"
      ? "review"
      : translationPhase === "draft"
        ? "draft"
        : qualityMode === false
          ? "draft"
          : reviewByHeuristic
            ? "review"
            : "draft";

  // Credit-mode subtitles use Anthropic's current models (2026-10-13 trials):
  // review on Opus 5.5 (about half of GPT-5.5's cost, more complete and
  // consistent), drafts on Haiku 5.5 (about a tenth of GPT-5.1's cost). The
  // GPT models remain only as the fallback when Anthropic is unavailable; GPT
  // choices belong to API-key users, whose requests don't come through Stage5.
  const selectedModel =
    effectivePhase === "review"
      ? canUseAnthropic
        ? STAGE5_CLAUDE_OPUS_5_5_MODEL
        : STAGE5_REVIEW_MODEL
      : canUseAnthropic
        ? STAGE5_CLAUDE_HAIKU_5_5_MODEL
        : DEFAULT_TRANSLATION_MODEL;

  if (selectedModel !== normalized) {
    console.log(
      `🧭 Server model authority selected ${selectedModel} (requested=${normalized}, phase=${
        translationPhase ?? effectivePhase
      }, modelFamily=${modelFamily ?? "auto"}, qualityMode=${
        typeof qualityMode === "boolean" ? qualityMode : "auto"
      })`,
    );
  }

  return selectedModel;
}

// Output allowance for a plain translation part, sized from its prompt. Over
// 11,394 settled production parts (2026-06..10) output never exceeded 0.75x
// the prompt (largest 2,161 tokens), so 2x + 1,024 leaves >2.6x headroom. The
// returned value both sizes the credit hold and caps the provider call, so
// actual spend still can never exceed the hold. A flat 16,000 held ~40x the
// real cost per part, and parallel parts then starved each other on modest
// balances: users who could afford the job were refused (2026-10-12).
export const TRANSLATION_OUTPUT_TOKENS_PER_PROMPT_TOKEN = 2;
export const TRANSLATION_OUTPUT_BASE_TOKENS = 1_024;
export const ALWAYS_THINKING_OUTPUT_BASE_TOKENS = 4_096;

export function resolveTranslationReservationMaxCompletionTokens({
  model,
  reasoning,
  promptTokens,
  usesToolsOrWebSearch = false,
}: {
  model: string;
  reasoning?: { effort?: "low" | "medium" | "high" | "xhigh" } | null;
  promptTokens?: number;
  usesToolsOrWebSearch?: boolean;
}): number {
  const effort = String(reasoning?.effort || "").trim().toLowerCase();
  const deepReasoning =
    effort === "medium" || effort === "high" || effort === "xhigh";
  if (isClaudeModel(model) && deepReasoning) {
    return EXTENDED_TRANSLATION_MAX_COMPLETION_TOKENS;
  }
  // Reasoning, tool calls and web search can legitimately produce output
  // unrelated to prompt size (and Claude thinking raises its own limit), so
  // they keep the flat cap. Callers without a prompt estimate do too.
  if (
    deepReasoning ||
    usesToolsOrWebSearch ||
    typeof promptTokens !== "number" ||
    !Number.isFinite(promptTokens) ||
    promptTokens <= 0
  ) {
    return DEFAULT_TRANSLATION_MAX_COMPLETION_TOKENS;
  }
  if (isClaudeModel(model) && claudeModelAlwaysThinks(model)) {
    // Opus 5.5 thinks on every call (at our explicit low effort). On the
    // 2026-10-13 trial its output including thinking peaked at 0.38x the
    // prompt (largest 1,321 tokens), so 2x + 4,096 leaves ~7x headroom.
    return Math.min(
      EXTENDED_TRANSLATION_MAX_COMPLETION_TOKENS,
      Math.ceil(promptTokens * TRANSLATION_OUTPUT_TOKENS_PER_PROMPT_TOKEN) +
        ALWAYS_THINKING_OUTPUT_BASE_TOKENS,
    );
  }
  return Math.min(
    DEFAULT_TRANSLATION_MAX_COMPLETION_TOKENS,
    Math.ceil(promptTokens * TRANSLATION_OUTPUT_TOKENS_PER_PROMPT_TOKEN) +
      TRANSLATION_OUTPUT_BASE_TOKENS,
  );
}

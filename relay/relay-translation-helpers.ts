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
  void modelFamily;
  void canUseAnthropic;
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

  const selectedModel =
    effectivePhase === "review"
      ? STAGE5_REVIEW_MODEL
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

export function resolveTranslationReservationMaxCompletionTokens({
  model,
  reasoning,
}: {
  model: string;
  reasoning?: { effort?: "low" | "medium" | "high" } | null;
}): number {
  const effort = String(reasoning?.effort || "").trim().toLowerCase();
  if (isClaudeModel(model) && (effort === "medium" || effort === "high")) {
    return EXTENDED_TRANSLATION_MAX_COMPLETION_TOKENS;
  }
  return DEFAULT_TRANSLATION_MAX_COMPLETION_TOKENS;
}

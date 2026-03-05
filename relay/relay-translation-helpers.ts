export type TranslationModelFamily = "gpt" | "claude" | "auto";

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

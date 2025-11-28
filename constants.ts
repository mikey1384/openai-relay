export const DEFAULT_TRANSLATION_MODEL = "gpt-5.1";
export const CLAUDE_OPUS_MODEL = "claude-opus-4-5-20251101";

// File upload limits
export const MAX_FILE_SIZE = 200 * 1024 * 1024; // 200MB

// Model provider detection
export function isClaudeModel(model: string): boolean {
  return model.startsWith("claude-");
}

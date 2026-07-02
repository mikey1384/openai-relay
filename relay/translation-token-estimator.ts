import { encodingForModel } from "js-tiktoken";
import { isClaudeModel } from "../constants.js";

type TiktokenEncoder = {
  encode(value: string): number[];
};

const textEncoder = new TextEncoder();
const encoderCache = new Map<string, TiktokenEncoder>();
const GPT_PROMPT_BASE_OVERHEAD_TOKENS = 16;
const GPT_PROMPT_PER_MESSAGE_OVERHEAD_TOKENS = 8;
const CLAUDE_PROMPT_BASE_OVERHEAD_TOKENS = 32;
const CLAUDE_PROMPT_PER_MESSAGE_OVERHEAD_TOKENS = 16;
const GPT_COMPLETION_FALLBACK_OVERHEAD_TOKENS = 8;
const CLAUDE_COMPLETION_FALLBACK_OVERHEAD_TOKENS = 16;

function getUtf8Length(value: string): number {
  return textEncoder.encode(value).length;
}

function normalizeOpenAiTokenizerModel(model: string): string {
  const normalized = String(model || "").trim().toLowerCase();
  if (normalized.startsWith("gpt-5")) {
    return "gpt-5";
  }
  if (normalized.startsWith("gpt-4o")) {
    return "gpt-4o";
  }
  return "gpt-5";
}

function getOpenAiEncoder(model: string): TiktokenEncoder | null {
  const key = normalizeOpenAiTokenizerModel(model);
  const cached = encoderCache.get(key);
  if (cached) {
    return cached;
  }

  try {
    const encoder = encodingForModel(key as any) as unknown as TiktokenEncoder;
    encoderCache.set(key, encoder);
    return encoder;
  } catch {
    return null;
  }
}

function normalizeUnknownToString(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((entry) => normalizeUnknownToString(entry)).join("\n");
  }
  if (value && typeof value === "object") {
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }
  if (value == null) {
    return "";
  }
  return String(value);
}

function normalizeMessages(messages: unknown): Array<{
  role: string;
  content: string;
  tool_calls?: unknown;
  tool_call_id?: string;
}> {
  if (!Array.isArray(messages)) {
    return [];
  }

  return messages.map((message) => {
    const role =
      typeof (message as any)?.role === "string"
        ? (message as any).role
        : "user";
    const content = normalizeUnknownToString((message as any)?.content);
    // Tool-calling turns carry structured fields that count as prompt
    // tokens at the provider; keep them in the estimation source.
    const toolCalls = Array.isArray((message as any)?.tool_calls)
      ? (message as any).tool_calls
      : undefined;
    const toolCallId =
      typeof (message as any)?.tool_call_id === "string"
        ? (message as any).tool_call_id
        : undefined;
    return {
      role,
      content,
      ...(toolCalls ? { tool_calls: toolCalls } : {}),
      ...(toolCallId ? { tool_call_id: toolCallId } : {}),
    };
  });
}

function getPromptSource({
  messages,
  payload,
}: {
  messages?: unknown;
  payload?: Record<string, unknown> | null;
}): { source: string; messageCount: number } {
  const directMessages = normalizeMessages(messages);
  const payloadMessages = normalizeMessages(payload?.messages);
  const normalizedMessages =
    directMessages.length > 0 ? directMessages : payloadMessages;

  if (normalizedMessages.length > 0) {
    return {
      source: JSON.stringify(normalizedMessages),
      messageCount: normalizedMessages.length,
    };
  }

  const rawPayload =
    payload && typeof payload === "object"
      ? payload
      : { payload: normalizeUnknownToString(payload) };

  return {
    source: JSON.stringify(rawPayload),
    messageCount: 1,
  };
}

function getCompletionText(completion: unknown): string {
  const rawContent =
    (completion as any)?.content ??
    (completion as any)?.choices?.[0]?.message?.content ??
    "";
  return normalizeUnknownToString(rawContent);
}

export function estimateTranslationPromptTokenReserve({
  model,
  messages,
  payload,
  tools,
}: {
  model: string;
  messages?: unknown;
  payload?: Record<string, unknown> | null;
  tools?: unknown;
}): number {
  const promptSource = getPromptSource({ messages, payload });
  // Tool schemas ride in the prompt at the provider; leaving them out of
  // the reserve can make actual spend exceed the reservation, which fails
  // finalization after the model call already succeeded.
  const toolsSource =
    tools && Array.isArray(tools) && tools.length > 0
      ? JSON.stringify(tools)
      : "";
  const source = promptSource.source + toolsSource;
  const messageCount = promptSource.messageCount;

  if (isClaudeModel(model)) {
    return (
      getUtf8Length(source) +
      CLAUDE_PROMPT_BASE_OVERHEAD_TOKENS +
      CLAUDE_PROMPT_PER_MESSAGE_OVERHEAD_TOKENS * Math.max(1, messageCount)
    );
  }

  const encoder = getOpenAiEncoder(model);
  if (!encoder) {
    return (
      getUtf8Length(source) +
      CLAUDE_PROMPT_BASE_OVERHEAD_TOKENS +
      CLAUDE_PROMPT_PER_MESSAGE_OVERHEAD_TOKENS * Math.max(1, messageCount)
    );
  }

  return (
    encoder.encode(source).length +
    GPT_PROMPT_BASE_OVERHEAD_TOKENS +
    GPT_PROMPT_PER_MESSAGE_OVERHEAD_TOKENS * Math.max(1, messageCount)
  );
}

export function estimateTranslationCompletionTokensFallback({
  model,
  completion,
  maxCompletionTokens,
}: {
  model: string;
  completion: unknown;
  maxCompletionTokens?: number;
}): number {
  const text = getCompletionText(completion);
  let estimate: number;

  if (isClaudeModel(model)) {
    estimate = getUtf8Length(text) + CLAUDE_COMPLETION_FALLBACK_OVERHEAD_TOKENS;
  } else {
    const encoder = getOpenAiEncoder(model);
    estimate = encoder
      ? encoder.encode(text).length + GPT_COMPLETION_FALLBACK_OVERHEAD_TOKENS
      : getUtf8Length(text) + CLAUDE_COMPLETION_FALLBACK_OVERHEAD_TOKENS;
  }

  if (typeof maxCompletionTokens === "number" && Number.isFinite(maxCompletionTokens)) {
    return Math.min(Math.max(0, Math.ceil(maxCompletionTokens)), estimate);
  }

  return estimate;
}

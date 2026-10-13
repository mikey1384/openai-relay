import Anthropic from "@anthropic-ai/sdk";

export function makeAnthropic(apiKey: string) {
  return new Anthropic({
    apiKey,
    timeout: 600_000,
    maxRetries: 3,
  });
}

export interface ClaudeMessage {
  role: "user" | "assistant";
  content: string;
}

export interface ChatToolDefinition {
  type: "function";
  function: {
    name: string;
    description?: string;
    parameters: Record<string, unknown>;
  };
}

export interface ChatToolCall {
  id: string;
  type: "function";
  function: {
    name: string;
    arguments: string;
  };
}

export type ChatToolChoice = "auto" | "required";

interface ChatToolMessage {
  role: string;
  content: string | null;
  tool_calls?: ChatToolCall[];
  tool_call_id?: string;
}

function parseToolArguments(raw: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(String(raw || "{}"));
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed
      : {};
  } catch {
    return {};
  }
}

/**
 * Map chat-completions-style messages (system/user/assistant with
 * tool_calls, and role:'tool' results) onto Anthropic Messages blocks.
 */
function toAnthropicMessages(messages: ChatToolMessage[]): {
  systemPrompt: string | undefined;
  anthropicMessages: Array<{ role: "user" | "assistant"; content: any }>;
} {
  let systemPrompt: string | undefined;
  const anthropicMessages: Array<{ role: "user" | "assistant"; content: any }> =
    [];

  const pushToolResult = (toolCallId: string, content: string) => {
    const block = {
      type: "tool_result",
      tool_use_id: toolCallId,
      content,
    };
    const last = anthropicMessages[anthropicMessages.length - 1];
    // Anthropic requires tool results for parallel tool calls to share
    // one user message.
    if (last && last.role === "user" && Array.isArray(last.content)) {
      last.content.push(block);
      return;
    }
    anthropicMessages.push({ role: "user", content: [block] });
  };

  for (const msg of messages) {
    if (msg.role === "system") {
      systemPrompt = String(msg.content || "");
      continue;
    }
    if (msg.role === "tool") {
      if (msg.tool_call_id) {
        pushToolResult(msg.tool_call_id, String(msg.content || ""));
      }
      continue;
    }
    if (msg.role === "assistant") {
      const blocks: any[] = [];
      const text = String(msg.content || "");
      if (text.trim()) {
        blocks.push({ type: "text", text });
      }
      for (const call of msg.tool_calls || []) {
        blocks.push({
          type: "tool_use",
          id: call.id,
          name: call.function.name,
          input: parseToolArguments(call.function.arguments),
        });
      }
      if (blocks.length > 0) {
        anthropicMessages.push({ role: "assistant", content: blocks });
      }
      continue;
    }
    if (msg.role === "user") {
      const text = String(msg.content || "");
      if (text.trim()) {
        anthropicMessages.push({ role: "user", content: text });
      }
    }
  }

  if (anthropicMessages.length === 0 || anthropicMessages[0].role !== "user") {
    anthropicMessages.unshift({ role: "user", content: "Please proceed." });
  }

  return { systemPrompt, anthropicMessages };
}

function toAnthropicTools(tools: ChatToolDefinition[]): Array<{
  name: string;
  description?: string;
  input_schema: Record<string, unknown>;
}> {
  return tools.map((tool) => ({
    name: tool.function.name,
    description: tool.function.description,
    input_schema: tool.function.parameters,
  }));
}

function extractAnthropicToolCalls(
  content: Array<{ type: string; [key: string]: unknown }>,
): ChatToolCall[] {
  const toolCalls: ChatToolCall[] = [];
  for (const block of content) {
    if (block.type !== "tool_use") continue;
    toolCalls.push({
      id: String(block.id || ""),
      type: "function",
      function: {
        name: String(block.name || ""),
        arguments: JSON.stringify(block.input ?? {}),
      },
    });
  }
  return toolCalls;
}

export type ClaudeEffort = "low" | "medium" | "high" | "xhigh";

// Legacy extended thinking budget tokens for Claude models that still support
// budget_tokens.
const THINKING_BUDGET: Record<"low" | "medium" | "high", number> = {
  low: 0, // No extended thinking
  medium: 8000, // Moderate reasoning
  high: 16000, // Deep reasoning
};

/**
 * How each Claude model takes thinking and tool settings. Stated once per
 * model so new IDs never fall into an older model's request shape:
 * - "budget": thinking via budget_tokens; off by omitting `thinking`.
 * - "adaptive-optional": adaptive thinking on request; off by omission.
 * - "adaptive-always": thinking can't be disabled (`disabled` and
 *   budget_tokens are 400s); effort is the only control (Opus 5.5).
 * - "between-tools-off": `disabled` is a 400; off is {type:"between_tools"},
 *   adaptive on request (Sonnet 5.5).
 * - "disabled-off": thinking runs unless turned off with {type:"disabled"}
 *   (valid at effort high or below, the default), adaptive on request
 *   (Haiku 5.5).
 * forcedToolChoice=false: tool_choice any/tool is a 400; use auto plus an
 * instruction and verify a tool was called.
 */
export type ClaudeThinkingMode =
  | "budget"
  | "adaptive-optional"
  | "adaptive-always"
  | "between-tools-off"
  | "disabled-off";

export interface ClaudeModelCapabilities {
  thinking: ClaudeThinkingMode;
  forcedToolChoice: boolean;
  webSearchTool: "web_search_20250305" | "web_search_20260209";
}

const LEGACY_CLAUDE_CAPABILITIES: ClaudeModelCapabilities = {
  thinking: "budget",
  forcedToolChoice: true,
  webSearchTool: "web_search_20250305",
};

const CLAUDE_MODEL_CAPABILITIES: Record<string, ClaudeModelCapabilities> = {
  "claude-opus-4-8": {
    thinking: "adaptive-optional",
    forcedToolChoice: true,
    webSearchTool: "web_search_20250305",
  },
  "claude-sonnet-5": {
    thinking: "adaptive-optional",
    forcedToolChoice: true,
    webSearchTool: "web_search_20250305",
  },
  "claude-opus-5-5": {
    thinking: "adaptive-always",
    forcedToolChoice: false,
    webSearchTool: "web_search_20260209",
  },
  "claude-sonnet-5-5": {
    thinking: "between-tools-off",
    forcedToolChoice: false,
    webSearchTool: "web_search_20260209",
  },
  "claude-haiku-5-5": {
    thinking: "disabled-off",
    forcedToolChoice: true,
    webSearchTool: "web_search_20250305",
  },
};

export function getClaudeModelCapabilities(
  model: string,
): ClaudeModelCapabilities {
  return CLAUDE_MODEL_CAPABILITIES[model] ?? LEGACY_CLAUDE_CAPABILITIES;
}

/** True when the model thinks on every request, so output limits need room. */
export function claudeModelAlwaysThinks(model: string): boolean {
  return getClaudeModelCapabilities(model).thinking === "adaptive-always";
}

type ClaudeThinkingConfig = {
  // Whether reasoning tokens may be produced (affects max_tokens headroom and
  // the legacy system-prompt fold).
  thinkingOn: boolean;
  maxTokens: number;
  apply?: (requestParams: any) => void;
  logMessage?: string;
};

export function resolveClaudeThinkingConfig({
  model,
  maxTokens,
  effort,
  forceToolUse = false,
}: {
  model: string;
  maxTokens: number;
  effort?: ClaudeEffort;
  forceToolUse?: boolean;
}): ClaudeThinkingConfig {
  const { thinking: mode } = getClaudeModelCapabilities(model);
  const wantsThinking = Boolean(effort && effort !== "low");

  if (mode === "adaptive-always") {
    const resolvedEffort = effort ?? "low";
    return {
      thinkingOn: true,
      maxTokens: Math.max(maxTokens, MAX_TOKENS_WITH_THINKING),
      apply: (requestParams) => {
        requestParams.thinking = { type: "adaptive" };
        requestParams.output_config = {
          ...(requestParams.output_config || {}),
          effort: resolvedEffort,
        };
      },
      logMessage: `[anthropic-config] Adaptive thinking (always on) with effort: ${resolvedEffort}`,
    };
  }

  if (mode === "disabled-off") {
    if (wantsThinking && !forceToolUse) {
      return {
        thinkingOn: true,
        maxTokens: Math.max(maxTokens, MAX_TOKENS_WITH_THINKING),
        apply: (requestParams) => {
          requestParams.thinking = { type: "adaptive" };
          requestParams.output_config = {
            ...(requestParams.output_config || {}),
            effort,
          };
        },
        logMessage: `[anthropic-config] Adaptive thinking enabled with effort: ${effort}`,
      };
    }
    return {
      thinkingOn: false,
      maxTokens,
      apply: (requestParams) => {
        requestParams.thinking = { type: "disabled" };
      },
    };
  }

  if (mode === "between-tools-off") {
    if (wantsThinking && !forceToolUse) {
      return {
        thinkingOn: true,
        maxTokens: Math.max(maxTokens, MAX_TOKENS_WITH_THINKING),
        apply: (requestParams) => {
          requestParams.thinking = { type: "adaptive" };
          requestParams.output_config = {
            ...(requestParams.output_config || {}),
            effort,
          };
        },
        logMessage: `[anthropic-config] Adaptive thinking enabled with effort: ${effort}`,
      };
    }
    // `between_tools` takes no other field and is valid only at effort high
    // or below (the default), so effort is not sent with it.
    return {
      thinkingOn: false,
      maxTokens,
      apply: (requestParams) => {
        requestParams.thinking = { type: "between_tools" };
      },
    };
  }

  if (!wantsThinking || forceToolUse) {
    return { thinkingOn: false, maxTokens };
  }

  if (mode === "adaptive-optional") {
    return {
      thinkingOn: true,
      maxTokens: Math.max(maxTokens, MAX_TOKENS_WITH_THINKING),
      apply: (requestParams) => {
        requestParams.thinking = { type: "adaptive" };
        requestParams.output_config = {
          ...(requestParams.output_config || {}),
          effort,
        };
      },
      logMessage: `[anthropic-config] Adaptive thinking enabled with effort: ${effort}`,
    };
  }

  const legacyEffort = effort === "xhigh" ? "high" : effort!;
  const budgetTokens = THINKING_BUDGET[legacyEffort as "medium" | "high"];
  return {
    thinkingOn: true,
    maxTokens: Math.min(
      MAX_TOKENS_WITH_THINKING,
      Math.max(maxTokens, budgetTokens + 1024),
    ),
    apply: (requestParams) => {
      requestParams.thinking = {
        type: "enabled",
        budget_tokens: budgetTokens,
      };
    },
    logMessage: `[anthropic-config] Extended thinking enabled with budget: ${budgetTokens} tokens`,
  };
}

const REQUIRED_TOOL_INSTRUCTION =
  "Respond by calling one of the provided tools. Do not answer in plain text.";

/**
 * Turn non-answers into errors so callers release the credit hold instead of
 * billing them: a refusal (stop_reason "refusal"), output cut off at
 * max_tokens, or a missing tool call the caller required. Previously a
 * refusal returned empty text as success, was billed, and the app retried
 * and was billed again.
 */
export class ClaudeIncompleteResponseError extends Error {
  constructor(
    readonly reason: "refusal" | "max_tokens" | "missing_tool_call",
    detail?: string,
  ) {
    super(
      `Claude response not usable (${reason}${detail ? `: ${detail}` : ""})`,
    );
    this.name = "ClaudeIncompleteResponseError";
  }
}

export function assertClaudeCompletion(
  response: { stop_reason?: string | null; stop_details?: any; content?: any[] },
  { requireToolCall = false }: { requireToolCall?: boolean } = {},
): void {
  if (response?.stop_reason === "refusal") {
    throw new ClaudeIncompleteResponseError(
      "refusal",
      response?.stop_details?.category ?? undefined,
    );
  }
  if (response?.stop_reason === "max_tokens") {
    throw new ClaudeIncompleteResponseError("max_tokens");
  }
  if (
    requireToolCall &&
    !(response?.content || []).some((block: any) => block?.type === "tool_use")
  ) {
    throw new ClaudeIncompleteResponseError("missing_tool_call");
  }
}

export interface ClaudeTranslateOptions {
  messages: ClaudeMessage[];
  model: string;
  apiKey: string;
  signal?: AbortSignal;
  maxTokens?: number;
  effort?: ClaudeEffort;
  tools?: ChatToolDefinition[];
  toolChoice?: ChatToolChoice;
}

const MAX_TOKENS_DEFAULT = 16000;
const MAX_TOKENS_WITH_THINKING = 32000;

export async function translateWithClaude({
  messages,
  model,
  apiKey,
  signal,
  maxTokens = MAX_TOKENS_DEFAULT,
  effort,
  tools,
  toolChoice,
}: ClaudeTranslateOptions): Promise<{
  model: string;
  choices: Array<{
    message: {
      role: string;
      content: string;
      tool_calls?: ChatToolCall[];
    };
  }>;
  usage: { prompt_tokens: number; completion_tokens: number };
}> {
  const client = makeAnthropic(apiKey);
  const hasTools = Array.isArray(tools) && tools.length > 0;
  const forceToolUse = hasTools && toolChoice === "required";
  const capabilities = getClaudeModelCapabilities(model);
  // Models that reject forced tool_choice get auto plus an instruction, and
  // the response is checked for the required call below.
  const instructToolUse = forceToolUse && !capabilities.forcedToolChoice;

  const { systemPrompt, anthropicMessages } = toAnthropicMessages(
    messages as ChatToolMessage[],
  );

  // Anthropic rejects forced tool_choice combined with thinking. The tool
  // contract wins: thinking is dropped for a forced call where the model
  // allows it (resolveClaudeThinkingConfig handles models that always think).
  const thinkingConfig = resolveClaudeThinkingConfig({
    model,
    maxTokens,
    effort,
    forceToolUse,
  });

  // Build request parameters
  const requestParams: any = {
    model,
    max_tokens: thinkingConfig.maxTokens,
    messages: anthropicMessages,
  };

  const effectiveSystemPrompt = instructToolUse
    ? [systemPrompt, REQUIRED_TOOL_INSTRUCTION].filter(Boolean).join("\n\n")
    : systemPrompt;
  const foldsSystemPrompt =
    capabilities.thinking === "budget" ||
    capabilities.thinking === "adaptive-optional";

  if (hasTools) {
    requestParams.tools = toAnthropicTools(tools!);
    if (forceToolUse && capabilities.forcedToolChoice) {
      requestParams.tool_choice = { type: "any" };
    }
    // Tool flows need the system prompt intact — never fold it into the
    // first user turn.
    if (effectiveSystemPrompt) {
      requestParams.system = effectiveSystemPrompt;
    }
  } else if (
    effectiveSystemPrompt &&
    thinkingConfig.thinkingOn &&
    foldsSystemPrompt
  ) {
    // Older models: prepend system context to the first user message when
    // using extended thinking.
    const first = anthropicMessages[0];
    if (typeof first.content === "string") {
      first.content = `${effectiveSystemPrompt}\n\n${first.content}`;
    } else {
      requestParams.system = effectiveSystemPrompt;
    }
  } else if (effectiveSystemPrompt) {
    requestParams.system = effectiveSystemPrompt;
  }

  if (thinkingConfig.apply) {
    thinkingConfig.apply(requestParams);
    if (thinkingConfig.logMessage) console.log(thinkingConfig.logMessage);
  }

  const response = await client.messages.create(requestParams, { signal });
  assertClaudeCompletion(response as any, { requireToolCall: forceToolUse });

  // Extract text content, handling both regular and thinking responses
  let textContent = '';
  for (const block of response.content) {
    if (block.type === 'text') {
      textContent += block.text;
    }
    // Skip 'thinking' blocks - they contain internal reasoning
  }

  const toolCalls = extractAnthropicToolCalls(response.content as any);

  return {
    model,
    choices: [
      {
        message: {
          role: "assistant",
          content: textContent,
          ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
        },
      },
    ],
    usage: {
      prompt_tokens: response.usage.input_tokens,
      completion_tokens: response.usage.output_tokens,
    },
  };
}

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

type ClaudeEffort = "low" | "medium" | "high" | "xhigh";

// Legacy extended thinking budget tokens for Claude models that still support
// budget_tokens. Claude Opus 4.8 uses adaptive thinking instead.
const THINKING_BUDGET: Record<"low" | "medium" | "high", number> = {
  low: 0, // No extended thinking
  medium: 8000, // Moderate reasoning
  high: 16000, // Deep reasoning
};

const CLAUDE_OPUS_4_8_MODEL = "claude-opus-4-8";
const CLAUDE_SONNET_5_MODEL = "claude-sonnet-5";

// Claude 4.8+/5-family models use adaptive thinking; older models still
// take explicit budget_tokens.
function usesAdaptiveThinking(model: string): boolean {
  return model === CLAUDE_OPUS_4_8_MODEL || model === CLAUDE_SONNET_5_MODEL;
}

type ClaudeThinkingConfig =
  | { enabled: false; maxTokens: number }
  | {
      enabled: true;
      maxTokens: number;
      apply: (requestParams: any) => void;
      logMessage: string;
    };

function resolveClaudeThinkingConfig({
  model,
  maxTokens,
  effort,
}: {
  model: string;
  maxTokens: number;
  effort?: ClaudeEffort;
}): ClaudeThinkingConfig {
  if (!effort || effort === "low") {
    return { enabled: false, maxTokens };
  }

  if (usesAdaptiveThinking(model)) {
    return {
      enabled: true,
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

  const legacyEffort = effort === "xhigh" ? "high" : effort;
  const budgetTokens = THINKING_BUDGET[legacyEffort];
  return {
    enabled: true,
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

  const { systemPrompt, anthropicMessages } = toAnthropicMessages(
    messages as ChatToolMessage[],
  );

  // Anthropic rejects forced tool_choice combined with thinking. The tool
  // contract wins: drop thinking for this call rather than silently
  // downgrading a required tool call to auto.
  const thinkingConfig: ClaudeThinkingConfig = forceToolUse
    ? { enabled: false, maxTokens }
    : resolveClaudeThinkingConfig({
        model,
        maxTokens,
        effort,
      });

  // Build request parameters
  const requestParams: any = {
    model,
    max_tokens: thinkingConfig.maxTokens,
    messages: anthropicMessages,
  };

  if (hasTools) {
    requestParams.tools = toAnthropicTools(tools!);
    if (forceToolUse) {
      requestParams.tool_choice = { type: "any" };
    }
    // Tool flows need the system prompt intact — never fold it into the
    // first user turn.
    if (systemPrompt) {
      requestParams.system = systemPrompt;
    }
  } else if (systemPrompt && !thinkingConfig.enabled) {
    requestParams.system = systemPrompt;
  } else if (systemPrompt && thinkingConfig.enabled) {
    // Prepend system context to first user message when using extended thinking
    const first = anthropicMessages[0];
    if (typeof first.content === "string") {
      first.content = `${systemPrompt}\n\n${first.content}`;
    } else {
      requestParams.system = systemPrompt;
    }
  }

  if (thinkingConfig.enabled) {
    thinkingConfig.apply(requestParams);
    console.log(thinkingConfig.logMessage);
  }

  const response = await client.messages.create(requestParams, { signal });

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

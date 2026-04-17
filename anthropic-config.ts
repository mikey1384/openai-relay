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

type ClaudeEffort = "low" | "medium" | "high" | "xhigh";

// Legacy extended thinking budget tokens for Claude models that still support
// budget_tokens. Claude Opus 4.7 uses adaptive thinking instead.
const THINKING_BUDGET: Record<"low" | "medium" | "high", number> = {
  low: 0, // No extended thinking
  medium: 8000, // Moderate reasoning
  high: 16000, // Deep reasoning
};

const CLAUDE_OPUS_4_7_MODEL = "claude-opus-4-7";

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

  if (model === CLAUDE_OPUS_4_7_MODEL) {
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
}: ClaudeTranslateOptions): Promise<{
  model: string;
  choices: Array<{ message: { role: string; content: string } }>;
  usage: { prompt_tokens: number; completion_tokens: number };
}> {
  const client = makeAnthropic(apiKey);

  // Extract system message if present
  let systemPrompt: string | undefined;
  const userMessages: ClaudeMessage[] = [];

  for (const msg of messages) {
    if ((msg as any).role === "system") {
      systemPrompt = (msg as any).content;
    } else {
      userMessages.push({
        role: msg.role as "user" | "assistant",
        content: msg.content,
      });
    }
  }

  // Ensure first message is from user (Anthropic requirement)
  if (userMessages.length === 0 || userMessages[0].role !== "user") {
    userMessages.unshift({ role: "user", content: "Please proceed." });
  }

  const thinkingConfig = resolveClaudeThinkingConfig({
    model,
    maxTokens,
    effort,
  });

  // Build request parameters
  const requestParams: any = {
    model,
    max_tokens: thinkingConfig.maxTokens,
    messages: userMessages,
  };

  // Add system prompt (not compatible with extended thinking, prepend to first message)
  if (systemPrompt && !thinkingConfig.enabled) {
    requestParams.system = systemPrompt;
  } else if (systemPrompt && thinkingConfig.enabled) {
    // Prepend system context to first user message when using extended thinking
    userMessages[0].content = `${systemPrompt}\n\n${userMessages[0].content}`;
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

  return {
    model,
    choices: [
      {
        message: {
          role: "assistant",
          content: textContent,
        },
      },
    ],
    usage: {
      prompt_tokens: response.usage.input_tokens,
      completion_tokens: response.usage.output_tokens,
    },
  };
}

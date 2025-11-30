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

// Extended thinking budget tokens by effort level
const THINKING_BUDGET: Record<'low' | 'medium' | 'high', number> = {
  low: 0, // No extended thinking
  medium: 8000, // Moderate reasoning
  high: 16000, // Deep reasoning
};

export interface ClaudeTranslateOptions {
  messages: ClaudeMessage[];
  model: string;
  apiKey: string;
  signal?: AbortSignal;
  maxTokens?: number;
  effort?: 'low' | 'medium' | 'high';
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

  // Determine if extended thinking should be enabled
  const budgetTokens = effort ? THINKING_BUDGET[effort] : 0;
  const useExtendedThinking = budgetTokens > 0;

  // Build request parameters
  const requestParams: any = {
    model,
    max_tokens: useExtendedThinking ? MAX_TOKENS_WITH_THINKING : maxTokens,
    messages: userMessages,
  };

  // Add system prompt (not compatible with extended thinking, prepend to first message)
  if (systemPrompt && !useExtendedThinking) {
    requestParams.system = systemPrompt;
  } else if (systemPrompt && useExtendedThinking) {
    // Prepend system context to first user message when using extended thinking
    userMessages[0].content = `${systemPrompt}\n\n${userMessages[0].content}`;
  }

  // Add extended thinking configuration
  if (useExtendedThinking) {
    requestParams.thinking = {
      type: 'enabled',
      budget_tokens: budgetTokens,
    };
    console.log(`[anthropic-config] Extended thinking enabled with budget: ${budgetTokens} tokens`);
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

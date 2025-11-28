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

export interface ClaudeTranslateOptions {
  messages: ClaudeMessage[];
  model: string;
  apiKey: string;
  signal?: AbortSignal;
  maxTokens?: number;
}

export async function translateWithClaude({
  messages,
  model,
  apiKey,
  signal,
  maxTokens = 8192,
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

  const response = await client.messages.create(
    {
      model,
      max_tokens: maxTokens,
      system: systemPrompt,
      messages: userMessages,
    },
    { signal }
  );

  // Convert Anthropic response to OpenAI-compatible format
  const content =
    response.content[0]?.type === "text" ? response.content[0].text : "";

  return {
    choices: [
      {
        message: {
          role: "assistant",
          content,
        },
      },
    ],
    usage: {
      prompt_tokens: response.usage.input_tokens,
      completion_tokens: response.usage.output_tokens,
    },
  };
}

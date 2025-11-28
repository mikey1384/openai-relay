import OpenAI from "openai";

export function makeOpenAI(apiKey: string) {
  return new OpenAI({
    apiKey: apiKey,
    timeout: 600_000,
    maxRetries: 3,
  });
}

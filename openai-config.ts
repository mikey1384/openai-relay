import OpenAI from "openai";

export function makeOpenAI(apiKey: string) {
  return new OpenAI({
    apiKey: apiKey,
    timeout: 60_000, // 1 minute for direct calls
    maxRetries: 3,
  });
}

export function makeGroq(apiKey: string) {
  return new OpenAI({
    apiKey: apiKey,
    baseURL: "https://api.groq.com/openai/v1",
    timeout: 60_000,
    maxRetries: 3,
  });
}

import OpenAI from "openai";
import Groq from "groq-sdk";

export function makeOpenAI(apiKey: string) {
  return new OpenAI({
    apiKey: apiKey,
    timeout: 600_000, // 10 minutes for long-running requests (e.g., gpt-5)
    maxRetries: 3,
  });
}

export function makeGroq(apiKey: string) {
  return new Groq({
    apiKey,
  });
}

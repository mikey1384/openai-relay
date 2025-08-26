import OpenAI from "openai";
import Groq from "groq-sdk";

export function makeOpenAI(apiKey: string) {
  return new OpenAI({
    apiKey: apiKey,
    timeout: 60_000, // 1 minute for direct calls
    maxRetries: 3,
  });
}

export function makeGroq(apiKey: string) {
  return new Groq({
    apiKey,
  });
}

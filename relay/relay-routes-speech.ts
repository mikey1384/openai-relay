import { Buffer } from "node:buffer";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { RelayRoutesContext } from "./relay-routes.js";

export async function handleSpeechRoutes(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext
): Promise<boolean> {
  if (req.method !== "POST" || req.url !== "/speech") {
    return false;
  }

  await handleSpeech(req, res, ctx);
  return true;
}

async function handleSpeech(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext
): Promise<void> {
  console.log("🎤 Processing speech synthesis request...");

  const {
    RELAY_SECRET,
    MAX_BODY_SIZE,
    getHeader,
    sendError,
    makeOpenAI,
    sendJson,
    validateRelaySecret,
  } = ctx;

  if (!validateRelaySecret(req, RELAY_SECRET)) {
    console.log("❌ Invalid or missing relay secret for /speech");
    sendError(res, 401, "Unauthorized - invalid relay secret");
    return;
  }

  const openaiKey = getHeader(req, "x-openai-key");
  if (!openaiKey) {
    console.log("❌ Missing OpenAI API key for /speech");
    sendError(res, 401, "Unauthorized - missing OpenAI key");
    return;
  }

  let body = "";
  let bodySize = 0;
  let rejected = false;

  req.on("data", (chunk: Buffer) => {
    if (rejected) return;
    bodySize += chunk.length;
    if (bodySize > MAX_BODY_SIZE) {
      rejected = true;
      sendError(res, 413, "Request body too large");
      req.destroy();
      return;
    }
    body += chunk.toString();
  });

  req.on("error", (err) => {
    if (rejected) return;
    rejected = true;
    console.error("❌ Request stream error in /speech:", err.message);
    sendError(res, 500, "Request stream error", err.message);
  });

  req.on("end", async () => {
    if (rejected) return;
    try {
      const parsed = JSON.parse(body || "{}");
      const text = parsed.text;
      if (!text || typeof text !== "string") {
        sendError(res, 400, "Invalid request: text is required");
        return;
      }

      const voice = parsed.voice || "alloy";
      const model = parsed.model || "tts-1";
      const format = parsed.format || "mp3";
      const responseFormat = parsed.response_format;

      console.log(
        `🎶 Generating speech (${text.length} chars) model=${model} voice=${voice} format=${format}`
      );

      const client = makeOpenAI(openaiKey);
      const speech = await client.audio.speech.create({
        model,
        voice,
        input: text,
        ...(responseFormat || format
          ? { response_format: responseFormat || format }
          : {}),
      });

      const arrayBuffer = await speech.arrayBuffer();
      const buffer = Buffer.from(arrayBuffer);
      const audioBase64 = buffer.toString("base64");

      sendJson(res, {
        audioBase64,
        voice,
        model,
        format,
        length: text.length,
      });
    } catch (error: any) {
      console.error("❌ Relay speech synthesis error:", error.message || error);
      sendError(
        res,
        500,
        "Speech synthesis failed",
        error?.message || String(error)
      );
    }
  });
}

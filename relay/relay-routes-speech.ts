import { Buffer } from "node:buffer";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { RelayRoutesContext } from "./relay-routes.js";
import {
  confirmRelayReservation,
  finalizeRelayCredits,
  releaseRelayCredits,
} from "./relay-billing-client.js";
import { STAGE5_RELAY_BILLING_SERVICES } from "./relay-billing-contract.js";
import { getInternalRelayBillingContext } from "./relay-billing-helpers.js";

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
    CF_API_BASE,
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

  const internalBilling = getInternalRelayBillingContext(req, getHeader);
  if (!internalBilling) {
    sendError(res, 401, "Unauthorized - missing Stage5 billing context");
    return;
  }
  const confirmResult = await confirmRelayReservation({
    cfApiBase: CF_API_BASE,
    relaySecret: RELAY_SECRET,
    payload: {
      deviceId: internalBilling.deviceId,
      requestKey: internalBilling.requestKey,
      service: STAGE5_RELAY_BILLING_SERVICES.TTS,
    },
  });
  if (!confirmResult.ok) {
    sendError(res, confirmResult.status, confirmResult.error || "Reservation confirmation failed");
    return;
  }
  let reservationOutcome: "pending" | "settled" | "released" = "pending";
  const releaseReservation = async (meta?: unknown): Promise<void> => {
    if (reservationOutcome !== "pending") return;
    reservationOutcome = "released";
    const releaseResult = await releaseRelayCredits({
      cfApiBase: CF_API_BASE,
      relaySecret: RELAY_SECRET,
      payload: {
        deviceId: internalBilling.deviceId,
        requestKey: internalBilling.requestKey,
        service: STAGE5_RELAY_BILLING_SERVICES.TTS,
        ...(typeof meta === "undefined" ? {} : { meta }),
      },
    });
    if (!releaseResult.ok) {
      console.error(
        "❌ Failed to release /speech reservation:",
        releaseResult.error,
      );
    }
  };
  const finalizeReservation = async ({
    characters,
    model,
  }: {
    characters: number;
    model: string;
  }): Promise<boolean> => {
    if (reservationOutcome !== "pending") return true;
    const finalizeResult = await finalizeRelayCredits({
      cfApiBase: CF_API_BASE,
      relaySecret: RELAY_SECRET,
      payload: {
        deviceId: internalBilling.deviceId,
        requestKey: internalBilling.requestKey,
        service: STAGE5_RELAY_BILLING_SERVICES.TTS,
        characters,
        model,
      },
    });
    if (!finalizeResult.ok) {
      console.error("❌ Failed to finalize /speech reservation:", finalizeResult.error);
      await releaseReservation({
        reason: "speech-finalize-failed",
        error: finalizeResult.error,
      });
      return false;
    }
    reservationOutcome = "settled";
    return true;
  };

  const openaiKey = getHeader(req, "x-openai-key");
  if (!openaiKey) {
    console.log("❌ Missing OpenAI API key for /speech");
    await releaseReservation({ reason: "missing-openai-key" });
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
      void releaseReservation({ reason: "body-too-large" });
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
    void releaseReservation({ reason: "request-stream-error", error: err.message });
    sendError(res, 500, "Request stream error", err.message);
  });

  req.on("end", async () => {
    if (rejected) return;
    try {
      const parsed = JSON.parse(body || "{}");
      const text = parsed.text;
      if (!text || typeof text !== "string") {
        await releaseReservation({ reason: "invalid-text" });
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
      const finalized = await finalizeReservation({
        characters: text.length,
        model,
      });
      if (!finalized) {
        sendError(res, 500, "Speech billing finalization failed");
        return;
      }

      sendJson(res, {
        audioBase64,
        voice,
        model,
        format,
        length: text.length,
      });
    } catch (error: any) {
      console.error("❌ Relay speech synthesis error:", error.message || error);
      await releaseReservation({
        reason: "speech-synthesis-error",
        error: error?.message || String(error),
      });
      sendError(
        res,
        500,
        "Speech synthesis failed",
        error?.message || String(error)
      );
    }
  });
}

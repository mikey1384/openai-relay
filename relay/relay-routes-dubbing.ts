import { Buffer } from "node:buffer";
import type { IncomingMessage, ServerResponse } from "node:http";
import { IncomingForm } from "formidable";
import type { RelayRoutesContext } from "./relay-routes.js";
import {
  authorizeRelayDevice,
  deductRelayCredits,
} from "./relay-billing-client.js";
import { STAGE5_RELAY_BILLING_SERVICES } from "./relay-billing-contract.js";

export async function handleDubbingRoutes(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext
): Promise<boolean> {
  if (req.method === "POST" && req.url === "/dub-direct") {
    await handleDubDirect(req, res, ctx);
    return true;
  }

  if (req.method === "POST" && req.url === "/dub-elevenlabs") {
    await handleDubElevenLabs(req, res, ctx);
    return true;
  }

  if (req.method === "POST" && req.url === "/dub-video-elevenlabs") {
    await handleDubVideoElevenLabs(req, res, ctx);
    return true;
  }

  if (req.method === "POST" && req.url === "/dub") {
    await handleDub(req, res, ctx);
    return true;
  }

  return false;
}

// ElevenLabs voice name to ID mapping
const ELEVENLABS_VOICE_IDS: Record<string, string> = {
  rachel: "21m00Tcm4TlvDq8ikWAM",
  adam: "pNInz6obpgDQGcFmaJgB",
  josh: "TxGEqnHWrfWFTfGW9XjX",
  sarah: "EXAVITQu4vr4xnSDxMaL",
  charlie: "IKne3meq5aSn9XLyUdCD",
  emily: "LcfcDJNUP1GQjkzn1xUU",
  matilda: "XrExE9yKIg1WjnnlVkGX",
  brian: "nPczCjzI2devNBz1zQrb",
};

async function handleDubDirect(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext
): Promise<void> {
  console.log("📡 Processing direct dub request...");

  const { RELAY_SECRET, getHeader, sendError, sendJson, readJsonBody, makeOpenAI } =
    ctx;

  // Get API key from header (app sends its Stage5 API key)
  const apiKey = getHeader(req, "authorization")?.replace(/^Bearer\s+/i, "");
  if (!apiKey) {
    console.log("❌ Missing API key for /dub-direct");
    sendError(res, 401, "Unauthorized - missing API key");
    return;
  }

  const idempotencyKey =
    getHeader(req, "idempotency-key") || getHeader(req, "x-idempotency-key");

  // Step 1: Authorize with CF Worker
  const { CF_API_BASE } = ctx;
  console.log(`🔐 Authorizing with CF Worker...`);

  const authResult = await authorizeRelayDevice({
    cfApiBase: CF_API_BASE,
    relaySecret: RELAY_SECRET,
    apiKey,
  });
  if (!authResult.ok) {
    console.log(`❌ Authorization failed: ${authResult.status}`);
    sendError(res, authResult.status, authResult.error || "Authorization failed");
    return;
  }
  const deviceId = authResult.deviceId;
  console.log(`✅ Authorized device ${deviceId}, balance: ${authResult.creditBalance}`);

  // Step 2: Parse request and synthesize speech
  try {
    const parsed = await readJsonBody(req);
    const segments = parsed?.segments || [];
    const voice = parsed?.voice || "alloy";
    const model = parsed?.model || "tts-1";
    const format = parsed?.format || "mp3";
    const ttsProvider = parsed?.ttsProvider || "openai";

    if (!Array.isArray(segments) || segments.length === 0) {
      sendError(res, 400, "Segments array required");
      return;
    }

    // Calculate total characters for billing
    const totalCharacters = segments.reduce(
      (sum: number, seg: any) => sum + (seg.text?.length || seg.translation?.length || 0),
      0
    );

    console.log(
      `🎧 Synthesizing ${segments.length} segments (${totalCharacters} chars) with ${ttsProvider}...`
    );

    let result: any;
    let ttsModel = model;

    if (ttsProvider === "elevenlabs") {
      // Use ElevenLabs
      const elevenLabsKey = process.env.ELEVENLABS_API_KEY;
      if (!elevenLabsKey) {
        sendError(res, 500, "ElevenLabs not configured");
        return;
      }

      // Map voice name to ElevenLabs voice ID
      const voiceId = ELEVENLABS_VOICE_IDS[voice] || ELEVENLABS_VOICE_IDS.rachel;

      ttsModel = "eleven_multilingual_v2";
      const segmentResults: Array<{
        index: number;
        audioBase64: string;
        targetDuration?: number;
      }> = [];

      for (const seg of segments) {
        const text = seg.text || seg.translation || "";
        if (!text.trim()) continue;

        const elevenRes = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
          method: "POST",
          headers: {
            "xi-api-key": elevenLabsKey,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text,
            model_id: "eleven_multilingual_v2",
            voice_settings: {
              stability: 0.5,
              similarity_boost: 0.75,
            },
          }),
        });

        if (!elevenRes.ok) {
          const errText = await elevenRes.text();
          throw new Error(`ElevenLabs API error: ${elevenRes.status} ${errText}`);
        }

        const audioBuffer = await elevenRes.arrayBuffer();
        segmentResults.push({
          index: seg.index ?? segmentResults.length,
          audioBase64: Buffer.from(audioBuffer).toString("base64"),
          targetDuration: seg.targetDuration,
        });
      }

      result = {
        segments: segmentResults,
        format: "mp3",
        voice,
        model: ttsModel,
        segmentCount: segmentResults.length,
      };
    } else {
      // Use OpenAI TTS
      console.log("   >>> Entering OpenAI TTS branch");
      const openaiKey = process.env.OPENAI_API_KEY;
      console.log("   >>> OpenAI key exists:", !!openaiKey);
      if (!openaiKey) {
        sendError(res, 500, "OpenAI not configured");
        return;
      }

      const client = makeOpenAI(openaiKey);
      const segmentResults: Array<{
        index: number;
        audioBase64: string;
        targetDuration?: number;
      }> = [];

      console.log(`   DEBUG: About to process ${segments.length} segments for OpenAI TTS`);
      console.log(`   DEBUG: First segment:`, JSON.stringify(segments[0]));

      for (const seg of segments) {
        const text = seg.text || seg.translation || "";
        if (!text.trim()) continue;

        console.log(
          `   • OpenAI TTS: voice=${voice}, model=${model}, format=${format}, text="${text.slice(0, 30)}..."`
        );

        try {
          const ttsRes = await client.audio.speech.create({
            model,
            voice: voice as any,
            input: text,
            response_format: format as any,
          });

          const audioBuffer = await ttsRes.arrayBuffer();
          segmentResults.push({
            index: seg.index ?? segmentResults.length,
            audioBase64: Buffer.from(audioBuffer).toString("base64"),
            targetDuration: seg.targetDuration,
          });
          console.log(`   ✓ OpenAI TTS segment complete`);
        } catch (ttsErr: any) {
          console.error(
            `❌ OpenAI TTS error: ${ttsErr?.message || ttsErr}`,
            ttsErr?.response?.data || ""
          );
          throw ttsErr;
        }
      }

      result = {
        segments: segmentResults,
        format,
        voice,
        model,
        segmentCount: segmentResults.length,
      };
    }

    console.log(`🎯 TTS complete! ${result.segmentCount} segments`);

    // Step 3: Deduct credits
    console.log(`💳 Deducting credits for ${totalCharacters} characters...`);
    try {
      const deductResult = await deductRelayCredits({
        cfApiBase: CF_API_BASE,
        relaySecret: RELAY_SECRET,
        payload: {
          deviceId,
          service: STAGE5_RELAY_BILLING_SERVICES.TTS,
          characters: totalCharacters,
          model: ttsModel,
          ...(idempotencyKey ? { idempotencyKey } : {}),
        },
      });

      if (!deductResult.ok) {
        console.error(`❌ Credit deduction failed: ${deductResult.status}`);
        const status = deductResult.status === 402 ? 402 : 500;
        sendError(
          res,
          status,
          deductResult.error || "Credit deduction failed"
        );
        return;
      }

      console.log(`✅ Credits deducted successfully`);
    } catch (deductErr: any) {
      console.error("❌ Credit deduction request failed:", deductErr.message);
      sendError(res, 500, "Credit deduction failed", deductErr?.message);
      return;
    }

    // Return result to app (only after successful deduction)
    sendJson(res, result);
  } catch (error: any) {
    console.error("❌ Dub error:", error.message);
    sendError(res, 500, "Dub synthesis failed", error.message);
  }
}

async function handleDubElevenLabs(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext
): Promise<void> {
  console.log("🎬 Processing ElevenLabs TTS dub request...");

  const {
    RELAY_SECRET,
    getHeader,
    sendError,
    sendJson,
    validateRelaySecret,
    readJsonBody,
    synthesizeWithElevenLabs,
  } = ctx;

  if (!validateRelaySecret(req, RELAY_SECRET)) {
    console.log("❌ Invalid or missing relay secret for /dub-elevenlabs");
    sendError(res, 401, "Unauthorized - invalid relay secret");
    return;
  }

  const elevenLabsKey =
    getHeader(req, "x-elevenlabs-key") || process.env.ELEVENLABS_API_KEY;
  if (!elevenLabsKey) {
    console.log("❌ ElevenLabs API key not provided");
    sendError(res, 500, "ElevenLabs not configured");
    return;
  }

  try {
    const parsed = await readJsonBody(req);
    const segmentsPayload = Array.isArray(parsed?.segments)
      ? parsed.segments
          .map((segment: any, idx: number) => {
            const text = typeof segment?.text === "string" ? segment.text.trim() : "";
            if (!text) return null;
            const index = Number.isFinite(segment?.index)
              ? Number(segment.index)
              : idx + 1;
            const targetDuration =
              typeof segment?.targetDuration === "number" &&
              Number.isFinite(segment.targetDuration)
                ? segment.targetDuration
                : undefined;
            return { index, text, targetDuration };
          })
          .filter(Boolean)
      : [];

    if (!segmentsPayload.length) {
      sendError(res, 400, "No valid segments provided");
      return;
    }

    const voice = parsed?.voice || "adam";
    const totalCharacters = segmentsPayload.reduce(
      (sum: number, seg: any) => sum + seg.text.length,
      0
    );

    console.log(
      `🎧 Synthesizing ${segmentsPayload.length} segments (${totalCharacters} chars) with ElevenLabs voice=${voice}`
    );

    const segmentResponses: Array<{
      index: number;
      audioBase64: string;
      targetDuration?: number;
    }> = [];

    // Process segments with concurrency limit
    const CONCURRENCY = 3;
    for (let i = 0; i < segmentsPayload.length; i += CONCURRENCY) {
      const batch = segmentsPayload.slice(i, i + CONCURRENCY);
      const results = await Promise.all(
        batch.map(async (seg: any) => {
          const audioBuffer = await synthesizeWithElevenLabs({
            text: seg.text,
            voice,
            apiKey: elevenLabsKey,
          });
          return {
            index: seg.index,
            audioBase64: audioBuffer.toString("base64"),
            targetDuration: seg.targetDuration,
          };
        })
      );
      segmentResponses.push(...results);
      console.log(
        `   • Completed ${Math.min(i + CONCURRENCY, segmentsPayload.length)}/${segmentsPayload.length} segments`
      );
    }

    sendJson(res, {
      voice,
      model: "eleven_multilingual_v2",
      format: "mp3",
      segmentCount: segmentResponses.length,
      totalCharacters,
      segments: segmentResponses,
    });
  } catch (error: any) {
    console.error("❌ ElevenLabs TTS error:", error.message);
    sendError(res, 500, "Dub synthesis failed", error.message);
  }
}

async function handleDubVideoElevenLabs(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext
): Promise<void> {
  console.log("🎬 Processing ElevenLabs video dubbing request...");

  const {
    RELAY_SECRET,
    getHeader,
    sendError,
    sendJson,
    validateRelaySecret,
    dubWithElevenLabs,
  } = ctx;

  if (!validateRelaySecret(req, RELAY_SECRET)) {
    console.log("❌ Invalid or missing relay secret for /dub-video-elevenlabs");
    sendError(res, 401, "Unauthorized - invalid relay secret");
    return;
  }

  const elevenLabsKey =
    getHeader(req, "x-elevenlabs-key") || process.env.ELEVENLABS_API_KEY;
  if (!elevenLabsKey) {
    console.log("❌ ElevenLabs API key not provided");
    sendError(res, 500, "ElevenLabs not configured");
    return;
  }

  try {
    const form = new IncomingForm({
      maxFileSize: 500 * 1024 * 1024, // 500MB for video files
    });
    const [fields, files] = await form.parse(req);

    const file = Array.isArray(files.file) ? files.file[0] : files.file;
    if (!file) {
      console.log("❌ No file provided for dubbing");
      sendError(res, 400, "No file provided");
      return;
    }

    const targetLanguage = Array.isArray(fields.target_language)
      ? fields.target_language[0]
      : fields.target_language;
    if (!targetLanguage) {
      console.log("❌ No target language provided");
      sendError(res, 400, "target_language is required");
      return;
    }

    const sourceLanguage = Array.isArray(fields.source_language)
      ? fields.source_language[0]
      : fields.source_language;
    const numSpeakersField = Array.isArray(fields.num_speakers)
      ? fields.num_speakers[0]
      : fields.num_speakers;
    // Safe parseInt with Number.isFinite validation
    const numSpeakersRaw = numSpeakersField ? parseInt(numSpeakersField, 10) : undefined;
    const numSpeakers =
      numSpeakersRaw !== undefined &&
      Number.isFinite(numSpeakersRaw) &&
      numSpeakersRaw > 0
        ? numSpeakersRaw
        : undefined;
    const dropBackgroundAudioField = Array.isArray(fields.drop_background_audio)
      ? fields.drop_background_audio[0]
      : fields.drop_background_audio;
    const dropBackgroundAudio = dropBackgroundAudioField !== "false";

    console.log(
      `🎬 Dubbing video: ${file.originalFilename} (${(
        file.size /
        1024 /
        1024
      ).toFixed(1)}MB) → ${targetLanguage}`
    );

    const fs = await import("fs");
    const fileBuffer = await fs.promises.readFile(file.filepath);

    const result = await dubWithElevenLabs({
      fileBuffer,
      fileName: file.originalFilename || "video.mp4",
      mimeType: file.mimetype || "video/mp4",
      sourceLanguage: sourceLanguage || undefined,
      targetLanguage,
      apiKey: elevenLabsKey,
      numSpeakers,
      dropBackgroundAudio,
      onProgress: (status: string) => console.log(`   • ${status}`),
    });

    console.log(`🎯 ElevenLabs video dubbing completed!`);
    sendJson(res, result);
  } catch (error: any) {
    console.error("❌ ElevenLabs video dubbing error:", error.message);
    sendError(res, 500, "Video dubbing failed", error.message);
  }
}

async function handleDub(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext
): Promise<void> {
  console.log("🎬 Processing dub synthesis request...");

  const {
    RELAY_SECRET,
    DUB_MAX_SEGMENTS,
    DUB_MAX_TOTAL_CHARACTERS,
    DUB_MAX_RETRIES,
    DUB_RETRY_BASE_DELAY_MS,
    DUB_RETRY_MAX_DELAY_MS,
    DUB_MAX_CONCURRENCY,
    MAX_TTS_CHARS_PER_CHUNK,
    getHeader,
    sendError,
    sendJson,
    validateRelaySecret,
    readJsonBody,
    makeOpenAI,
    chunkLines,
    shouldRetrySegmentError,
    sleep,
  } = ctx;

  if (!validateRelaySecret(req, RELAY_SECRET)) {
    console.log("❌ Invalid or missing relay secret for /dub");
    sendError(res, 401, "Unauthorized - invalid relay secret");
    return;
  }

  const openaiKey = getHeader(req, "x-openai-key");
  if (!openaiKey) {
    console.log("❌ Missing OpenAI API key for /dub");
    sendError(res, 401, "Unauthorized - missing OpenAI key");
    return;
  }

  try {
    const parsed = await readJsonBody(req);
    const segmentsPayload = Array.isArray(parsed?.segments)
      ? parsed.segments
          .map((segment: any, idx: number) => {
            const rawText = typeof segment?.text === "string" ? segment.text : "";
            const text = rawText.trim();
            if (!text) {
              return null;
            }
            const index = Number.isFinite(segment?.index)
              ? Number(segment.index)
              : idx + 1;
            const start =
              typeof segment?.start === "number" && Number.isFinite(segment.start)
                ? segment.start
                : undefined;
            const end =
              typeof segment?.end === "number" && Number.isFinite(segment.end)
                ? segment.end
                : undefined;
            const targetDuration =
              typeof segment?.targetDuration === "number" &&
              Number.isFinite(segment.targetDuration)
                ? segment.targetDuration
                : typeof start === "number" && typeof end === "number"
                  ? Math.max(0, end - start)
                  : undefined;
            return {
              index,
              text,
              start,
              end,
              targetDuration,
            };
          })
          .filter(
            (
              seg: any
            ): seg is {
              index: number;
              text: string;
              start?: number;
              end?: number;
              targetDuration?: number;
            } => Boolean(seg)
          )
      : [];

    const lines = Array.isArray(parsed?.lines)
      ? parsed.lines.map((line: any) => String(line ?? "").trim())
      : [];
    const voice =
      typeof parsed?.voice === "string" && parsed.voice.trim()
        ? parsed.voice.trim()
        : "alloy";
    const model =
      typeof parsed?.model === "string" && parsed.model.trim()
        ? parsed.model.trim()
        : "tts-1";
    const format =
      typeof parsed?.format === "string" && parsed.format.trim()
        ? parsed.format.trim()
        : "mp3";

    const client = makeOpenAI(openaiKey);

    if (segmentsPayload.length > 0) {
      if (segmentsPayload.length > DUB_MAX_SEGMENTS) {
        sendError(
          res,
          413,
          "Dub request too large",
          `Received ${segmentsPayload.length} segments, limit is ${DUB_MAX_SEGMENTS}`
        );
        return;
      }

      const totalCharacters = segmentsPayload.reduce(
        (sum: number, seg: { text: string }) => sum + seg.text.length,
        0
      );

      if (totalCharacters > DUB_MAX_TOTAL_CHARACTERS) {
        sendError(
          res,
          413,
          "Dub request too large",
          `Received ${totalCharacters} characters, limit is ${DUB_MAX_TOTAL_CHARACTERS}`
        );
        return;
      }

      console.log(
        `🎧 Synthesizing ${segmentsPayload.length} segment(s) (${totalCharacters} chars) model=${model} voice=${voice} format=${format}`
      );

      const segmentResponses: Array<{
        index: number;
        audioBase64: string;
        targetDuration?: number;
      }> = new Array(segmentsPayload.length);

      let requestClosed = false;
      const segmentAbortControllers = new Map<number, AbortController>();

      req.on("close", () => {
        requestClosed = true;
        for (const controller of segmentAbortControllers.values()) {
          controller.abort();
        }
      });

      const synthesizeSegment = async (segIdx: number) => {
        const seg = segmentsPayload[segIdx];
        let attempt = 0;
        const abortController = new AbortController();
        segmentAbortControllers.set(segIdx, abortController);

        try {
          while (true) {
            if (requestClosed) {
              throw new Error("Client disconnected");
            }

            attempt += 1;
            try {
              const speech = await client.audio.speech.create(
                {
                  model,
                  voice,
                  input: seg.text,
                  response_format: format,
                },
                { signal: abortController.signal }
              );
              const arrayBuffer = await speech.arrayBuffer();
              segmentResponses[segIdx] = {
                index: seg.index,
                audioBase64: Buffer.from(arrayBuffer).toString("base64"),
                targetDuration: seg.targetDuration,
              };
              return;
            } catch (segmentError: any) {
              if (requestClosed || abortController.signal.aborted) {
                throw segmentError;
              }

              if (attempt >= DUB_MAX_RETRIES || !shouldRetrySegmentError(segmentError)) {
                throw segmentError;
              }

              const delay = Math.min(
                DUB_RETRY_MAX_DELAY_MS,
                DUB_RETRY_BASE_DELAY_MS * Math.pow(2, attempt - 1)
              );
              console.warn(
                `⚠️ Segment ${segIdx + 1}/${segmentsPayload.length} retry ${attempt}/${DUB_MAX_RETRIES} in ${delay}ms:`,
                segmentError?.message || segmentError
              );
              await sleep(delay);
            }
          }
        } finally {
          segmentAbortControllers.delete(segIdx);
        }
      };

      // Use a queue to distribute work safely across workers
      const pendingIndices = segmentsPayload.map((_: any, i: number) => i);
      const workerCount = Math.min(DUB_MAX_CONCURRENCY, segmentsPayload.length);

      const workers = Array.from({ length: workerCount }, async (_, workerIdx) => {
        while (true) {
          if (requestClosed) {
            return;
          }

          const current = pendingIndices.shift();
          if (current === undefined) {
            return;
          }

          const seg = segmentsPayload[current];
          console.log(
            `   • Worker ${workerIdx + 1}/${workerCount} segment ${
              current + 1
            }/${segmentsPayload.length} (index=${seg.index}, ${seg.text.length} chars)`
          );
          await synthesizeSegment(current);
          console.log(`     · Completed segment ${current + 1}/${segmentsPayload.length}`);
        }
      });

      try {
        await Promise.all(workers);
      } catch (segmentError: any) {
        if (requestClosed) {
          console.warn("⚠️ Dub request aborted by upstream client while synthesizing segments");
          return;
        }

        const details = segmentError?.response?.data ?? segmentError?.message;
        console.error("❌ Relay segment synthesis failed:", details);
        sendError(
          res,
          500,
          "Dub synthesis failed",
          typeof details === "string" ? details : JSON.stringify(details)
        );
        return;
      }

      if (requestClosed) {
        console.warn("⚠️ Dub request closed before completion; skipping response");
        return;
      }

      const completedSegments = segmentResponses.filter(Boolean);

      sendJson(res, {
        voice,
        model,
        format,
        segmentCount: completedSegments.length,
        totalCharacters,
        segments: completedSegments,
      });
      return;
    }

    if (!lines.length) {
      sendError(res, 400, "Invalid request: lines array required");
      return;
    }

    const totalCharacters = lines.reduce((sum: number, line: string) => sum + line.length, 0);
    const chunks = chunkLines(lines, MAX_TTS_CHARS_PER_CHUNK);

    if (!chunks.length) {
      sendError(res, 400, "No valid dialogue for dubbing");
      return;
    }

    console.log(
      `🎧 Synthesizing ${chunks.length} chunk(s) (${totalCharacters} chars) model=${model} voice=${voice} format=${format}`
    );

    const chunkBuffers: Buffer[] = [];

    for (let idx = 0; idx < chunks.length; idx++) {
      const chunk = chunks[idx];
      console.log(`   • Chunk ${idx + 1}/${chunks.length} (${chunk.length} chars)`);
      const speech = await client.audio.speech.create({
        model,
        voice,
        input: chunk,
        response_format: format,
      });
      const arrayBuffer = await speech.arrayBuffer();
      chunkBuffers.push(Buffer.from(arrayBuffer));
    }

    const combined = Buffer.concat(chunkBuffers);
    const audioBase64 = combined.toString("base64");

    sendJson(res, {
      audioBase64,
      voice,
      model,
      format,
      chunkCount: chunks.length,
      totalCharacters,
    });
  } catch (error: any) {
    console.error("❌ Relay dub synthesis error:", error?.message || error);
    sendError(res, 500, "Dub synthesis failed", error?.message || String(error));
  }
}

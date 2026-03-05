import { Buffer } from "node:buffer";
import type { IncomingMessage, ServerResponse } from "node:http";
import { IncomingForm } from "formidable";
import type { RelayRoutesContext } from "./relay-routes.js";
import {
  authorizeRelayDevice,
  deductRelayCredits,
} from "./relay-billing-client.js";
import { STAGE5_RELAY_BILLING_SERVICES } from "./relay-billing-contract.js";

export async function handleTranscriptionRoutes(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext,
): Promise<boolean> {
  if (req.method === "POST" && req.url === "/transcribe") {
    await handleTranscribe(req, res, ctx);
    return true;
  }

  if (req.method === "POST" && req.url === "/transcribe-elevenlabs") {
    await handleTranscribeElevenLabs(req, res, ctx);
    return true;
  }

  if (req.method === "POST" && req.url === "/transcribe-direct") {
    await handleTranscribeDirect(req, res, ctx);
    return true;
  }

  if (req.method === "POST" && req.url === "/transcribe-from-r2") {
    await handleTranscribeFromR2(req, res, ctx);
    return true;
  }

  return false;
}

async function handleTranscribe(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext,
): Promise<void> {
  console.log("📡 Processing transcribe request...");

  const {
    CF_API_BASE,
    RELAY_SECRET,
    ELEVENLABS_TRANSCRIPTION_MODEL,
    WHISPER_TRANSCRIPTION_MODEL,
    SCRIBE_MAX_RETRIES,
    getHeader,
    sendError,
    sendJson,
    validateRelaySecret,
    transcribeWithScribeWithRetries,
    transcribeWithWhisperFromPath,
    resolveDirectTranscriptionQuality,
    getWhisperFileSizeGuardMessage,
    toWhisperCompatibleScribeResult,
  } = ctx;

  if (!validateRelaySecret(req, RELAY_SECRET)) {
    console.log("❌ Invalid or missing relay secret");
    sendError(res, 401, "Unauthorized - invalid relay secret");
    return;
  }

  const idempotencyKey =
    getHeader(req, "idempotency-key") || getHeader(req, "x-idempotency-key");

  console.log("🎯 Relay secret validated, processing transcription...");

  try {
    const form = new IncomingForm({
      maxFileSize: 500 * 1024 * 1024, // 500MB for long audio files
    });
    const [fields, files] = await form.parse(req);

    const file = Array.isArray(files.file) ? files.file[0] : files.file;
    if (!file) {
      console.log("❌ No file provided");
      sendError(res, 400, "No file provided");
      return;
    }
    const whisperSizeGuardMessage = getWhisperFileSizeGuardMessage(file.size);

    const modelHint = Array.isArray(fields.model)
      ? fields.model[0]
      : fields.model;
    const modelIdHint = Array.isArray(fields.model_id)
      ? fields.model_id[0]
      : fields.model_id;
    const qualityModeRaw =
      (Array.isArray(fields.qualityMode)
        ? fields.qualityMode[0]
        : fields.qualityMode) ??
      (Array.isArray(fields.quality_mode)
        ? fields.quality_mode[0]
        : fields.quality_mode);
    const language = Array.isArray(fields.language)
      ? fields.language[0]
      : fields.language;
    const prompt = Array.isArray(fields.prompt)
      ? fields.prompt[0]
      : fields.prompt;
    const { useHighQuality, source: qualitySource } =
      resolveDirectTranscriptionQuality({
        explicitQualityRaw: qualityModeRaw,
        modelHint: typeof modelHint === "string" ? modelHint : undefined,
        modelIdHint: typeof modelIdHint === "string" ? modelIdHint : undefined,
      });

    console.log(
      `🎵 /transcribe selected ${
        useHighQuality ? "elevenlabs" : "whisper"
      } (qualitySource=${qualitySource}) for ${file.originalFilename} (${(
        file.size /
        1024 /
        1024
      ).toFixed(1)}MB)`,
    );

    const openaiKey = getHeader(req, "x-openai-key");
    const elevenLabsKey =
      getHeader(req, "x-elevenlabs-key") || process.env.ELEVENLABS_API_KEY;

    let effectiveHighQuality = useHighQuality;
    if (effectiveHighQuality && !elevenLabsKey && openaiKey) {
      effectiveHighQuality = false;
      console.warn(
        "⚠️ ElevenLabs key missing for high-quality /transcribe; falling back to Whisper.",
      );
    }

    if (effectiveHighQuality) {
      if (!elevenLabsKey) {
        sendError(res, 500, "ElevenLabs not configured");
        return;
      }

      try {
        const { result: scribeResult, attempts } =
          await transcribeWithScribeWithRetries({
            filePath: file.filepath,
            apiKey: elevenLabsKey,
            languageCode: language || "auto",
            idempotencyKey,
            contextLabel: "/transcribe",
          });
        const whisperFormat = toWhisperCompatibleScribeResult(scribeResult);
        if (attempts > 1) {
          (whisperFormat as any).retry = {
            provider: ELEVENLABS_TRANSCRIPTION_MODEL,
            attempts,
          };
        }

        console.log("🎯 Relay transcription completed with ElevenLabs.");
        sendJson(res, whisperFormat);
      } catch (scribeError: any) {
        if (!openaiKey) {
          throw scribeError;
        }

        const attempts =
          Number((scribeError as any)?.scribeAttempts) || SCRIBE_MAX_RETRIES;
        if (whisperSizeGuardMessage) {
          const reason = scribeError?.message || String(scribeError);
          console.warn(
            `⚠️ /transcribe cannot fall back to Whisper after ${attempts} ElevenLabs attempts: ${whisperSizeGuardMessage}`,
          );
          sendError(
            res,
            502,
            "ElevenLabs transcription failed and Whisper fallback is unavailable for this file size",
            `${reason}. ${whisperSizeGuardMessage}`,
          );
          return;
        }
        console.warn(
          `⚠️ /transcribe falling back to Whisper after ${attempts} ElevenLabs attempts: ${
            scribeError?.message || String(scribeError)
          }`,
        );
        const transcription = await transcribeWithWhisperFromPath({
          openaiKey,
          filePath: file.filepath,
          fileName: file.originalFilename || "audio.webm",
          mimeType: file.mimetype || "audio/webm",
          language: language || undefined,
          prompt: prompt || undefined,
        });
        sendJson(res, {
          ...transcription,
          fallback: {
            from: ELEVENLABS_TRANSCRIPTION_MODEL,
            to: WHISPER_TRANSCRIPTION_MODEL,
            attempts,
            reason: scribeError?.message || String(scribeError),
          },
        });
      }
    } else {
      if (!openaiKey) {
        sendError(res, 401, "Unauthorized - missing OpenAI key");
        return;
      }
      if (whisperSizeGuardMessage) {
        sendError(
          res,
          413,
          "File too large for Whisper transcription",
          whisperSizeGuardMessage,
        );
        return;
      }

      const transcription = await transcribeWithWhisperFromPath({
        openaiKey,
        filePath: file.filepath,
        fileName: file.originalFilename || "audio.webm",
        mimeType: file.mimetype || "audio/webm",
        language: language || undefined,
        prompt: prompt || undefined,
      });

      console.log("🎯 Relay transcription completed with Whisper.");
      sendJson(res, transcription);
    }
  } catch (error: any) {
    console.error("❌ Relay transcription error:", error.message);
    sendError(res, 500, "Transcription failed", error.message);
  }
}

// TODO(stage5-cleanup): Remove this legacy endpoint after all supported clients
// use /transcribe (worker path) or /transcribe-direct (app -> relay path).
async function handleTranscribeElevenLabs(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext,
): Promise<void> {
  console.log("📡 Processing ElevenLabs Scribe transcription request...");

  const {
    RELAY_SECRET,
    getHeader,
    sendError,
    sendJson,
    validateRelaySecret,
    transcribeWithScribe,
    getWhisperFileSizeGuardMessage,
    toWhisperCompatibleScribeResult,
  } = ctx;

  if (!validateRelaySecret(req, RELAY_SECRET)) {
    console.log("❌ Invalid or missing relay secret");
    sendError(res, 401, "Unauthorized - invalid relay secret");
    return;
  }

  const idempotencyKey =
    getHeader(req, "idempotency-key") || getHeader(req, "x-idempotency-key");

  const elevenLabsKey =
    getHeader(req, "x-elevenlabs-key") || process.env.ELEVENLABS_API_KEY;
  if (!elevenLabsKey) {
    console.log("❌ ElevenLabs API key not provided");
    sendError(res, 500, "ElevenLabs not configured");
    return;
  }

  try {
    const form = new IncomingForm({
      maxFileSize: 500 * 1024 * 1024, // 500MB for long audio files
    });
    const [fields, files] = await form.parse(req);

    const file = Array.isArray(files.file) ? files.file[0] : files.file;
    if (!file) {
      console.log("❌ No file provided");
      sendError(res, 400, "No file provided");
      return;
    }
    const whisperSizeGuardMessage = getWhisperFileSizeGuardMessage(file.size);

    const language = Array.isArray(fields.language)
      ? fields.language[0]
      : fields.language;

    console.log(
      `🎵 Transcribing with ElevenLabs Scribe: ${file.originalFilename} (${(
        file.size /
        1024 /
        1024
      ).toFixed(1)}MB)`,
    );

    const result = await transcribeWithScribe({
      filePath: file.filepath,
      apiKey: elevenLabsKey,
      languageCode: language || "auto",
      idempotencyKey,
    });
    const whisperFormat = toWhisperCompatibleScribeResult(result);

    console.log(`🎯 ElevenLabs Scribe transcription completed!`);
    sendJson(res, whisperFormat);
  } catch (error: any) {
    console.error("❌ ElevenLabs Scribe error:", error.message);
    sendError(res, 500, "Transcription failed", error.message);
  }
}

async function handleTranscribeDirect(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext,
): Promise<void> {
  console.log("📡 Processing direct transcription request...");

  const {
    CF_API_BASE,
    RELAY_SECRET,
    ELEVENLABS_TRANSCRIPTION_MODEL,
    WHISPER_TRANSCRIPTION_MODEL,
    SCRIBE_MAX_RETRIES,
    getHeader,
    sendError,
    sendJson,
    transcribeWithScribeWithRetries,
    transcribeWithWhisperFromPath,
    resolveDirectTranscriptionQuality,
    getWhisperFileSizeGuardMessage,
    toWhisperCompatibleScribeResult,
  } = ctx;

  // Get API key from header (app sends its Stage5 API key)
  const apiKey = getHeader(req, "authorization")?.replace(/^Bearer\s+/i, "");
  if (!apiKey) {
    console.log("❌ Missing API key for /transcribe-direct");
    sendError(res, 401, "Unauthorized - missing API key");
    return;
  }

  // Stable idempotency key from the app to prevent double billing on retries.
  const idempotencyKey =
    getHeader(req, "idempotency-key") || getHeader(req, "x-idempotency-key");

  // Step 1: Authorize with CF Worker
  console.log(`🔐 Authorizing with CF Worker...`);

  const authResult = await authorizeRelayDevice({
    cfApiBase: CF_API_BASE,
    relaySecret: RELAY_SECRET,
    apiKey,
  });
  if (!authResult.ok) {
    console.log(`❌ Authorization failed: ${authResult.status}`);
    sendError(
      res,
      authResult.status,
      authResult.error || "Authorization failed",
    );
    return;
  }
  const deviceId = authResult.deviceId;
  console.log(
    `✅ Authorized device ${deviceId}, balance: ${authResult.creditBalance}`,
  );

  // Step 2: Parse and transcribe the file
  try {
    const form = new IncomingForm({
      maxFileSize: 500 * 1024 * 1024, // 500MB
    });
    const [fields, files] = await form.parse(req);

    const file = Array.isArray(files.file) ? files.file[0] : files.file;
    if (!file) {
      console.log("❌ No file provided");
      sendError(res, 400, "No file provided");
      return;
    }
    const whisperSizeGuardMessage = getWhisperFileSizeGuardMessage(file.size);

    const language = Array.isArray(fields.language)
      ? fields.language[0]
      : fields.language;
    const prompt = Array.isArray(fields.prompt)
      ? fields.prompt[0]
      : fields.prompt;
    const modelHint = Array.isArray(fields.model)
      ? fields.model[0]
      : fields.model;
    const modelIdHint = Array.isArray(fields.model_id)
      ? fields.model_id[0]
      : fields.model_id;
    const qualityModeRaw =
      (Array.isArray(fields.qualityMode)
        ? fields.qualityMode[0]
        : fields.qualityMode) ??
      (Array.isArray(fields.quality_mode)
        ? fields.quality_mode[0]
        : fields.quality_mode);
    const { useHighQuality, source: qualitySource } =
      resolveDirectTranscriptionQuality({
        explicitQualityRaw: qualityModeRaw,
        modelHint: typeof modelHint === "string" ? modelHint : undefined,
        modelIdHint: typeof modelIdHint === "string" ? modelIdHint : undefined,
      });

    console.log(
      `🎵 Direct transcription mode: ${
        useHighQuality ? "elevenlabs" : "whisper"
      } (qualitySource=${qualitySource}) for ${file.originalFilename} (${(
        file.size /
        1024 /
        1024
      ).toFixed(1)}MB)`,
    );

    const openaiKey = process.env.OPENAI_API_KEY;
    const elevenLabsKey = process.env.ELEVENLABS_API_KEY;
    let effectiveHighQuality = useHighQuality;
    if (effectiveHighQuality && !elevenLabsKey && openaiKey) {
      effectiveHighQuality = false;
      console.warn(
        "⚠️ ElevenLabs key missing for high-quality /transcribe-direct; falling back to Whisper.",
      );
    }

    let transcriptionResult: any;
    let billedModel: string;

    if (effectiveHighQuality) {
      if (!elevenLabsKey) {
        sendError(res, 500, "ElevenLabs not configured");
        return;
      }

      try {
        const { result, attempts } = await transcribeWithScribeWithRetries({
          filePath: file.filepath,
          apiKey: elevenLabsKey,
          languageCode: language || "auto",
          idempotencyKey,
          contextLabel: "/transcribe-direct",
        });
        transcriptionResult = toWhisperCompatibleScribeResult(result);
        if (attempts > 1) {
          transcriptionResult = {
            ...transcriptionResult,
            retry: {
              provider: ELEVENLABS_TRANSCRIPTION_MODEL,
              attempts,
            },
          };
        }
        billedModel = ELEVENLABS_TRANSCRIPTION_MODEL;
      } catch (scribeError: any) {
        if (!openaiKey) {
          throw scribeError;
        }

        const attempts =
          Number((scribeError as any)?.scribeAttempts) || SCRIBE_MAX_RETRIES;
        if (whisperSizeGuardMessage) {
          const reason = scribeError?.message || String(scribeError);
          console.warn(
            `⚠️ /transcribe-direct cannot fall back to Whisper after ${attempts} ElevenLabs attempts: ${whisperSizeGuardMessage}`,
          );
          sendError(
            res,
            502,
            "ElevenLabs transcription failed and Whisper fallback is unavailable for this file size",
            `${reason}. ${whisperSizeGuardMessage}`,
          );
          return;
        }
        console.warn(
          `⚠️ /transcribe-direct falling back to Whisper after ${attempts} ElevenLabs attempts: ${
            scribeError?.message || String(scribeError)
          }`,
        );

        transcriptionResult = await transcribeWithWhisperFromPath({
          openaiKey,
          filePath: file.filepath,
          fileName: file.originalFilename || "audio.webm",
          mimeType: file.mimetype || "audio/webm",
          language: language || undefined,
          prompt: prompt || undefined,
        });
        transcriptionResult = {
          ...transcriptionResult,
          fallback: {
            from: ELEVENLABS_TRANSCRIPTION_MODEL,
            to: WHISPER_TRANSCRIPTION_MODEL,
            attempts,
            reason: scribeError?.message || String(scribeError),
          },
        };
        billedModel = WHISPER_TRANSCRIPTION_MODEL;
      }
    } else {
      if (!openaiKey) {
        sendError(res, 500, "OpenAI not configured");
        return;
      }
      if (whisperSizeGuardMessage) {
        sendError(
          res,
          413,
          "File too large for Whisper transcription",
          whisperSizeGuardMessage,
        );
        return;
      }

      transcriptionResult = await transcribeWithWhisperFromPath({
        openaiKey,
        filePath: file.filepath,
        fileName: file.originalFilename || "audio.webm",
        mimeType: file.mimetype || "audio/webm",
        language: language || undefined,
        prompt: prompt || undefined,
      });
      billedModel = WHISPER_TRANSCRIPTION_MODEL;
    }

    const durationForBilling =
      Number.isFinite(transcriptionResult?.duration) &&
      transcriptionResult.duration > 0
        ? transcriptionResult.duration
        : Number.isFinite(transcriptionResult?.approx_duration) &&
            transcriptionResult.approx_duration > 0
          ? transcriptionResult.approx_duration
          : 0;

    console.log(
      `🎯 Transcription completed! Duration: ${durationForBilling.toFixed(1)}s model=${billedModel}`,
    );

    // Step 3: Deduct credits
    console.log(
      `💳 Deducting credits for ${Math.ceil(durationForBilling)}s...`,
    );
    try {
      const deductResult = await deductRelayCredits({
        cfApiBase: CF_API_BASE,
        relaySecret: RELAY_SECRET,
        payload: {
          deviceId,
          service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
          seconds: durationForBilling,
          model: billedModel,
          ...(idempotencyKey ? { idempotencyKey } : {}),
        },
      });

      if (!deductResult.ok) {
        console.error(`❌ Credit deduction failed: ${deductResult.status}`);
        const status = deductResult.status === 402 ? 402 : 500;
        sendError(res, status, deductResult.error || "Credit deduction failed");
        return;
      }

      console.log(`✅ Credits deducted successfully`);
    } catch (deductErr: any) {
      console.error("❌ Credit deduction request failed:", deductErr.message);
      sendError(res, 500, "Credit deduction failed", deductErr?.message);
      return;
    }

    // Return result to app (only after successful deduction)
    sendJson(res, transcriptionResult);
  } catch (error: any) {
    console.error("❌ Transcription error:", error.message);
    sendError(res, 500, "Transcription failed", error.message);
  }
}

// TODO(stage5-cleanup): Remove this legacy endpoint after stage5-api no longer
// calls /transcribe-from-r2 (R2 webhook transcription flow retired).
async function handleTranscribeFromR2(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext,
): Promise<void> {
  console.log("📡 Processing ElevenLabs Scribe from R2 URL...");

  const {
    RELAY_SECRET,
    R2_FETCH_TIMEOUT_MS,
    getHeader,
    sendError,
    sendJson,
    validateRelaySecret,
    transcribeWithScribe,
    toWhisperCompatibleScribeResult,
    validateR2Url,
  } = ctx;

  if (!validateRelaySecret(req, RELAY_SECRET)) {
    console.log("❌ Invalid or missing relay secret for /transcribe-from-r2");
    sendError(res, 401, "Unauthorized - invalid relay secret");
    return;
  }

  const idempotencyKey =
    getHeader(req, "idempotency-key") || getHeader(req, "x-idempotency-key");

  const elevenLabsKey =
    getHeader(req, "x-elevenlabs-key") || process.env.ELEVENLABS_API_KEY;
  if (!elevenLabsKey) {
    console.log("❌ ElevenLabs API key not provided");
    sendError(res, 500, "ElevenLabs not configured");
    return;
  }

  // Capture relay secret for webhook callback
  const relaySecret = getHeader(req, "x-relay-secret") || "";

  try {
    // Parse JSON body
    let body = "";
    for await (const chunk of req) {
      body += chunk;
    }
    const { r2Url, language, webhookUrl } = JSON.parse(body);

    if (!r2Url) {
      sendError(res, 400, "r2Url is required");
      return;
    }

    // SSRF prevention: validate R2 URL
    const r2Validation = validateR2Url(r2Url);
    if (!r2Validation.valid) {
      console.log(`❌ Invalid R2 URL: ${r2Validation.error}`);
      sendError(res, 400, "Invalid R2 URL", r2Validation.error);
      return;
    }

    console.log(`🎵 Fetching audio from R2 for transcription...`);

    // Helper to process transcription (used for both sync and async modes)
    const processTranscription = async () => {
      // Fetch the file from R2 with timeout
      const abortController = new AbortController();
      const timeoutId = setTimeout(
        () => abortController.abort(),
        R2_FETCH_TIMEOUT_MS,
      );

      let r2Response: Response;
      try {
        r2Response = await fetch(r2Url, { signal: abortController.signal });
      } finally {
        clearTimeout(timeoutId);
      }

      if (!r2Response.ok) {
        throw new Error(`Failed to fetch from R2: ${r2Response.status}`);
      }

      const audioBuffer = Buffer.from(await r2Response.arrayBuffer());
      const fileSizeMB = audioBuffer.length / (1024 * 1024);

      console.log(
        `🎵 Transcribing with ElevenLabs Scribe (${fileSizeMB.toFixed(1)}MB from R2)`,
      );

      // Write to temp file for ElevenLabs
      const fs = await import("fs");
      const os = await import("os");
      const path = await import("path");
      const tempFile = path.join(os.tmpdir(), `r2-audio-${Date.now()}.webm`);
      await fs.promises.writeFile(tempFile, audioBuffer);

      try {
        const result = await transcribeWithScribe({
          filePath: tempFile,
          apiKey: elevenLabsKey,
          languageCode: language || "auto",
          idempotencyKey,
        });
        const whisperFormat = toWhisperCompatibleScribeResult(result);
        const duration =
          Number.isFinite(whisperFormat?.duration) && whisperFormat.duration > 0
            ? whisperFormat.duration
            : 0;

        console.log(
          `🎯 ElevenLabs Scribe (R2) completed! Duration: ${duration.toFixed(1)}s`,
        );
        return { success: true, result: whisperFormat };
      } finally {
        // Cleanup temp file
        try {
          await fs.promises.unlink(tempFile);
        } catch (cleanupErr: any) {
          console.warn(
            `⚠️ Failed to cleanup temp file ${tempFile}:`,
            cleanupErr.message,
          );
        }
      }
    };

    // If webhook URL provided, process async and return immediately
    if (webhookUrl) {
      console.log(`📞 Webhook mode: will POST result to ${webhookUrl}`);
      sendJson(res, {
        status: "processing",
        message: "Transcription started, result will be sent to webhook",
      });

      // Process in background and call webhook when done
      processTranscription()
        .then(async ({ result }) => {
          try {
            console.log(`📞 Calling webhook: ${webhookUrl}`);
            const webhookRes = await fetch(webhookUrl, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "X-Relay-Secret": relaySecret,
              },
              body: JSON.stringify({ success: true, result }),
            });
            if (webhookRes.ok) {
              console.log(`✅ Webhook callback successful`);
            } else {
              console.error(`❌ Webhook callback failed: ${webhookRes.status}`);
            }
          } catch (webhookErr: any) {
            console.error(`❌ Webhook callback error:`, webhookErr.message);
          }
        })
        .catch(async (error: any) => {
          console.error(
            `❌ Transcription failed, notifying webhook:`,
            error.message,
          );
          try {
            await fetch(webhookUrl, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                "X-Relay-Secret": relaySecret,
              },
              body: JSON.stringify({ success: false, error: error.message }),
            });
          } catch (webhookErr: any) {
            console.error(
              `❌ Webhook error callback failed:`,
              webhookErr.message,
            );
          }
        });
      return;
    }

    // Synchronous mode (no webhook) - original behavior
    const { result } = await processTranscription();
    sendJson(res, result);
  } catch (error: any) {
    console.error("❌ ElevenLabs Scribe (R2) error:", error.message);
    sendError(res, 500, "Transcription from R2 failed", error.message);
  }
}

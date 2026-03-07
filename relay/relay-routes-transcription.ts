import { Buffer } from "node:buffer";
import type { IncomingMessage, ServerResponse } from "node:http";
import { IncomingForm } from "formidable";
import type { RelayRoutesContext } from "./relay-routes.js";
import {
  createDirectRequestLease,
  normalizeRelayRecoveryFailureStatus,
  persistDirectReplayOrRelease,
  recoverOrRestartDuplicateReservation,
  startDirectRequestLeaseHeartbeat,
} from "./direct-replay-recovery.js";
import {
  deleteStoredDirectReplayArtifact,
  extractStoredDirectReplayResult,
  materializeStoredDirectReplayResult,
  storeSuccessDirectReplayArtifact,
  type DirectReplayResult as SharedDirectReplayResult,
  type StoredDirectReplayResult,
} from "./direct-replay-artifacts.js";
import {
  authorizeRelayDevice,
  confirmRelayReservation,
  finalizeRelayCredits,
  releaseRelayCredits,
  reserveRelayCredits,
} from "./relay-billing-client.js";
import { STAGE5_RELAY_BILLING_SERVICES } from "./relay-billing-contract.js";
import {
  getInternalRelayBillingContext,
} from "./relay-billing-helpers.js";
import { probeMediaDurationSeconds } from "./audio-probe.js";
import { buildDirectRelayTranscriptionRequestKey } from "./transcription-idempotency.js";

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

const TRANSCRIPTION_RESERVE_PADDING_SECONDS = Math.max(
  0,
  Number.parseInt(process.env.TRANSCRIPTION_RESERVE_PADDING_SECONDS || "2", 10),
);

function toReservationSeconds(durationSeconds: number): number {
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
    return 0;
  }
  return Math.ceil(durationSeconds) + TRANSCRIPTION_RESERVE_PADDING_SECONDS;
}

async function probeReservationSecondsFromPath(filePath: string): Promise<number> {
  const durationSeconds = await probeMediaDurationSeconds(filePath);
  const reservationSeconds = toReservationSeconds(durationSeconds);
  if (reservationSeconds <= 0) {
    throw new Error("Unable to determine a billable audio duration");
  }
  return reservationSeconds;
}

function buildRelayOwnedDirectRequestOwnership(): {
  directRequestOwnership: {
    version: 1;
    state: "relay-owned";
    updatedAt: string;
  };
} {
  return {
    directRequestOwnership: {
      version: 1,
      state: "relay-owned",
      updatedAt: new Date().toISOString(),
    },
  };
}

type DirectTranscriptionReplayResult = SharedDirectReplayResult;

type DirectTranscriptionReplayEntry = {
  done: boolean;
  promise: Promise<DirectTranscriptionReplayResult>;
  resolve: (result: DirectTranscriptionReplayResult) => void;
  result?: DirectTranscriptionReplayResult;
  expiresAt?: number;
};

type PendingDirectTranscriptionFinalize = {
  seconds: number;
  model: string;
};

const DIRECT_TRANSCRIPTION_REPLAY_TTL_MS = Math.max(
  1_000,
  Number.parseInt(
    process.env.DIRECT_TRANSCRIPTION_REPLAY_TTL_MS || String(10 * 60 * 1_000),
    10,
  ),
);
const directTranscriptionReplayCache = new Map<
  string,
  DirectTranscriptionReplayEntry
>();

function createDirectTranscriptionReplayEntry(): DirectTranscriptionReplayEntry {
  let resolve!: (result: DirectTranscriptionReplayResult) => void;
  const promise = new Promise<DirectTranscriptionReplayResult>((innerResolve) => {
    resolve = innerResolve;
  });
  return {
    done: false,
    promise,
    resolve,
  };
}

function pruneDirectTranscriptionReplayCache(now = Date.now()): void {
  for (const [requestKey, entry] of directTranscriptionReplayCache.entries()) {
    if (
      entry.done &&
      entry.result?.kind === "success" &&
      typeof entry.expiresAt === "number" &&
      entry.expiresAt <= now
    ) {
      directTranscriptionReplayCache.delete(requestKey);
    }
  }
}

function settleDirectTranscriptionReplayEntry({
  requestKey,
  entry,
  result,
  cacheSuccess = false,
}: {
  requestKey: string;
  entry: DirectTranscriptionReplayEntry;
  result: DirectTranscriptionReplayResult;
  cacheSuccess?: boolean;
}): void {
  if (entry.done) {
    return;
  }

  entry.done = true;
  entry.result = result;
  entry.resolve(result);

  if (cacheSuccess && result.kind === "success") {
    entry.expiresAt = Date.now() + DIRECT_TRANSCRIPTION_REPLAY_TTL_MS;
    directTranscriptionReplayCache.set(requestKey, entry);
    return;
  }

  directTranscriptionReplayCache.delete(requestKey);
}

function sendDirectTranscriptionReplay(
  res: ServerResponse,
  replay: DirectTranscriptionReplayResult,
  sendError: RelayRoutesContext["sendError"],
  sendJson: RelayRoutesContext["sendJson"],
): void {
  if (replay.kind === "success") {
    sendJson(res, replay.data, replay.status);
    return;
  }

  sendError(res, replay.status, replay.error, replay.details);
}

function asObject(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function extractStoredDirectTranscriptionReplay(
  reservationMeta: unknown,
): StoredDirectReplayResult | null {
  return extractStoredDirectReplayResult(
    reservationMeta,
    "Transcription failed",
  );
}

function buildStoredDirectTranscriptionReplayMeta(
  storedReplay: StoredDirectReplayResult | null,
): Record<string, unknown> {
  if (!storedReplay) {
    return {};
  }
  return {
    directReplayResult: storedReplay,
  };
}

function extractPendingDirectTranscriptionFinalize(
  reservationMeta: unknown,
): PendingDirectTranscriptionFinalize | null {
  const metaObject = asObject(reservationMeta);
  const pendingObject = asObject(metaObject?.pendingFinalize);
  if (!pendingObject) {
    return null;
  }

  const seconds = Number(pendingObject.seconds);
  const model =
    typeof pendingObject.model === "string" ? pendingObject.model.trim() : "";
  if (!model || !Number.isFinite(seconds) || seconds < 0) {
    return null;
  }

  return { seconds, model };
}

async function recoverReservedDirectTranscriptionReplay({
  cfApiBase,
  relaySecret,
  deviceId,
  requestKey,
  reservationMeta,
}: {
  cfApiBase: string;
  relaySecret: string;
  deviceId: string;
  requestKey: string;
  reservationMeta: unknown;
}): Promise<
  | { kind: "replay"; replay: DirectTranscriptionReplayResult }
  | { kind: "error"; status: number; error: string; details?: string }
  | null
> {
  const storedReplay = extractStoredDirectTranscriptionReplay(reservationMeta);
  const replay = await materializeStoredDirectReplayResult({
    cfApiBase,
    relaySecret,
    storedReplay,
  });
  const pendingFinalize = extractPendingDirectTranscriptionFinalize(
    reservationMeta,
  );
  if (!storedReplay || !replay || !pendingFinalize) {
    return null;
  }
  if (replay.kind !== "success") {
    return {
      kind: "error",
      status: replay.status,
      error: replay.error,
      ...(replay.details ? { details: replay.details } : {}),
    };
  }

  const finalizeResult = await finalizeRelayCredits({
    cfApiBase,
    relaySecret,
    payload: {
      deviceId,
      requestKey,
      service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
      seconds: pendingFinalize.seconds,
      model: pendingFinalize.model,
      meta: {
        ...buildStoredDirectTranscriptionReplayMeta(storedReplay),
        pendingFinalize: null,
      },
    },
  });

  if (!finalizeResult.ok) {
    return {
      kind: "error",
      status: normalizeRelayRecoveryFailureStatus(finalizeResult.status),
      error: finalizeResult.error || "Credit finalize failed",
    };
  }

  return { kind: "replay", replay };
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
      service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
    },
  });
  if (!confirmResult.ok) {
    sendError(res, confirmResult.status, confirmResult.error || "Reservation confirmation failed");
    return;
  }

  const idempotencyKey =
    getHeader(req, "idempotency-key") || getHeader(req, "x-idempotency-key");

  let requestClosed = false;
  const requestAbortController = new AbortController();
  req.on("close", () => {
    requestClosed = true;
    requestAbortController.abort();
  });

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
    const elevenLabsKey = getHeader(req, "x-elevenlabs-key");

    let effectiveHighQuality = useHighQuality;
    if (effectiveHighQuality && !elevenLabsKey && openaiKey) {
      effectiveHighQuality = false;
      console.warn(
        "⚠️ ElevenLabs key missing for high-quality /transcribe; falling back to Whisper.",
      );
    }

    let reservationSeconds: number;
    try {
      reservationSeconds = await probeReservationSecondsFromPath(file.filepath);
    } catch (probeError: any) {
      sendError(
        res,
        422,
        "Unable to determine audio duration for billing",
        probeError?.message || String(probeError),
      );
      return;
    }

    // Worker-side reservations are only provisional. Probe the uploaded media
    // here and top up to the actual duration before any vendor call starts.
    const exactConfirmResult = await confirmRelayReservation({
      cfApiBase: CF_API_BASE,
      relaySecret: RELAY_SECRET,
      payload: {
        deviceId: internalBilling.deviceId,
        requestKey: internalBilling.requestKey,
        service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
        seconds: reservationSeconds,
        model: effectiveHighQuality
          ? ELEVENLABS_TRANSCRIPTION_MODEL
          : WHISPER_TRANSCRIPTION_MODEL,
        meta: buildRelayOwnedDirectRequestOwnership(),
      },
    });
    if (!exactConfirmResult.ok) {
      sendError(
        res,
        exactConfirmResult.status,
        exactConfirmResult.error || "Reservation confirmation failed",
      );
      return;
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
            signal: requestAbortController.signal,
          });
        const whisperFormat = toWhisperCompatibleScribeResult(scribeResult);
        if (attempts > 1) {
          (whisperFormat as any).retry = {
            provider: ELEVENLABS_TRANSCRIPTION_MODEL,
            attempts,
          };
        }

        console.log("🎯 Relay transcription completed with ElevenLabs.");
        if (requestClosed) {
          console.warn("⚠️ /transcribe client disconnected after ElevenLabs success");
          return;
        }
        sendJson(res, whisperFormat);
      } catch (scribeError: any) {
        if (requestClosed || requestAbortController.signal.aborted) {
          console.warn("⚠️ /transcribe client disconnected during ElevenLabs transcription");
          return;
        }
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
          signal: requestAbortController.signal,
        });
        if (requestClosed) {
          console.warn("⚠️ /transcribe client disconnected after Whisper fallback success");
          return;
        }
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
        signal: requestAbortController.signal,
      });

      console.log("🎯 Relay transcription completed with Whisper.");
      if (requestClosed) {
        console.warn("⚠️ /transcribe client disconnected after Whisper success");
        return;
      }
      sendJson(res, transcription);
    }
  } catch (error: any) {
    if (requestClosed || requestAbortController.signal.aborted) {
      console.warn("⚠️ /transcribe aborted by upstream client");
      return;
    }
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
    CF_API_BASE,
    RELAY_SECRET,
    ELEVENLABS_TRANSCRIPTION_MODEL,
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
      service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
    },
  });
  if (!confirmResult.ok) {
    sendError(res, confirmResult.status, confirmResult.error || "Reservation confirmation failed");
    return;
  }

  const idempotencyKey =
    getHeader(req, "idempotency-key") || getHeader(req, "x-idempotency-key");

  let requestClosed = false;
  const requestAbortController = new AbortController();
  req.on("close", () => {
    requestClosed = true;
    requestAbortController.abort();
  });

  const elevenLabsKey = getHeader(req, "x-elevenlabs-key");
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

    let reservationSeconds: number;
    try {
      reservationSeconds = await probeReservationSecondsFromPath(file.filepath);
    } catch (probeError: any) {
      sendError(
        res,
        422,
        "Unable to determine audio duration for billing",
        probeError?.message || String(probeError),
      );
      return;
    }

    // Direct relay transcription also settles the duration estimate to the
    // probed media length before starting vendor work.
    const exactConfirmResult = await confirmRelayReservation({
      cfApiBase: CF_API_BASE,
      relaySecret: RELAY_SECRET,
      payload: {
        deviceId: internalBilling.deviceId,
        requestKey: internalBilling.requestKey,
        service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
        seconds: reservationSeconds,
        model: ELEVENLABS_TRANSCRIPTION_MODEL,
        meta: buildRelayOwnedDirectRequestOwnership(),
      },
    });
    if (!exactConfirmResult.ok) {
      sendError(
        res,
        exactConfirmResult.status,
        exactConfirmResult.error || "Reservation confirmation failed",
      );
      return;
    }

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
      signal: requestAbortController.signal,
    });
    const whisperFormat = toWhisperCompatibleScribeResult(result);

    console.log(`🎯 ElevenLabs Scribe transcription completed!`);
    if (requestClosed) {
      console.warn("⚠️ /transcribe-elevenlabs client disconnected after success");
      return;
    }
    sendJson(res, whisperFormat);
  } catch (error: any) {
    if (requestClosed || requestAbortController.signal.aborted) {
      console.warn("⚠️ /transcribe-elevenlabs aborted by upstream client");
      return;
    }
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
  const appVersion = getHeader(req, "x-stage5-app-version");

  // Step 1: Authorize with CF Worker
  console.log(`🔐 Authorizing with CF Worker...`);

  const authResult = await authorizeRelayDevice({
    cfApiBase: CF_API_BASE,
    relaySecret: RELAY_SECRET,
    apiKey,
    service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
    clientIdempotencyKey: idempotencyKey,
    appVersion,
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
  let replayContext:
    | { requestKey: string; entry: DirectTranscriptionReplayEntry }
    | null = null;
  let stopLeaseHeartbeat: (() => void) | null = null;
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

    let reservationSeconds: number;
    try {
      reservationSeconds = await probeReservationSecondsFromPath(file.filepath);
    } catch (probeError: any) {
      sendError(
        res,
        422,
        "Unable to determine audio duration for billing",
        probeError?.message || String(probeError),
      );
      return;
    }

    const requestKey = await buildDirectRelayTranscriptionRequestKey({
      deviceId,
      clientIdempotencyKey: idempotencyKey,
      filePath: file.filepath,
      language: language || null,
      prompt: prompt || null,
      modelHint: typeof modelHint === "string" ? modelHint : null,
      modelIdHint: typeof modelIdHint === "string" ? modelIdHint : null,
      qualityMode: typeof qualityModeRaw === "string" ? qualityModeRaw : null,
    });
    pruneDirectTranscriptionReplayCache();
    const existingReplay = directTranscriptionReplayCache.get(requestKey);
    if (existingReplay) {
      const replay = existingReplay.result ?? await existingReplay.promise;
      sendDirectTranscriptionReplay(res, replay, sendError, sendJson);
      return;
    }

    const replayEntry = createDirectTranscriptionReplayEntry();
    directTranscriptionReplayCache.set(requestKey, replayEntry);
    replayContext = { requestKey, entry: replayEntry };
    const sendReplaySuccess = (data: unknown, status = 200): void => {
      const replay: DirectTranscriptionReplayResult = {
        kind: "success",
        status,
        data,
      };
      settleDirectTranscriptionReplayEntry({
        requestKey,
        entry: replayEntry,
        result: replay,
        cacheSuccess: true,
      });
      sendDirectTranscriptionReplay(res, replay, sendError, sendJson);
    };
    const sendReplayError = (
      status: number,
      error: string,
      details?: string,
    ): void => {
      const replay: DirectTranscriptionReplayResult = {
        kind: "error",
        status,
        error,
        ...(details ? { details } : {}),
      };
      settleDirectTranscriptionReplayEntry({
        requestKey,
        entry: replayEntry,
        result: replay,
      });
      sendDirectTranscriptionReplay(res, replay, sendError, sendJson);
    };

    const directRequestLease = createDirectRequestLease();
    let reserveResult: Awaited<ReturnType<typeof reserveRelayCredits>> | null = null;
    for (let orphanRecoveryAttempt = 0; orphanRecoveryAttempt < 2; orphanRecoveryAttempt += 1) {
      reserveResult = await reserveRelayCredits({
        cfApiBase: CF_API_BASE,
        relaySecret: RELAY_SECRET,
        payload: {
          deviceId,
          requestKey,
          service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
          seconds: reservationSeconds,
          model: effectiveHighQuality
            ? ELEVENLABS_TRANSCRIPTION_MODEL
            : WHISPER_TRANSCRIPTION_MODEL,
          meta: {
            directRequestLease,
          },
        },
      });
      if (!reserveResult.ok) {
        sendReplayError(
          reserveResult.status === 402 ? 402 : reserveResult.status,
          reserveResult.error || "Credit reservation failed",
        );
        return;
      }
      if (reserveResult.status !== "duplicate") {
        break;
      }

      if (reserveResult.reservationStatus === "reserved") {
        const recovered = await recoverReservedDirectTranscriptionReplay({
          cfApiBase: CF_API_BASE,
          relaySecret: RELAY_SECRET,
          deviceId,
          requestKey,
          reservationMeta: reserveResult.reservationMeta,
        });
        if (recovered?.kind === "replay") {
          settleDirectTranscriptionReplayEntry({
            requestKey,
            entry: replayEntry,
            result: recovered.replay,
            cacheSuccess: recovered.replay.kind === "success",
          });
          sendDirectTranscriptionReplay(
            res,
            recovered.replay,
            sendError,
            sendJson,
          );
          return;
        }
        if (recovered?.kind === "error") {
          sendReplayError(
            recovered.status,
            recovered.error,
            recovered.details,
          );
          return;
        }

        const orphanRecovery = await recoverOrRestartDuplicateReservation({
          cfApiBase: CF_API_BASE,
          relaySecret: RELAY_SECRET,
          deviceId,
          requestKey,
          service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
          reservationMeta: reserveResult.reservationMeta,
          reservationUpdatedAt: reserveResult.reservationUpdatedAt,
        });
        if (!orphanRecovery.ok) {
          sendReplayError(
            orphanRecovery.status,
            orphanRecovery.error,
            orphanRecovery.details,
          );
          return;
        }
        if (orphanRecovery.action === "reservation-settled") {
          const settledReplay = await materializeStoredDirectReplayResult({
            cfApiBase: CF_API_BASE,
            relaySecret: RELAY_SECRET,
            storedReplay: extractStoredDirectTranscriptionReplay(
              orphanRecovery.reservationMeta,
            ),
          });
          if (settledReplay?.kind === "success") {
            settleDirectTranscriptionReplayEntry({
              requestKey,
              entry: replayEntry,
              result: settledReplay,
              cacheSuccess: true,
            });
            sendDirectTranscriptionReplay(
              res,
              settledReplay,
              sendError,
              sendJson,
            );
            return;
          }
          if (settledReplay?.kind === "error") {
            sendReplayError(
              settledReplay.status,
              settledReplay.error,
              settledReplay.details,
            );
            return;
          }
          sendReplayError(409, "Duplicate request already completed");
          return;
        }
        if (orphanRecovery.action === "retry-reserve") {
          continue;
        }
      }

      const persistedReplay = await materializeStoredDirectReplayResult({
        cfApiBase: CF_API_BASE,
        relaySecret: RELAY_SECRET,
        storedReplay: extractStoredDirectTranscriptionReplay(
          reserveResult.reservationMeta,
        ),
      });
      if (persistedReplay?.kind === "success") {
        settleDirectTranscriptionReplayEntry({
          requestKey,
          entry: replayEntry,
          result: persistedReplay,
          cacheSuccess: true,
        });
        sendDirectTranscriptionReplay(
          res,
          persistedReplay,
          sendError,
          sendJson,
        );
        return;
      }
      if (persistedReplay?.kind === "error") {
        sendReplayError(
          persistedReplay.status,
          persistedReplay.error,
          persistedReplay.details,
        );
        return;
      }

      sendReplayError(409, "Duplicate request");
      return;
    }
    if (!reserveResult || !reserveResult.ok || reserveResult.status === "duplicate") {
      sendReplayError(500, "Credit reservation failed");
      return;
    }
    stopLeaseHeartbeat = startDirectRequestLeaseHeartbeat({
      cfApiBase: CF_API_BASE,
      relaySecret: RELAY_SECRET,
      deviceId,
      requestKey,
      service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
      lease: directRequestLease,
    });

    let transcriptionResult: any;
    let billedModel: string;

    try {
      if (effectiveHighQuality) {
        if (!elevenLabsKey) {
          await releaseRelayCredits({
            cfApiBase: CF_API_BASE,
            relaySecret: RELAY_SECRET,
            payload: {
              deviceId,
              requestKey,
              service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
            },
          });
          sendReplayError(500, "ElevenLabs not configured");
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
            await releaseRelayCredits({
              cfApiBase: CF_API_BASE,
              relaySecret: RELAY_SECRET,
              payload: {
                deviceId,
                requestKey,
                service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
              },
            });
            sendReplayError(
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
          await releaseRelayCredits({
            cfApiBase: CF_API_BASE,
            relaySecret: RELAY_SECRET,
            payload: {
              deviceId,
              requestKey,
              service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
            },
          });
          sendReplayError(500, "OpenAI not configured");
          return;
        }
        if (whisperSizeGuardMessage) {
          await releaseRelayCredits({
            cfApiBase: CF_API_BASE,
            relaySecret: RELAY_SECRET,
            payload: {
              deviceId,
              requestKey,
              service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
            },
          });
          sendReplayError(
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
    } catch (transcriptionError: any) {
      await releaseRelayCredits({
        cfApiBase: CF_API_BASE,
        relaySecret: RELAY_SECRET,
        payload: {
          deviceId,
          requestKey,
          service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
          meta: { reason: "vendor-error", message: transcriptionError?.message || String(transcriptionError) },
        },
      });
      throw transcriptionError;
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

    const replaySuccess: DirectTranscriptionReplayResult = {
      kind: "success",
      status: 200,
      data: transcriptionResult,
    };
    const pendingFinalize: PendingDirectTranscriptionFinalize = {
      seconds: durationForBilling,
      model: billedModel,
    };
    const storedReplayResult = await storeSuccessDirectReplayArtifact({
      cfApiBase: CF_API_BASE,
      relaySecret: RELAY_SECRET,
      deviceId,
      requestKey,
      service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
      replay: replaySuccess,
    });
    if (!storedReplayResult.ok) {
      sendReplayError(
        storedReplayResult.status,
        storedReplayResult.error || "Replay persistence failed",
      );
      return;
    }
    const persistResult = await persistDirectReplayOrRelease({
      cfApiBase: CF_API_BASE,
      relaySecret: RELAY_SECRET,
      deviceId,
      requestKey,
      service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
      replayResult: storedReplayResult.storedReplay,
      pendingFinalize,
    });
    if (!persistResult.ok) {
      await deleteStoredDirectReplayArtifact({
        cfApiBase: CF_API_BASE,
        relaySecret: RELAY_SECRET,
        storedReplay: storedReplayResult.storedReplay,
      });
      sendReplayError(
        persistResult.status,
        persistResult.error || "Replay persistence failed",
        persistResult.details,
      );
      return;
    }
    stopLeaseHeartbeat?.();
    stopLeaseHeartbeat = null;

    console.log(`💳 Finalizing credits for ${Math.ceil(durationForBilling)}s...`);
    try {
      const finalizeResult = await finalizeRelayCredits({
        cfApiBase: CF_API_BASE,
        relaySecret: RELAY_SECRET,
        payload: {
          deviceId,
          requestKey,
          service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
          seconds: durationForBilling,
          model: billedModel,
          meta: {
            ...buildStoredDirectTranscriptionReplayMeta(
              storedReplayResult.storedReplay,
            ),
            pendingFinalize: null,
          },
        },
      });

      if (!finalizeResult.ok) {
        console.error(`❌ Credit finalize failed: ${finalizeResult.status}`);
        sendReplayError(
          normalizeRelayRecoveryFailureStatus(finalizeResult.status),
          finalizeResult.error || "Credit finalize failed",
        );
        return;
      }

      console.log(`✅ Credits finalized successfully`);
    } catch (deductErr: any) {
      console.error("❌ Credit finalize request failed:", deductErr.message);
      sendReplayError(500, "Credit finalize failed", deductErr?.message);
      return;
    }

    // Return result to app (only after successful deduction)
    sendReplaySuccess(transcriptionResult);
  } catch (error: any) {
    console.error("❌ Transcription error:", error.message);
    if (replayContext) {
      const replay: DirectTranscriptionReplayResult = {
        kind: "error",
        status: 500,
        error: "Transcription failed",
        details: error.message,
      };
      settleDirectTranscriptionReplayEntry({
        requestKey: replayContext.requestKey,
        entry: replayContext.entry,
        result: replay,
      });
      sendDirectTranscriptionReplay(res, replay, sendError, sendJson);
      return;
    }

    sendError(res, 500, "Transcription failed", error.message);
  } finally {
    stopLeaseHeartbeat?.();
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
    CF_API_BASE,
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
      service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
    },
  });
  if (!confirmResult.ok) {
    sendError(res, confirmResult.status, confirmResult.error || "Reservation confirmation failed");
    return;
  }

  const idempotencyKey =
    getHeader(req, "idempotency-key") || getHeader(req, "x-idempotency-key");

  const elevenLabsKey = getHeader(req, "x-elevenlabs-key");
  if (!elevenLabsKey) {
    console.log("❌ ElevenLabs API key not provided");
    sendError(res, 500, "ElevenLabs not configured");
    return;
  }

  try {
    // Parse JSON body
    let body = "";
    for await (const chunk of req) {
      body += chunk;
    }
    const { r2Url, language, webhookUrl, webhookToken } = JSON.parse(body);

    if (!r2Url) {
      sendError(res, 400, "r2Url is required");
      return;
    }
    if (webhookUrl && !String(webhookToken || "").trim()) {
      sendError(res, 400, "webhookToken is required when webhookUrl is provided");
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
        const reservationSeconds = await probeReservationSecondsFromPath(tempFile);
        const exactConfirmResult = await confirmRelayReservation({
          cfApiBase: CF_API_BASE,
          relaySecret: RELAY_SECRET,
          payload: {
            deviceId: internalBilling.deviceId,
            requestKey: internalBilling.requestKey,
            service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
            seconds: reservationSeconds,
            model: ctx.ELEVENLABS_TRANSCRIPTION_MODEL,
          },
        });
        if (!exactConfirmResult.ok) {
          throw new Error(
            exactConfirmResult.error || "Reservation confirmation failed"
          );
        }

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
                "X-Relay-Secret": RELAY_SECRET,
                "X-Stage5-Webhook-Token": String(webhookToken || ""),
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
                "X-Relay-Secret": RELAY_SECRET,
                "X-Stage5-Webhook-Token": String(webhookToken || ""),
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

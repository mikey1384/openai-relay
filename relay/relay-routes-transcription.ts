import crypto from "node:crypto";
import { Buffer } from "node:buffer";
import { createWriteStream } from "node:fs";
import * as fs from "node:fs/promises";
import type { IncomingMessage, ServerResponse } from "node:http";
import os from "node:os";
import path from "node:path";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
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
  if (
    req.method === "POST" &&
    req.url === ELEVENLABS_SPEECH_TO_TEXT_WEBHOOK_PATH
  ) {
    await handleElevenLabsSpeechToTextWebhook(req, res, ctx);
    return true;
  }

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

const ELEVENLABS_SPEECH_TO_TEXT_WEBHOOK_PATH =
  "/webhook/elevenlabs/speech-to-text";
const ELEVENLABS_WEBHOOK_TOLERANCE_MS = 30 * 60 * 1_000;
const TRANSCRIPTION_RESERVE_PADDING_SECONDS = Math.max(
  0,
  Number.parseInt(process.env.TRANSCRIPTION_RESERVE_PADDING_SECONDS || "2", 10),
);

type Stage5TranscriptionWebhookMetadata = {
  stage5WebhookUrl: string;
  stage5WebhookToken: string;
  stage5JobId?: string;
  requestKey?: string;
  deviceId?: string;
  language?: string;
};

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

async function fetchR2AudioToTempFile({
  r2Url,
  fetchTimeoutMs,
}: {
  r2Url: string;
  fetchTimeoutMs: number;
}): Promise<{
  tempFile: string;
  fileSizeMB: number;
  cleanup: () => Promise<void>;
}> {
  const abortController = new AbortController();
  const timeoutId = setTimeout(() => abortController.abort(), fetchTimeoutMs);

  const tempFile = path.join(
    os.tmpdir(),
    `r2-audio-${Date.now()}-${crypto.randomUUID()}.webm`,
  );
  let fileSizeBytes = 0;

  try {
    const r2Response = await fetch(r2Url, { signal: abortController.signal });
    if (!r2Response.ok) {
      throw new Error(`Failed to fetch from R2: ${r2Response.status}`);
    }
    if (!r2Response.body) {
      throw new Error("Failed to fetch from R2: missing response body");
    }

    await pipeline(
      Readable.fromWeb(r2Response.body as any),
      new Transform({
        transform(chunk, _encoding, callback) {
          if (typeof chunk === "string") {
            fileSizeBytes += Buffer.byteLength(chunk);
          } else {
            fileSizeBytes += Buffer.from(chunk).length;
          }
          callback(null, chunk);
        },
      }),
      createWriteStream(tempFile),
    );
  } catch (error: any) {
    try {
      await fs.unlink(tempFile);
    } catch {
      // Best effort cleanup for partial downloads.
    }
    if (abortController.signal.aborted) {
      throw new Error(`Failed to fetch from R2: timed out after ${fetchTimeoutMs}ms`);
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
  const fileSizeMB = fileSizeBytes / (1024 * 1024);

  let cleaned = false;
  return {
    tempFile,
    fileSizeMB,
    cleanup: async () => {
      if (cleaned) return;
      cleaned = true;
      try {
        await fs.unlink(tempFile);
      } catch (cleanupErr: any) {
        console.warn(
          `⚠️ Failed to cleanup temp file ${tempFile}:`,
          cleanupErr?.message || cleanupErr,
        );
      }
    },
  };
}

async function forwardStage5DurableTranscriptionWebhook({
  relaySecret,
  webhookUrl,
  webhookToken,
  body,
}: {
  relaySecret: string;
  webhookUrl: string;
  webhookToken: string;
  body: {
    success: boolean;
    result?: unknown;
    error?: string;
  };
}): Promise<void> {
  const stage5Response = await fetch(webhookUrl, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Relay-Secret": relaySecret,
      "X-Stage5-Webhook-Token": webhookToken,
    },
    body: JSON.stringify(body),
  });
  if (!stage5Response.ok) {
    const errorText = await stage5Response.text();
    throw new Error(
      `Stage5 transcription webhook forwarding failed: ${stage5Response.status} ${errorText}`,
    );
  }
}

async function readRawBody(
  req: IncomingMessage,
  {
    maxBytes,
  }: {
    maxBytes?: number;
  } = {},
): Promise<Buffer> {
  const normalizedMaxBytes = Number.isFinite(maxBytes)
    ? Math.max(1, Math.floor(Number(maxBytes)))
    : Number.POSITIVE_INFINITY;
  const chunks: Buffer[] = [];
  let totalSize = 0;
  for await (const chunk of req) {
    const buffer =
      typeof chunk === "string" ? Buffer.from(chunk, "utf8") : Buffer.from(chunk);
    totalSize += buffer.length;
    if (totalSize > normalizedMaxBytes) {
      req.destroy();
      throw new Error("Request body too large");
    }
    chunks.push(buffer);
  }
  return Buffer.concat(chunks, totalSize);
}

function parseElevenLabsSignatureHeader(
  header: string | undefined,
): { timestamp: string; signatures: string[] } | null {
  const value = String(header || "").trim();
  if (!value) return null;

  let timestamp = "";
  const signatures: string[] = [];
  for (const part of value.split(",")) {
    const [rawKey, rawValue] = part.split("=", 2);
    const key = String(rawKey || "").trim();
    const parsedValue = String(rawValue || "").trim();
    if (!key || !parsedValue) continue;
    if (key === "t") {
      timestamp = parsedValue;
      continue;
    }
    if (key === "v0") {
      signatures.push(parsedValue);
    }
  }

  if (!timestamp || signatures.length === 0) {
    return null;
  }

  return { timestamp, signatures };
}

function hasMatchingHexSignature(
  expectedHex: string,
  candidateHex: string,
): boolean {
  try {
    const expected = Buffer.from(expectedHex, "hex");
    const candidate = Buffer.from(candidateHex, "hex");
    return (
      expected.length > 0 &&
      expected.length === candidate.length &&
      crypto.timingSafeEqual(expected, candidate)
    );
  } catch {
    return false;
  }
}

function verifyElevenLabsWebhookSignature({
  rawBody,
  signatureHeader,
  secret,
}: {
  rawBody: Buffer;
  signatureHeader: string | undefined;
  secret: string;
}): boolean {
  const parsed = parseElevenLabsSignatureHeader(signatureHeader);
  if (!parsed) return false;

  const timestampMs = Number.parseInt(parsed.timestamp, 10) * 1_000;
  if (!Number.isFinite(timestampMs)) {
    return false;
  }
  if (Math.abs(Date.now() - timestampMs) > ELEVENLABS_WEBHOOK_TOLERANCE_MS) {
    return false;
  }

  const signedPayload = Buffer.concat([
    Buffer.from(`${parsed.timestamp}.`, "utf8"),
    rawBody,
  ]);
  const expected = crypto
    .createHmac("sha256", secret)
    .update(signedPayload)
    .digest("hex");

  return parsed.signatures.some((candidate) =>
    hasMatchingHexSignature(expected, candidate),
  );
}

function parseStage5TranscriptionWebhookMetadata(
  raw: unknown,
): Stage5TranscriptionWebhookMetadata | null {
  if (typeof raw === "string") {
    try {
      return parseStage5TranscriptionWebhookMetadata(JSON.parse(raw));
    } catch {
      return null;
    }
  }
  if (!raw || typeof raw !== "object") {
    return null;
  }

  const record = raw as Record<string, unknown>;
  const stage5WebhookUrl = String(
    record.stage5WebhookUrl ?? record.stage5_webhook_url ?? "",
  ).trim();
  const stage5WebhookToken = String(
    record.stage5WebhookToken ?? record.stage5_webhook_token ?? "",
  ).trim();

  if (!stage5WebhookUrl || !stage5WebhookToken) {
    return null;
  }

  const stage5JobId = String(
    record.stage5JobId ?? record.stage5_job_id ?? "",
  ).trim();
  const requestKey = String(
    record.requestKey ?? record.request_key ?? "",
  ).trim();
  const deviceId = String(record.deviceId ?? record.device_id ?? "").trim();
  const language = String(record.language ?? "").trim();

  return {
    stage5WebhookUrl,
    stage5WebhookToken,
    ...(stage5JobId ? { stage5JobId } : {}),
    ...(requestKey ? { requestKey } : {}),
    ...(deviceId ? { deviceId } : {}),
    ...(language ? { language } : {}),
  };
}

function extractStage5WebhookErrorMessage(data: Record<string, unknown>): string {
  const candidates = [
    data.error,
    data.error_message,
    data.errorMessage,
    data.message,
  ];

  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim()) {
      return candidate.trim();
    }
  }

  return "ElevenLabs speech-to-text webhook reported a failure";
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

async function handleElevenLabsSpeechToTextWebhook(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext,
): Promise<void> {
  const {
    ELEVENLABS_WEBHOOK_SECRET,
    ELEVENLABS_WEBHOOK_MAX_BODY_SIZE,
    RELAY_SECRET,
    MAX_BODY_SIZE,
    getHeader,
    sendError,
    sendJson,
    normalizeScribeResult,
    toWhisperCompatibleScribeResult,
  } = ctx;

  if (!ELEVENLABS_WEBHOOK_SECRET) {
    console.error(
      "❌ ElevenLabs speech-to-text webhook secret is not configured",
    );
    sendError(res, 500, "ElevenLabs webhook secret not configured");
    return;
  }

  try {
    const rawBody = await readRawBody(req, {
      maxBytes: ELEVENLABS_WEBHOOK_MAX_BODY_SIZE,
    });
    const signatureHeader = getHeader(req, "elevenlabs-signature");
    if (
      !verifyElevenLabsWebhookSignature({
        rawBody,
        signatureHeader,
        secret: ELEVENLABS_WEBHOOK_SECRET,
      })
    ) {
      console.error("❌ Invalid ElevenLabs speech-to-text webhook signature");
      sendError(res, 401, "Unauthorized - invalid ElevenLabs signature");
      return;
    }

    const rawBodyText = rawBody.toString("utf8");
    const event = JSON.parse(rawBodyText) as Record<string, any>;
    const data =
      event?.data && typeof event.data === "object" ? event.data : event;
    const metadata = parseStage5TranscriptionWebhookMetadata(
      data?.webhook_metadata,
    );
    if (!metadata) {
      console.warn(
        "⚠️ Ignoring ElevenLabs speech-to-text webhook without Stage5 metadata",
      );
      sendJson(res, { status: "ignored" });
      return;
    }

    const requestId = String(
      data?.request_id ?? data?.requestId ?? event?.request_id ?? event?.requestId ?? "",
    ).trim() || "unknown";
    const payload =
      data?.transcription && typeof data.transcription === "object"
        ? data.transcription
        : data;
    const hasTranscriptionPayload =
      payload &&
      typeof payload === "object" &&
      typeof payload.text === "string" &&
      typeof payload.language_code === "string";
    const status = String(data?.status ?? event?.status ?? "").trim().toLowerCase();

    const webhookBody = hasTranscriptionPayload
      ? {
          success: true,
          result: toWhisperCompatibleScribeResult(
            normalizeScribeResult(payload as any),
          ),
        }
      : {
          success: false,
          error: extractStage5WebhookErrorMessage(data as Record<string, unknown>),
        };

    if (!hasTranscriptionPayload && status && status !== "failed") {
      console.log(
        `ℹ️ Ignoring ElevenLabs speech-to-text webhook request_id=${requestId} status=${status}`,
      );
      sendJson(res, { status: "ignored", requestId });
      return;
    }

    await forwardStage5DurableTranscriptionWebhook({
      relaySecret: RELAY_SECRET,
      webhookUrl: metadata.stage5WebhookUrl,
      webhookToken: metadata.stage5WebhookToken,
      body: webhookBody,
    });

    console.log(
      `✅ ElevenLabs speech-to-text webhook forwarded request_id=${requestId} job=${metadata.stage5JobId || "unknown"}`,
    );
    sendJson(res, { status: "ok", requestId });
  } catch (error: any) {
    if (error?.message === "Request body too large") {
      sendError(res, 413, "Request body too large");
      return;
    }
    console.error(
      "❌ ElevenLabs speech-to-text webhook handling error:",
      error?.message || error,
    );
    sendError(
      res,
      500,
      "ElevenLabs speech-to-text webhook handling failed",
      error?.message || String(error),
    );
  }
}

async function submitAsyncR2TranscriptionInBackground({
  cfApiBase,
  relaySecret,
  r2Url,
  fetchTimeoutMs,
  language,
  idempotencyKey,
  webhookId,
  stage5WebhookUrl,
  stage5WebhookToken,
  stage5JobId,
  deviceId,
  requestKey,
  elevenLabsKey,
  elevenLabsModel,
  startAsyncTranscriptionWithScribe,
}: {
  cfApiBase: string;
  relaySecret: string;
  r2Url: string;
  fetchTimeoutMs: number;
  language?: string;
  idempotencyKey?: string;
  webhookId?: string;
  stage5WebhookUrl: string;
  stage5WebhookToken: string;
  stage5JobId?: string;
  deviceId: string;
  requestKey: string;
  elevenLabsKey: string;
  elevenLabsModel: string;
  startAsyncTranscriptionWithScribe: RelayRoutesContext["startAsyncTranscriptionWithScribe"];
}): Promise<void> {
  const preparedAudio = await fetchR2AudioToTempFile({
    r2Url,
    fetchTimeoutMs,
  });

  try {
    const reservationSeconds = await probeReservationSecondsFromPath(
      preparedAudio.tempFile,
    );
    const exactConfirmResult = await confirmRelayReservation({
      cfApiBase,
      relaySecret,
      payload: {
        deviceId,
        requestKey,
        service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
        seconds: reservationSeconds,
        model: elevenLabsModel,
      },
    });
    if (!exactConfirmResult.ok) {
      throw new Error(
        `Reservation confirmation failed (${exactConfirmResult.status}): ${
          exactConfirmResult.error || "unknown error"
        }`,
      );
    }

    console.log(
      `📡 Submitting async ElevenLabs Scribe job for R2 URL with webhook callback to Stage5 (${preparedAudio.fileSizeMB.toFixed(1)}MB, ${reservationSeconds}s reserved)`,
    );
    const asyncRequest = await startAsyncTranscriptionWithScribe({
      apiKey: elevenLabsKey,
      cloudStorageUrl: r2Url,
      languageCode: language || "auto",
      idempotencyKey,
      webhookId,
      webhookMetadata: {
        stage5WebhookUrl,
        stage5WebhookToken,
        stage5JobId: stage5JobId || "",
        requestKey,
        deviceId,
        language: language || "auto",
        reservationSeconds,
      },
    });
    console.log(
      `📞 ElevenLabs async transcription accepted: request_id=${asyncRequest.request_id}`,
    );
  } catch (error: any) {
    const message = error?.message || String(error);
    console.error(
      `❌ Async ElevenLabs R2 submission failed for Stage5 job ${stage5JobId || "unknown"}:`,
      message,
    );
    try {
      await forwardStage5DurableTranscriptionWebhook({
        relaySecret,
        webhookUrl: stage5WebhookUrl,
        webhookToken: stage5WebhookToken,
        body: {
          success: false,
          error: message,
        },
      });
      console.log(
        `↩️ Reported async R2 submission failure back to Stage5 job ${stage5JobId || "unknown"}`,
      );
    } catch (forwardError: any) {
      console.error(
        `❌ Failed to report async R2 submission failure back to Stage5 job ${stage5JobId || "unknown"}:`,
        forwardError?.message || String(forwardError),
      );
      throw forwardError;
    }
  } finally {
    await preparedAudio.cleanup();
  }
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

function watchClientDisconnect(
  req: IncomingMessage,
  res: ServerResponse,
  onDisconnect: () => void
): {
  isDisconnected: () => boolean;
  cleanup: () => void;
} {
  let disconnected = false;
  const disconnect = () => {
    if (disconnected) return;
    disconnected = true;
    onDisconnect();
  };

  const onAborted = () => disconnect();
  const onResponseClose = () => {
    if (!res.writableEnded) {
      disconnect();
    }
  };

  req.on("aborted", onAborted);
  res.on("close", onResponseClose);

  return {
    // A fully consumed request body can still flip request internals, so only
    // treat an explicit abort / premature response close as disconnect.
    isDisconnected: () => disconnected || req.aborted,
    cleanup: () => {
      req.off("aborted", onAborted);
      res.off("close", onResponseClose);
    },
  };
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

  const requestAbortController = new AbortController();
  const disconnectWatcher = watchClientDisconnect(req, res, () => {
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
        if (disconnectWatcher.isDisconnected()) {
          console.warn("⚠️ /transcribe client disconnected after ElevenLabs success");
          return;
        }
        sendJson(res, whisperFormat);
      } catch (scribeError: any) {
        if (
          disconnectWatcher.isDisconnected() ||
          requestAbortController.signal.aborted
        ) {
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
        if (disconnectWatcher.isDisconnected()) {
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
      if (disconnectWatcher.isDisconnected()) {
        console.warn("⚠️ /transcribe client disconnected after Whisper success");
        return;
      }
      sendJson(res, transcription);
    }
  } catch (error: any) {
    if (
      disconnectWatcher.isDisconnected() ||
      requestAbortController.signal.aborted
    ) {
      console.warn("⚠️ /transcribe aborted by upstream client");
      return;
    }
    console.error("❌ Relay transcription error:", error.message);
    sendError(res, 500, "Transcription failed", error.message);
  } finally {
    disconnectWatcher.cleanup();
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

  const requestAbortController = new AbortController();
  const disconnectWatcher = watchClientDisconnect(req, res, () => {
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
    if (disconnectWatcher.isDisconnected()) {
      console.warn("⚠️ /transcribe-elevenlabs client disconnected after success");
      return;
    }
    sendJson(res, whisperFormat);
  } catch (error: any) {
    if (
      disconnectWatcher.isDisconnected() ||
      requestAbortController.signal.aborted
    ) {
      console.warn("⚠️ /transcribe-elevenlabs aborted by upstream client");
      return;
    }
    console.error("❌ ElevenLabs Scribe error:", error.message);
    sendError(res, 500, "Transcription failed", error.message);
  } finally {
    disconnectWatcher.cleanup();
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
    ELEVENLABS_SPEECH_TO_TEXT_WEBHOOK_ID,
    ELEVENLABS_WEBHOOK_SECRET,
    MAX_BODY_SIZE,
    RELAY_SECRET,
    R2_FETCH_TIMEOUT_MS,
    getHeader,
    sendError,
    sendJson,
    startAsyncTranscriptionWithScribe,
    validateRelaySecret,
    transcribeWithScribeWithRetries,
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
    const { r2Url, language, webhookUrl, webhookToken } = JSON.parse(
      (await readRawBody(req, { maxBytes: MAX_BODY_SIZE })).toString("utf8"),
    );

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

    if (webhookUrl) {
      if (!ELEVENLABS_WEBHOOK_SECRET) {
        sendError(
          res,
          500,
          "ElevenLabs async webhook secret not configured",
        );
        return;
      }

      const stage5JobId = webhookUrl.split("/").pop() || "";
      console.log(
        `📡 Queueing async ElevenLabs Scribe job for Stage5 webhook flow (job=${stage5JobId || "unknown"})`,
      );
      void submitAsyncR2TranscriptionInBackground({
        cfApiBase: CF_API_BASE,
        relaySecret: RELAY_SECRET,
        r2Url,
        fetchTimeoutMs: R2_FETCH_TIMEOUT_MS,
        language: language || "auto",
        idempotencyKey,
        webhookId: ELEVENLABS_SPEECH_TO_TEXT_WEBHOOK_ID || undefined,
        stage5WebhookUrl: webhookUrl,
        stage5WebhookToken: String(webhookToken || ""),
        stage5JobId,
        deviceId: internalBilling.deviceId,
        requestKey: internalBilling.requestKey,
        elevenLabsKey,
        elevenLabsModel: ctx.ELEVENLABS_TRANSCRIPTION_MODEL,
        startAsyncTranscriptionWithScribe,
      }).catch((error: any) => {
        console.error(
          `❌ Detached async R2 transcription workflow crashed for Stage5 job ${stage5JobId || "unknown"}:`,
          error?.message || String(error),
        );
      });
      sendJson(res, {
        status: "processing",
        message: "Transcription queued, result will be sent to webhook",
      });
      return;
    }

    console.log(`🎵 Fetching audio from R2 for transcription...`);

    const preparedAudio = await fetchR2AudioToTempFile({
      r2Url,
      fetchTimeoutMs: R2_FETCH_TIMEOUT_MS,
    });
    try {
      const reservationSeconds = await probeReservationSecondsFromPath(
        preparedAudio.tempFile,
      );
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
        sendError(
          res,
          exactConfirmResult.status,
          exactConfirmResult.error || "Reservation confirmation failed",
        );
        return;
      }

      console.log(
        `🎵 Transcribing with ElevenLabs Scribe (${preparedAudio.fileSizeMB.toFixed(1)}MB from R2)`,
      );

      const { result, attempts } = await transcribeWithScribeWithRetries({
        filePath: preparedAudio.tempFile,
        apiKey: elevenLabsKey,
        languageCode: language || "auto",
        idempotencyKey,
        contextLabel: "/transcribe-from-r2",
      });
      let whisperFormat = toWhisperCompatibleScribeResult(result);
      if (attempts > 1) {
        whisperFormat = {
          ...whisperFormat,
          retry: {
            provider: ctx.ELEVENLABS_TRANSCRIPTION_MODEL,
            attempts,
          },
        };
      }
      const duration =
        Number.isFinite(whisperFormat?.duration) && whisperFormat.duration > 0
          ? whisperFormat.duration
          : 0;

      console.log(
        `🎯 ElevenLabs Scribe (R2) completed! Duration: ${duration.toFixed(1)}s`,
      );
      sendJson(res, whisperFormat);
    } finally {
      await preparedAudio.cleanup();
    }
  } catch (error: any) {
    if (error?.message === "Request body too large") {
      sendError(res, 413, "Request body too large");
      return;
    }
    console.error("❌ ElevenLabs Scribe (R2) error:", error.message);
    sendError(res, 500, "Transcription from R2 failed", error.message);
  }
}

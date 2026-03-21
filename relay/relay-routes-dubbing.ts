import { Buffer } from "node:buffer";
import type { IncomingMessage, ServerResponse } from "node:http";
import { IncomingForm } from "formidable";
import {
  ELEVENLABS_TTS_MAX_TEXT_CHARACTERS,
  ELEVENLABS_TTS_MODEL_ID,
} from "../elevenlabs-config.js";
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
  buildRelayRequestKey,
  getInternalRelayBillingContext,
} from "./relay-billing-helpers.js";

type DirectDubbingReplayResult = SharedDirectReplayResult;

type DirectDubbingReplayEntry = {
  done: boolean;
  promise: Promise<DirectDubbingReplayResult>;
  resolve: (result: DirectDubbingReplayResult) => void;
  result?: DirectDubbingReplayResult;
  expiresAt?: number;
};

type PendingDirectDubbingFinalize = {
  characters: number;
  model: string;
};

const DIRECT_DUBBING_REPLAY_TTL_MS = Math.max(
  1_000,
  Number.parseInt(
    process.env.DIRECT_DUBBING_REPLAY_TTL_MS || String(10 * 60 * 1_000),
    10,
  ),
);
const directDubbingReplayCache = new Map<string, DirectDubbingReplayEntry>();

function createDirectDubbingReplayEntry(): DirectDubbingReplayEntry {
  let resolve!: (result: DirectDubbingReplayResult) => void;
  const promise = new Promise<DirectDubbingReplayResult>((innerResolve) => {
    resolve = innerResolve;
  });
  return {
    done: false,
    promise,
    resolve,
  };
}

function pruneDirectDubbingReplayCache(now = Date.now()): void {
  for (const [requestKey, entry] of directDubbingReplayCache.entries()) {
    if (
      entry.done &&
      entry.result?.kind === "success" &&
      typeof entry.expiresAt === "number" &&
      entry.expiresAt <= now
    ) {
      directDubbingReplayCache.delete(requestKey);
    }
  }
}

function settleDirectDubbingReplayEntry({
  requestKey,
  entry,
  result,
  cacheSuccess = false,
}: {
  requestKey: string;
  entry: DirectDubbingReplayEntry;
  result: DirectDubbingReplayResult;
  cacheSuccess?: boolean;
}): void {
  if (entry.done) {
    return;
  }

  entry.done = true;
  entry.result = result;
  entry.resolve(result);

  if (cacheSuccess && result.kind === "success") {
    entry.expiresAt = Date.now() + DIRECT_DUBBING_REPLAY_TTL_MS;
    directDubbingReplayCache.set(requestKey, entry);
    return;
  }

  directDubbingReplayCache.delete(requestKey);
}

function sendDirectDubbingReplay(
  res: ServerResponse,
  replay: DirectDubbingReplayResult,
  sendError: RelayRoutesContext["sendError"],
  sendJson: RelayRoutesContext["sendJson"],
): void {
  if (replay.kind === "success") {
    sendJson(res, replay.data, replay.status);
    return;
  }

  sendError(res, replay.status, replay.error, replay.details);
}

function watchClientDisconnect(
  req: IncomingMessage,
  res: ServerResponse,
  onDisconnect: () => void,
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
    // `req.destroyed` flips after a request body has been fully consumed, so
    // it is not a reliable signal that the upstream client cancelled.
    isDisconnected: () => disconnected || req.aborted,
    cleanup: () => {
      req.off("aborted", onAborted);
      res.off("close", onResponseClose);
    },
  };
}

function asObject(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function getRelaySegmentText(segment: any): string {
  if (typeof segment?.text === "string") {
    return segment.text.trim();
  }
  if (typeof segment?.translation === "string") {
    return segment.translation.trim();
  }
  return "";
}

function findOversizedElevenLabsSegment(
  segments: any[],
): { index: number; length: number } | null {
  for (let index = 0; index < segments.length; index += 1) {
    const segment = segments[index];
    const text = getRelaySegmentText(segment);
    if (text.length <= ELEVENLABS_TTS_MAX_TEXT_CHARACTERS) {
      continue;
    }

    return {
      index:
        Number.isFinite(segment?.index) && Number(segment.index) >= 0
          ? Number(segment.index)
          : index + 1,
      length: text.length,
    };
  }

  return null;
}

function extractStoredDirectDubbingReplay(
  reservationMeta: unknown,
): StoredDirectReplayResult | null {
  return extractStoredDirectReplayResult(reservationMeta, "Dubbing failed");
}

function buildStoredDirectDubbingReplayMeta(
  storedReplay: StoredDirectReplayResult | null,
): Record<string, unknown> {
  if (!storedReplay) {
    return {};
  }
  return {
    directReplayResult: storedReplay,
  };
}

function extractPendingDirectDubbingFinalize(
  reservationMeta: unknown,
): PendingDirectDubbingFinalize | null {
  const metaObject = asObject(reservationMeta);
  const pendingObject = asObject(metaObject?.pendingFinalize);
  if (!pendingObject) {
    return null;
  }

  const characters = Number(pendingObject.characters);
  const model =
    typeof pendingObject.model === "string" ? pendingObject.model.trim() : "";
  if (!model || !Number.isFinite(characters) || characters < 0) {
    return null;
  }

  return { characters, model };
}

async function recoverReservedDirectDubbingReplay({
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
  | { kind: "replay"; replay: DirectDubbingReplayResult }
  | { kind: "error"; status: number; error: string; details?: string }
  | null
> {
  const storedReplay = extractStoredDirectDubbingReplay(reservationMeta);
  const replay = await materializeStoredDirectReplayResult({
    cfApiBase,
    relaySecret,
    storedReplay,
  });
  const pendingFinalize = extractPendingDirectDubbingFinalize(reservationMeta);
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
      service: STAGE5_RELAY_BILLING_SERVICES.TTS,
      characters: pendingFinalize.characters,
      model: pendingFinalize.model,
      meta: {
        ...buildStoredDirectDubbingReplayMeta(storedReplay),
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

export async function handleDubbingRoutes(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext,
): Promise<boolean> {
  if (req.method === "POST" && req.url === "/dub-direct") {
    await handleDubDirect(req, res, ctx);
    return true;
  }

  if (req.method === "POST" && req.url === "/dub-elevenlabs") {
    await handleDubElevenLabs(req, res, ctx);
    return true;
  }

  if (req.method === "POST" && req.url === "/dub") {
    await handleDub(req, res, ctx);
    return true;
  }

  return false;
}

async function handleDubDirect(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext,
): Promise<void> {
  console.log("📡 Processing direct dub request...");

  const {
    RELAY_SECRET,
    getHeader,
    sendError,
    sendJson,
    readJsonBody,
    makeOpenAI,
    synthesizeWithElevenLabs,
  } = ctx;

  // Get API key from header (app sends its Stage5 API key)
  const apiKey = getHeader(req, "authorization")?.replace(/^Bearer\s+/i, "");
  if (!apiKey) {
    console.log("❌ Missing API key for /dub-direct");
    sendError(res, 401, "Unauthorized - missing API key");
    return;
  }

  const idempotencyKey =
    getHeader(req, "idempotency-key") || getHeader(req, "x-idempotency-key");
  const appVersion = getHeader(req, "x-stage5-app-version");

  // Step 1: Authorize with CF Worker
  const { CF_API_BASE } = ctx;
  console.log(`🔐 Authorizing with CF Worker...`);

  const authResult = await authorizeRelayDevice({
    cfApiBase: CF_API_BASE,
    relaySecret: RELAY_SECRET,
    apiKey,
    service: STAGE5_RELAY_BILLING_SERVICES.TTS,
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

  // Step 2: Parse request and synthesize speech
  let replayContext: {
    requestKey: string;
    entry: DirectDubbingReplayEntry;
  } | null = null;
  let stopLeaseHeartbeat: (() => void) | null = null;
  const requestAbortController = new AbortController();
  const disconnectWatcher = watchClientDisconnect(req, res, () => {
    requestAbortController.abort();
  });
  try {
    const isRequestAborted = () =>
      disconnectWatcher.isDisconnected() ||
      requestAbortController.signal.aborted;
    const buildAbortError = () => {
      const error = new Error("Request cancelled");
      error.name = "AbortError";
      return error;
    };
    const throwIfRequestAborted = () => {
      if (isRequestAborted()) {
        throw buildAbortError();
      }
    };

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

    if (ttsProvider === "elevenlabs") {
      const oversizedSegment = findOversizedElevenLabsSegment(segments);
      if (oversizedSegment) {
        sendError(
          res,
          413,
          "Segment too long",
          `Segment ${oversizedSegment.index} has ${oversizedSegment.length} characters. ElevenLabs v3 accepts at most ${ELEVENLABS_TTS_MAX_TEXT_CHARACTERS} characters per segment.`,
        );
        return;
      }
    }

    // Calculate total characters for billing
    const totalCharacters = segments.reduce(
      (sum: number, seg: any) =>
        sum + (seg.text?.length || seg.translation?.length || 0),
      0,
    );
    const requestKey = buildRelayRequestKey({
      service: STAGE5_RELAY_BILLING_SERVICES.TTS,
      deviceId,
      clientIdempotencyKey: idempotencyKey,
      payload: {
        segments,
        voice,
        model,
        format,
        ttsProvider,
      },
    });
    pruneDirectDubbingReplayCache();
    const existingReplay = directDubbingReplayCache.get(requestKey);
    if (existingReplay) {
      const replay = existingReplay.result ?? (await existingReplay.promise);
      sendDirectDubbingReplay(res, replay, sendError, sendJson);
      return;
    }

    const replayEntry = createDirectDubbingReplayEntry();
    directDubbingReplayCache.set(requestKey, replayEntry);
    replayContext = { requestKey, entry: replayEntry };
    const sendReplaySuccess = (data: unknown, status = 200): void => {
      const replay: DirectDubbingReplayResult = {
        kind: "success",
        status,
        data,
      };
      settleDirectDubbingReplayEntry({
        requestKey,
        entry: replayEntry,
        result: replay,
        cacheSuccess: true,
      });
      sendDirectDubbingReplay(res, replay, sendError, sendJson);
    };
    const sendReplayError = (
      status: number,
      error: string,
      details?: string,
    ): void => {
      const replay: DirectDubbingReplayResult = {
        kind: "error",
        status,
        error,
        ...(details ? { details } : {}),
      };
      settleDirectDubbingReplayEntry({
        requestKey,
        entry: replayEntry,
        result: replay,
      });
      sendDirectDubbingReplay(res, replay, sendError, sendJson);
    };
    const directRequestLease = createDirectRequestLease();
    let reserveResult: Awaited<ReturnType<typeof reserveRelayCredits>> | null =
      null;
    for (
      let orphanRecoveryAttempt = 0;
      orphanRecoveryAttempt < 2;
      orphanRecoveryAttempt += 1
    ) {
      reserveResult = await reserveRelayCredits({
        cfApiBase: CF_API_BASE,
        relaySecret: RELAY_SECRET,
        payload: {
          deviceId,
          requestKey,
          service: STAGE5_RELAY_BILLING_SERVICES.TTS,
          characters: totalCharacters,
          model: ttsProvider === "elevenlabs" ? ELEVENLABS_TTS_MODEL_ID : model,
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
        const recovered = await recoverReservedDirectDubbingReplay({
          cfApiBase: CF_API_BASE,
          relaySecret: RELAY_SECRET,
          deviceId,
          requestKey,
          reservationMeta: reserveResult.reservationMeta,
        });
        if (recovered?.kind === "replay") {
          settleDirectDubbingReplayEntry({
            requestKey,
            entry: replayEntry,
            result: recovered.replay,
            cacheSuccess: recovered.replay.kind === "success",
          });
          sendDirectDubbingReplay(res, recovered.replay, sendError, sendJson);
          return;
        }
        if (recovered?.kind === "error") {
          sendReplayError(recovered.status, recovered.error, recovered.details);
          return;
        }

        const orphanRecovery = await recoverOrRestartDuplicateReservation({
          cfApiBase: CF_API_BASE,
          relaySecret: RELAY_SECRET,
          deviceId,
          requestKey,
          service: STAGE5_RELAY_BILLING_SERVICES.TTS,
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
            storedReplay: extractStoredDirectDubbingReplay(
              orphanRecovery.reservationMeta,
            ),
          });
          if (settledReplay?.kind === "success") {
            settleDirectDubbingReplayEntry({
              requestKey,
              entry: replayEntry,
              result: settledReplay,
              cacheSuccess: true,
            });
            sendDirectDubbingReplay(res, settledReplay, sendError, sendJson);
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
        storedReplay: extractStoredDirectDubbingReplay(
          reserveResult.reservationMeta,
        ),
      });
      if (persistedReplay?.kind === "success") {
        settleDirectDubbingReplayEntry({
          requestKey,
          entry: replayEntry,
          result: persistedReplay,
          cacheSuccess: true,
        });
        sendDirectDubbingReplay(res, persistedReplay, sendError, sendJson);
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
    if (
      !reserveResult ||
      !reserveResult.ok ||
      reserveResult.status === "duplicate"
    ) {
      sendReplayError(500, "Credit reservation failed");
      return;
    }
    stopLeaseHeartbeat = startDirectRequestLeaseHeartbeat({
      cfApiBase: CF_API_BASE,
      relaySecret: RELAY_SECRET,
      deviceId,
      requestKey,
      service: STAGE5_RELAY_BILLING_SERVICES.TTS,
      lease: directRequestLease,
    });

    console.log(
      `🎧 Synthesizing ${segments.length} segments (${totalCharacters} chars) with ${ttsProvider}...`,
    );

    let result: any;
    let ttsModel = model;

    try {
      if (ttsProvider === "elevenlabs") {
        // Use ElevenLabs
        const elevenLabsKey = process.env.ELEVENLABS_API_KEY;
        if (!elevenLabsKey) {
          await releaseRelayCredits({
            cfApiBase: CF_API_BASE,
            relaySecret: RELAY_SECRET,
            payload: {
              deviceId,
              requestKey,
              service: STAGE5_RELAY_BILLING_SERVICES.TTS,
            },
          });
          sendReplayError(500, "ElevenLabs not configured");
          return;
        }

        ttsModel = ELEVENLABS_TTS_MODEL_ID;
        const segmentResults: Array<{
          index: number;
          audioBase64: string;
          targetDuration?: number;
        }> = [];

        for (const seg of segments) {
          throwIfRequestAborted();
          const text = seg.text || seg.translation || "";
          if (!text.trim()) continue;

          const audioBuffer = await synthesizeWithElevenLabs({
            text,
            voice,
            modelId: ELEVENLABS_TTS_MODEL_ID,
            format,
            apiKey: elevenLabsKey,
            signal: requestAbortController.signal,
          });
          segmentResults.push({
            index: seg.index ?? segmentResults.length,
            audioBase64: audioBuffer.toString("base64"),
            targetDuration: seg.targetDuration,
          });
        }

        result = {
          segments: segmentResults,
          format,
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
          await releaseRelayCredits({
            cfApiBase: CF_API_BASE,
            relaySecret: RELAY_SECRET,
            payload: {
              deviceId,
              requestKey,
              service: STAGE5_RELAY_BILLING_SERVICES.TTS,
            },
          });
          sendReplayError(500, "OpenAI not configured");
          return;
        }

        const client = makeOpenAI(openaiKey);
        const segmentResults: Array<{
          index: number;
          audioBase64: string;
          targetDuration?: number;
        }> = [];

        console.log(
          `   DEBUG: About to process ${segments.length} segments for OpenAI TTS`,
        );
        console.log(`   DEBUG: First segment:`, JSON.stringify(segments[0]));

        for (const seg of segments) {
          throwIfRequestAborted();
          const text = seg.text || seg.translation || "";
          if (!text.trim()) continue;

          console.log(
            `   • OpenAI TTS: voice=${voice}, model=${model}, format=${format}, text="${text.slice(0, 30)}..."`,
          );

          try {
            const ttsRes = await client.audio.speech.create(
              {
                model,
                voice: voice as any,
                input: text,
                response_format: format as any,
              },
              { signal: requestAbortController.signal },
            );

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
              ttsErr?.response?.data || "",
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
    } catch (ttsError: any) {
      if (
        isRequestAborted() ||
        ttsError?.name === "AbortError" ||
        String(ttsError?.message || "").includes("Request cancelled")
      ) {
        await releaseRelayCredits({
          cfApiBase: CF_API_BASE,
          relaySecret: RELAY_SECRET,
          payload: {
            deviceId,
            requestKey,
            service: STAGE5_RELAY_BILLING_SERVICES.TTS,
            meta: { reason: "client-cancelled" },
          },
        });
        sendReplayError(408, "Request cancelled", "Request was cancelled");
        return;
      }

      await releaseRelayCredits({
        cfApiBase: CF_API_BASE,
        relaySecret: RELAY_SECRET,
        payload: {
          deviceId,
          requestKey,
          service: STAGE5_RELAY_BILLING_SERVICES.TTS,
          meta: {
            reason: "vendor-error",
            message: ttsError?.message || String(ttsError),
          },
        },
      });
      throw ttsError;
    }

    console.log(`🎯 TTS complete! ${result.segmentCount} segments`);

    const replaySuccess: DirectDubbingReplayResult = {
      kind: "success",
      status: 200,
      data: result,
    };
    const pendingFinalize: PendingDirectDubbingFinalize = {
      characters: totalCharacters,
      model: ttsModel,
    };
    const storedReplayResult = await storeSuccessDirectReplayArtifact({
      cfApiBase: CF_API_BASE,
      relaySecret: RELAY_SECRET,
      deviceId,
      requestKey,
      service: STAGE5_RELAY_BILLING_SERVICES.TTS,
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
      service: STAGE5_RELAY_BILLING_SERVICES.TTS,
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

    if (isRequestAborted()) {
      console.warn(
        "⚠️ /dub-direct cancelled after replay persistence; leaving pending finalize for retry recovery",
      );
      sendReplayError(408, "Request cancelled", "Request was cancelled");
      return;
    }

    console.log(`💳 Finalizing credits for ${totalCharacters} characters...`);
    try {
      const finalizeResult = await finalizeRelayCredits({
        cfApiBase: CF_API_BASE,
        relaySecret: RELAY_SECRET,
        payload: {
          deviceId,
          requestKey,
          service: STAGE5_RELAY_BILLING_SERVICES.TTS,
          characters: totalCharacters,
          model: ttsModel,
          meta: {
            ...buildStoredDirectDubbingReplayMeta(
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
    sendReplaySuccess(result);
  } catch (error: any) {
    console.error("❌ Dub error:", error.message);
    if (replayContext) {
      const replay: DirectDubbingReplayResult = {
        kind: "error",
        status: 500,
        error: "Dub synthesis failed",
        details: error.message,
      };
      settleDirectDubbingReplayEntry({
        requestKey: replayContext.requestKey,
        entry: replayContext.entry,
        result: replay,
      });
      sendDirectDubbingReplay(res, replay, sendError, sendJson);
      return;
    }

    sendError(res, 500, "Dub synthesis failed", error.message);
  } finally {
    disconnectWatcher.cleanup();
    stopLeaseHeartbeat?.();
  }
}

async function handleDubElevenLabs(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext,
): Promise<void> {
  console.log("🎬 Processing ElevenLabs TTS dub request...");

  const {
    CF_API_BASE,
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
    sendError(
      res,
      confirmResult.status,
      confirmResult.error || "Reservation confirmation failed",
    );
    return;
  }

  const elevenLabsKey = getHeader(req, "x-elevenlabs-key");
  if (!elevenLabsKey) {
    console.log("❌ ElevenLabs API key not provided");
    sendError(res, 500, "ElevenLabs not configured");
    return;
  }

  const requestAbortController = new AbortController();
  const disconnectWatcher = watchClientDisconnect(req, res, () => {
    requestAbortController.abort();
  });

  try {
    const parsed = await readJsonBody(req);
    const segmentsPayload = Array.isArray(parsed?.segments)
      ? parsed.segments
          .map((segment: any, idx: number) => {
            const text =
              typeof segment?.text === "string" ? segment.text.trim() : "";
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

    const oversizedSegment = findOversizedElevenLabsSegment(segmentsPayload);
    if (oversizedSegment) {
      sendError(
        res,
        413,
        "Segment too long",
        `Segment ${oversizedSegment.index} has ${oversizedSegment.length} characters. ElevenLabs v3 accepts at most ${ELEVENLABS_TTS_MAX_TEXT_CHARACTERS} characters per segment.`,
      );
      return;
    }

    const voice = parsed?.voice || "adam";
    const format =
      typeof parsed?.format === "string" && parsed.format.trim()
        ? parsed.format.trim()
        : "mp3";
    const totalCharacters = segmentsPayload.reduce(
      (sum: number, seg: any) => sum + seg.text.length,
      0,
    );

    console.log(
      `🎧 Synthesizing ${segmentsPayload.length} segments (${totalCharacters} chars) with ElevenLabs voice=${voice} format=${format}`,
    );

    const segmentResponses: Array<{
      index: number;
      audioBase64: string;
      targetDuration?: number;
    }> = [];

    // Process segments with concurrency limit
    const CONCURRENCY = 3;
    for (let i = 0; i < segmentsPayload.length; i += CONCURRENCY) {
      if (
        disconnectWatcher.isDisconnected() ||
        requestAbortController.signal.aborted
      ) {
        console.warn(
          "⚠️ /dub-elevenlabs aborted by upstream client during synthesis",
        );
        return;
      }
      const batch = segmentsPayload.slice(i, i + CONCURRENCY);
      const results = await Promise.all(
        batch.map(async (seg: any) => {
          if (
            disconnectWatcher.isDisconnected() ||
            requestAbortController.signal.aborted
          ) {
            throw new Error("Client disconnected");
          }
          const audioBuffer = await synthesizeWithElevenLabs({
            text: seg.text,
            voice,
            format,
            apiKey: elevenLabsKey,
            signal: requestAbortController.signal,
          });
          return {
            index: seg.index,
            audioBase64: audioBuffer.toString("base64"),
            targetDuration: seg.targetDuration,
          };
        }),
      );
      segmentResponses.push(...results);
      console.log(
        `   • Completed ${Math.min(i + CONCURRENCY, segmentsPayload.length)}/${segmentsPayload.length} segments`,
      );
    }

    sendJson(res, {
      voice,
      model: ELEVENLABS_TTS_MODEL_ID,
      format,
      segmentCount: segmentResponses.length,
      totalCharacters,
      segments: segmentResponses,
    });
  } catch (error: any) {
    if (
      error?.name === "AbortError" ||
      String(error?.message || "").includes("Client disconnected")
    ) {
      console.warn("⚠️ /dub-elevenlabs aborted by upstream client");
      return;
    }
    console.error("❌ ElevenLabs TTS error:", error.message);
    sendError(res, 500, "Dub synthesis failed", error.message);
  } finally {
    disconnectWatcher.cleanup();
  }
}

async function handleDub(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext,
): Promise<void> {
  console.log("🎬 Processing dub synthesis request...");

  const {
    CF_API_BASE,
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
    sendError(
      res,
      confirmResult.status,
      confirmResult.error || "Reservation confirmation failed",
    );
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
            const rawText =
              typeof segment?.text === "string" ? segment.text : "";
            const text = rawText.trim();
            if (!text) {
              return null;
            }
            const index = Number.isFinite(segment?.index)
              ? Number(segment.index)
              : idx + 1;
            const start =
              typeof segment?.start === "number" &&
              Number.isFinite(segment.start)
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
              seg: any,
            ): seg is {
              index: number;
              text: string;
              start?: number;
              end?: number;
              targetDuration?: number;
            } => Boolean(seg),
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
          `Received ${segmentsPayload.length} segments, limit is ${DUB_MAX_SEGMENTS}`,
        );
        return;
      }

      const totalCharacters = segmentsPayload.reduce(
        (sum: number, seg: { text: string }) => sum + seg.text.length,
        0,
      );

      if (totalCharacters > DUB_MAX_TOTAL_CHARACTERS) {
        sendError(
          res,
          413,
          "Dub request too large",
          `Received ${totalCharacters} characters, limit is ${DUB_MAX_TOTAL_CHARACTERS}`,
        );
        return;
      }

      console.log(
        `🎧 Synthesizing ${segmentsPayload.length} segment(s) (${totalCharacters} chars) model=${model} voice=${voice} format=${format}`,
      );

      const segmentResponses: Array<{
        index: number;
        audioBase64: string;
        targetDuration?: number;
      }> = new Array(segmentsPayload.length);

      const segmentAbortControllers = new Map<number, AbortController>();
      const disconnectWatcher = watchClientDisconnect(req, res, () => {
        for (const controller of segmentAbortControllers.values()) {
          controller.abort();
        }
      });
      try {
        const synthesizeSegment = async (segIdx: number) => {
          const seg = segmentsPayload[segIdx];
          let attempt = 0;
          const abortController = new AbortController();
          segmentAbortControllers.set(segIdx, abortController);

          try {
            while (attempt < DUB_MAX_RETRIES) {
              if (disconnectWatcher.isDisconnected()) {
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
                  { signal: abortController.signal },
                );
                const arrayBuffer = await speech.arrayBuffer();
                segmentResponses[segIdx] = {
                  index: seg.index,
                  audioBase64: Buffer.from(arrayBuffer).toString("base64"),
                  targetDuration: seg.targetDuration,
                };
                return;
              } catch (segmentError: any) {
                if (
                  disconnectWatcher.isDisconnected() ||
                  abortController.signal.aborted
                ) {
                  throw segmentError;
                }

                if (
                  attempt >= DUB_MAX_RETRIES ||
                  !shouldRetrySegmentError(segmentError)
                ) {
                  throw segmentError;
                }

                const delay = Math.min(
                  DUB_RETRY_MAX_DELAY_MS,
                  DUB_RETRY_BASE_DELAY_MS * Math.pow(2, attempt - 1),
                );
                console.warn(
                  `⚠️ Segment ${segIdx + 1}/${segmentsPayload.length} retry ${attempt}/${DUB_MAX_RETRIES} in ${delay}ms:`,
                  segmentError?.message || segmentError,
                );
                await sleep(delay);
              }
            }

            throw new Error(
              `Segment ${segIdx + 1} exhausted retries without completion`,
            );
          } finally {
            segmentAbortControllers.delete(segIdx);
          }
        };

        // Use a queue to distribute work safely across workers
        const pendingIndices = segmentsPayload.map((_: any, i: number) => i);
        const workerCount = Math.min(
          DUB_MAX_CONCURRENCY,
          segmentsPayload.length,
        );
        const claimNextPendingIndex = (): number | undefined => {
          if (disconnectWatcher.isDisconnected()) {
            return undefined;
          }
          return pendingIndices.shift();
        };

        const workers = Array.from(
          { length: workerCount },
          async (_, workerIdx) => {
            for (
              let current = claimNextPendingIndex();
              current !== undefined;
              current = claimNextPendingIndex()
            ) {
              const seg = segmentsPayload[current];
              console.log(
                `   • Worker ${workerIdx + 1}/${workerCount} segment ${
                  current + 1
                }/${segmentsPayload.length} (index=${seg.index}, ${seg.text.length} chars)`,
              );
              await synthesizeSegment(current);
              console.log(
                `     · Completed segment ${current + 1}/${segmentsPayload.length}`,
              );
            }
          },
        );

        try {
          await Promise.all(workers);
        } catch (segmentError: any) {
          if (disconnectWatcher.isDisconnected()) {
            console.warn(
              "⚠️ Dub request aborted by upstream client while synthesizing segments",
            );
            return;
          }

          const details = segmentError?.response?.data ?? segmentError?.message;
          console.error("❌ Relay segment synthesis failed:", details);
          sendError(
            res,
            500,
            "Dub synthesis failed",
            typeof details === "string" ? details : JSON.stringify(details),
          );
          return;
        }

        if (disconnectWatcher.isDisconnected()) {
          console.warn(
            "⚠️ Dub request closed before completion; skipping response",
          );
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
      } finally {
        disconnectWatcher.cleanup();
      }
    }

    if (!lines.length) {
      sendError(res, 400, "Invalid request: lines array required");
      return;
    }

    const totalCharacters = lines.reduce(
      (sum: number, line: string) => sum + line.length,
      0,
    );
    const chunks = chunkLines(lines, MAX_TTS_CHARS_PER_CHUNK);

    if (!chunks.length) {
      sendError(res, 400, "No valid dialogue for dubbing");
      return;
    }

    console.log(
      `🎧 Synthesizing ${chunks.length} chunk(s) (${totalCharacters} chars) model=${model} voice=${voice} format=${format}`,
    );

    const chunkBuffers: Buffer[] = [];

    for (let idx = 0; idx < chunks.length; idx++) {
      const chunk = chunks[idx];
      console.log(
        `   • Chunk ${idx + 1}/${chunks.length} (${chunk.length} chars)`,
      );
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
    sendError(
      res,
      500,
      "Dub synthesis failed",
      error?.message || String(error),
    );
  }
}

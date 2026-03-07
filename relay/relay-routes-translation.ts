import { randomUUID } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { DurableRelayJob } from "./relay-job-sync.js";
import type { RelayRoutesContext, RelayTranslationJob } from "./relay-routes.js";
import {
  createDirectRequestLease,
  normalizeRelayRecoveryFailureStatus,
  persistDirectReplayOrRelease,
  recoverOrRestartDuplicateReservation,
  startDirectRequestLeaseHeartbeat,
} from "./direct-replay-recovery.js";
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
import {
  estimateTranslationCompletionTokensFallback,
  estimateTranslationPromptTokenReserve,
} from "./translation-token-estimator.js";

type DirectTranslationReplayResult =
  | { kind: "success"; status: number; data: unknown }
  | { kind: "error"; status: number; error: string; details?: string };

type DirectTranslationReplayEntry = {
  done: boolean;
  promise: Promise<DirectTranslationReplayResult>;
  resolve: (result: DirectTranslationReplayResult) => void;
  result?: DirectTranslationReplayResult;
  expiresAt?: number;
};

type PendingDirectTranslationFinalize = {
  promptTokens: number;
  completionTokens: number;
  model: string;
  webSearchCalls: number;
};

const DIRECT_TRANSLATION_REPLAY_TTL_MS = Math.max(
  1_000,
  Number.parseInt(
    process.env.DIRECT_TRANSLATION_REPLAY_TTL_MS || String(10 * 60 * 1_000),
    10,
  ),
);
const directTranslationReplayCache = new Map<
  string,
  DirectTranslationReplayEntry
>();

function createDirectTranslationReplayEntry(): DirectTranslationReplayEntry {
  let resolve!: (result: DirectTranslationReplayResult) => void;
  const promise = new Promise<DirectTranslationReplayResult>((innerResolve) => {
    resolve = innerResolve;
  });
  return {
    done: false,
    promise,
    resolve,
  };
}

function pruneDirectTranslationReplayCache(now = Date.now()): void {
  for (const [requestKey, entry] of directTranslationReplayCache.entries()) {
    if (
      entry.done &&
      entry.result?.kind === "success" &&
      typeof entry.expiresAt === "number" &&
      entry.expiresAt <= now
    ) {
      directTranslationReplayCache.delete(requestKey);
    }
  }
}

function settleDirectTranslationReplayEntry({
  requestKey,
  entry,
  result,
  cacheSuccess = false,
}: {
  requestKey: string;
  entry: DirectTranslationReplayEntry;
  result: DirectTranslationReplayResult;
  cacheSuccess?: boolean;
}): void {
  if (entry.done) {
    return;
  }

  entry.done = true;
  entry.result = result;
  entry.resolve(result);

  if (cacheSuccess && result.kind === "success") {
    entry.expiresAt = Date.now() + DIRECT_TRANSLATION_REPLAY_TTL_MS;
    directTranslationReplayCache.set(requestKey, entry);
    return;
  }

  directTranslationReplayCache.delete(requestKey);
}

function sendDirectTranslationReplay(
  res: ServerResponse,
  replay: DirectTranslationReplayResult,
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

function extractPersistedDirectTranslationReplay(
  reservationMeta: unknown,
): DirectTranslationReplayResult | null {
  const metaObject = asObject(reservationMeta);
  const replayObject = asObject(metaObject?.directReplayResult);
  if (!replayObject) {
    return null;
  }

  const kind = replayObject.kind;
  const status = replayObject.status;
  if (
    (kind !== "success" && kind !== "error") ||
    typeof status !== "number" ||
    !Number.isFinite(status)
  ) {
    return null;
  }

  if (kind === "success") {
    return {
      kind,
      status,
      data: replayObject.data,
    };
  }

  return {
    kind,
    status,
    error:
      typeof replayObject.error === "string" && replayObject.error.trim()
        ? replayObject.error.trim()
        : "Translation failed",
    ...(typeof replayObject.details === "string" && replayObject.details.trim()
      ? { details: replayObject.details.trim() }
      : {}),
  };
}

function extractPendingDirectTranslationFinalize(
  reservationMeta: unknown,
): PendingDirectTranslationFinalize | null {
  const metaObject = asObject(reservationMeta);
  const pendingObject = asObject(metaObject?.pendingFinalize);
  if (!pendingObject) {
    return null;
  }

  const promptTokens = Number(pendingObject.promptTokens);
  const completionTokens = Number(pendingObject.completionTokens);
  const webSearchCalls = Number(pendingObject.webSearchCalls);
  const model =
    typeof pendingObject.model === "string" ? pendingObject.model.trim() : "";

  if (
    !model ||
    !Number.isFinite(promptTokens) ||
    promptTokens < 0 ||
    !Number.isFinite(completionTokens) ||
    completionTokens < 0 ||
    !Number.isFinite(webSearchCalls) ||
    webSearchCalls < 0
  ) {
    return null;
  }

  return {
    promptTokens,
    completionTokens,
    model,
    webSearchCalls,
  };
}

async function recoverReservedDirectTranslationReplay({
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
  | { kind: "replay"; replay: DirectTranslationReplayResult }
  | { kind: "error"; status: number; error: string; details?: string }
  | null
> {
  const replay = extractPersistedDirectTranslationReplay(reservationMeta);
  const pendingFinalize = extractPendingDirectTranslationFinalize(reservationMeta);
  if (!replay || !pendingFinalize) {
    return null;
  }

  const finalizeResult = await finalizeRelayCredits({
    cfApiBase,
    relaySecret,
    payload: {
      deviceId,
      requestKey,
      service: STAGE5_RELAY_BILLING_SERVICES.TRANSLATION,
      promptTokens: pendingFinalize.promptTokens,
      completionTokens: pendingFinalize.completionTokens,
      webSearchCalls: pendingFinalize.webSearchCalls,
      model: pendingFinalize.model,
      meta: {
        directReplayResult: replay,
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

export async function handleTranslationRoutes(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext,
): Promise<boolean> {
  if (req.method === "POST" && req.url === "/translate-direct") {
    await handleTranslateDirect(req, res, ctx);
    return true;
  }

  if (req.method === "POST" && req.url === "/translate") {
    await handleTranslate(req, res, ctx);
    return true;
  }

  if (
    req.method === "GET" &&
    req.url &&
    req.url.startsWith("/translate/result/")
  ) {
    await handleTranslateResult(req, res, ctx);
    return true;
  }

  return false;
}

async function handleTranslateDirect(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext,
): Promise<void> {
  console.log("📡 Processing direct translation request...");

  const {
    CF_API_BASE,
    RELAY_SECRET,
    DEFAULT_TRANSLATION_MODEL,
    getHeader,
    sendError,
    sendJson,
    readJsonBody,
    parseTranslationPhase,
    parseTranslationModelFamily,
    parseBooleanLike,
    resolveTranslationModel,
    resolveTranslationReservationMaxCompletionTokens,
    isAllowedStage5TranslationModel,
    isClaudeModel,
    translateWithClaude,
    translateWithClaudeWebSearch,
    translateWithOpenAiWebSearch,
    makeOpenAI,
  } = ctx;

  // Get API key from header (app sends its Stage5 API key)
  const apiKey = getHeader(req, "authorization")?.replace(/^Bearer\s+/i, "");
  if (!apiKey) {
    console.log("❌ Missing API key for /translate-direct");
    sendError(res, 401, "Unauthorized - missing API key");
    return;
  }

  const idempotencyKey =
    getHeader(req, "idempotency-key") || getHeader(req, "x-idempotency-key");
  const appVersion = getHeader(req, "x-stage5-app-version");

  // Step 1: Authorize with CF Worker
  console.log(`🔐 Authorizing with CF Worker...`);

  const authResult = await authorizeRelayDevice({
    cfApiBase: CF_API_BASE,
    relaySecret: RELAY_SECRET,
    apiKey,
    service: STAGE5_RELAY_BILLING_SERVICES.TRANSLATION,
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

  // Step 2: Parse request and call OpenAI/Anthropic
  let replayContext:
    | { requestKey: string; entry: DirectTranslationReplayEntry }
    | null = null;
  let stopLeaseHeartbeat: (() => void) | null = null;
  try {
    const parsed = await readJsonBody(req);
    const messages = parsed?.messages;
    const translationPhase = parseTranslationPhase(parsed?.translationPhase);
    const modelFamily = parseTranslationModelFamily(parsed?.modelFamily);
    const qualityMode =
      typeof parsed?.qualityMode === "boolean" ? parsed.qualityMode : undefined;
    const model = resolveTranslationModel({
      rawModel:
        typeof parsed?.model === "string" && parsed.model.trim()
          ? parsed.model.trim()
          : DEFAULT_TRANSLATION_MODEL,
      messages: Array.isArray(messages) ? messages : undefined,
      canUseAnthropic: Boolean(process.env.ANTHROPIC_API_KEY),
      modelFamily,
      translationPhase,
      qualityMode,
    });
    const reasoning = translationPhase === "review" ? undefined : parsed?.reasoning;
    const useWebSearch = parseBooleanLike(parsed?.webSearch) === true;
    const maxCompletionTokens = resolveTranslationReservationMaxCompletionTokens({
      model,
      reasoning,
    });

    if (!isAllowedStage5TranslationModel(model)) {
      sendError(res, 400, `Unsupported translation model: ${model}`);
      return;
    }

    if (!Array.isArray(messages) || messages.length === 0) {
      sendError(res, 400, "Messages array required");
      return;
    }

    const requestKey = buildRelayRequestKey({
      service: STAGE5_RELAY_BILLING_SERVICES.TRANSLATION,
      deviceId,
      clientIdempotencyKey: idempotencyKey,
      payload: {
        messages,
        translationPhase,
        modelFamily,
        qualityMode,
        model,
        reasoning,
        webSearch: useWebSearch,
      },
    });
    pruneDirectTranslationReplayCache();
    const existingReplay = directTranslationReplayCache.get(requestKey);
    if (existingReplay) {
      const replay = existingReplay.result ?? await existingReplay.promise;
      sendDirectTranslationReplay(res, replay, sendError, sendJson);
      return;
    }

    const replayEntry = createDirectTranslationReplayEntry();
    directTranslationReplayCache.set(requestKey, replayEntry);
    replayContext = { requestKey, entry: replayEntry };
    const sendReplaySuccess = (data: unknown, status = 200): void => {
      const replay: DirectTranslationReplayResult = {
        kind: "success",
        status,
        data,
      };
      settleDirectTranslationReplayEntry({
        requestKey,
        entry: replayEntry,
        result: replay,
        cacheSuccess: true,
      });
      sendDirectTranslationReplay(res, replay, sendError, sendJson);
    };
    const sendReplayError = (
      status: number,
      error: string,
      details?: string,
    ): void => {
      const replay: DirectTranslationReplayResult = {
        kind: "error",
        status,
        error,
        ...(details ? { details } : {}),
      };
      settleDirectTranslationReplayEntry({
        requestKey,
        entry: replayEntry,
        result: replay,
      });
      sendDirectTranslationReplay(res, replay, sendError, sendJson);
    };
    const promptTokenReserve = estimateTranslationPromptTokenReserve({
      model,
      messages,
    });
    const directRequestLease = createDirectRequestLease();
    let reserveResult: Awaited<ReturnType<typeof reserveRelayCredits>> | null = null;
    for (let orphanRecoveryAttempt = 0; orphanRecoveryAttempt < 2; orphanRecoveryAttempt += 1) {
      reserveResult = await reserveRelayCredits({
        cfApiBase: CF_API_BASE,
        relaySecret: RELAY_SECRET,
        payload: {
          deviceId,
          requestKey,
          service: STAGE5_RELAY_BILLING_SERVICES.TRANSLATION,
          promptTokens: promptTokenReserve,
          maxCompletionTokens,
          webSearchCalls: useWebSearch ? 1 : 0,
          model,
          meta: {
            directRequestLease,
          },
        },
      });
      if (!reserveResult.ok) {
        const status = reserveResult.status === 402 ? 402 : reserveResult.status;
        sendReplayError(
          status,
          reserveResult.error || "Credit reservation failed",
        );
        return;
      }
      if (reserveResult.status !== "duplicate") {
        break;
      }

      if (reserveResult.reservationStatus === "reserved") {
        const recovered = await recoverReservedDirectTranslationReplay({
          cfApiBase: CF_API_BASE,
          relaySecret: RELAY_SECRET,
          deviceId,
          requestKey,
          reservationMeta: reserveResult.reservationMeta,
        });
        if (recovered?.kind === "replay") {
          settleDirectTranslationReplayEntry({
            requestKey,
            entry: replayEntry,
            result: recovered.replay,
            cacheSuccess: recovered.replay.kind === "success",
          });
          sendDirectTranslationReplay(res, recovered.replay, sendError, sendJson);
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
          service: STAGE5_RELAY_BILLING_SERVICES.TRANSLATION,
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
          const settledReplay = extractPersistedDirectTranslationReplay(
            orphanRecovery.reservationMeta,
          );
          if (settledReplay) {
            settleDirectTranslationReplayEntry({
              requestKey,
              entry: replayEntry,
              result: settledReplay,
              cacheSuccess: settledReplay.kind === "success",
            });
            sendDirectTranslationReplay(res, settledReplay, sendError, sendJson);
            return;
          }
          sendReplayError(409, "Duplicate request already completed");
          return;
        }
        if (orphanRecovery.action === "retry-reserve") {
          continue;
        }
      }

      const persistedReplay = extractPersistedDirectTranslationReplay(
        reserveResult.reservationMeta,
      );
      if (persistedReplay) {
        settleDirectTranslationReplayEntry({
          requestKey,
          entry: replayEntry,
          result: persistedReplay,
          cacheSuccess: persistedReplay.kind === "success",
        });
        sendDirectTranslationReplay(res, persistedReplay, sendError, sendJson);
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
      service: STAGE5_RELAY_BILLING_SERVICES.TRANSLATION,
      lease: directRequestLease,
    });

    console.log(
      `🌐 Translating with ${model}${useWebSearch ? " (web search enabled)" : ""}...`,
    );

    const modelIsClaud = isClaudeModel(model);
    let result: any;
    let promptTokens = promptTokenReserve;
    let completionTokens = 0;

    try {
      if (modelIsClaud) {
        // Use Anthropic
        const anthropicKey = process.env.ANTHROPIC_API_KEY;
        if (!anthropicKey) {
          await releaseRelayCredits({
            cfApiBase: CF_API_BASE,
            relaySecret: RELAY_SECRET,
            payload: {
              deviceId,
              requestKey,
              service: STAGE5_RELAY_BILLING_SERVICES.TRANSLATION,
            },
          });
          sendReplayError(500, "Anthropic not configured");
          return;
        }

        const effort = reasoning?.effort as "low" | "medium" | "high" | undefined;
        if (useWebSearch) {
          const response = await translateWithClaudeWebSearch({
            messages: messages as any,
            model,
            apiKey: anthropicKey,
            maxTokens: maxCompletionTokens,
            effort,
          });
          result = {
            model: response.model,
            choices: [
              {
                message: {
                  role: "assistant",
                  content: response.content,
                },
              },
            ],
            usage: response.usage,
          };
          promptTokens = response.usage.prompt_tokens || 0;
          completionTokens = response.usage.completion_tokens || 0;
        } else {
          const response = await translateWithClaude({
            messages: messages as any,
            model,
            apiKey: anthropicKey,
            maxTokens: maxCompletionTokens,
            effort,
          });

          result = response;
          promptTokens = (response.usage as any)?.prompt_tokens || 0;
          completionTokens = (response.usage as any)?.completion_tokens || 0;
        }
      } else {
        // Use OpenAI
        const openaiKey = process.env.OPENAI_API_KEY;
        if (!openaiKey) {
          await releaseRelayCredits({
            cfApiBase: CF_API_BASE,
            relaySecret: RELAY_SECRET,
            payload: {
              deviceId,
              requestKey,
              service: STAGE5_RELAY_BILLING_SERVICES.TRANSLATION,
            },
          });
          sendReplayError(500, "OpenAI not configured");
          return;
        }

        if (useWebSearch) {
          const response = await translateWithOpenAiWebSearch({
            messages,
            model,
            apiKey: openaiKey,
            maxOutputTokens: maxCompletionTokens,
            reasoning,
          });
          result = {
            content: response.content,
            model: response.model,
            usage: response.usage,
          };
          promptTokens = response.usage.prompt_tokens || 0;
          completionTokens = response.usage.completion_tokens || 0;
        } else {
          const client = makeOpenAI(openaiKey);
          const response = await client.chat.completions.create({
            model,
            max_completion_tokens: maxCompletionTokens,
            messages: messages.map((m: any) => ({
              role: m.role,
              content: m.content,
            })),
          });

          result = {
            content: response.choices[0]?.message?.content || "",
            model,
            usage: response.usage,
          };
          promptTokens = response.usage?.prompt_tokens || 0;
          completionTokens = response.usage?.completion_tokens || 0;
        }
      }
    } catch (translationError: any) {
      await releaseRelayCredits({
        cfApiBase: CF_API_BASE,
        relaySecret: RELAY_SECRET,
        payload: {
          deviceId,
          requestKey,
          service: STAGE5_RELAY_BILLING_SERVICES.TRANSLATION,
          meta: { reason: "vendor-error", message: translationError?.message || String(translationError) },
        },
      });
      throw translationError;
    }

    promptTokens =
      Number.isFinite(promptTokens) && promptTokens > 0
        ? promptTokens
        : promptTokenReserve;
    completionTokens =
      Number.isFinite(completionTokens) && completionTokens > 0
        ? completionTokens
        : estimateTranslationCompletionTokensFallback({
            model:
              typeof result?.model === "string" && result.model.trim()
                ? result.model
                : model,
            completion: result,
            maxCompletionTokens,
          });

    console.log(
      `🎯 Translation complete! Tokens: ${promptTokens}+${completionTokens}`,
    );

    const replaySuccess: DirectTranslationReplayResult = {
      kind: "success",
      status: 200,
      data: result,
    };
    const pendingFinalize: PendingDirectTranslationFinalize = {
      promptTokens,
      completionTokens,
      webSearchCalls: useWebSearch ? 1 : 0,
      model,
    };
    const persistResult = await persistDirectReplayOrRelease({
      cfApiBase: CF_API_BASE,
      relaySecret: RELAY_SECRET,
      deviceId,
      requestKey,
      service: STAGE5_RELAY_BILLING_SERVICES.TRANSLATION,
      replayResult: replaySuccess,
      pendingFinalize,
    });
    if (!persistResult.ok) {
      sendReplayError(
        persistResult.status,
        persistResult.error || "Replay persistence failed",
        persistResult.details,
      );
      return;
    }
    stopLeaseHeartbeat?.();
    stopLeaseHeartbeat = null;

    console.log(`💳 Finalizing credits...`);
    try {
      const finalizeResult = await finalizeRelayCredits({
        cfApiBase: CF_API_BASE,
        relaySecret: RELAY_SECRET,
        payload: {
          deviceId,
          requestKey,
          service: STAGE5_RELAY_BILLING_SERVICES.TRANSLATION,
          promptTokens,
          completionTokens,
          webSearchCalls: useWebSearch ? 1 : 0,
          model,
          meta: {
            directReplayResult: replaySuccess,
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
    } catch (finalizeErr: any) {
      console.error("❌ Credit finalize request failed:", finalizeErr.message);
      sendReplayError(500, "Credit finalize failed", finalizeErr?.message);
      return;
    }

    // Return result to app (only after successful deduction)
    sendReplaySuccess(result);
  } catch (error: any) {
    console.error("❌ Translation error:", error.message);
    if (replayContext) {
      const replay: DirectTranslationReplayResult = {
        kind: "error",
        status: 500,
        error: "Translation failed",
        details: error.message,
      };
      settleDirectTranslationReplayEntry({
        requestKey: replayContext.requestKey,
        entry: replayContext.entry,
        result: replay,
      });
      sendDirectTranslationReplay(res, replay, sendError, sendJson);
      return;
    }

    sendError(res, 500, "Translation failed", error.message);
  } finally {
    stopLeaseHeartbeat?.();
  }
}

async function handleTranslate(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext,
): Promise<void> {
  console.log("🌐 Processing translate request (job)...");

  const {
    RELAY_SECRET,
    CF_API_BASE,
    DEFAULT_TRANSLATION_MODEL,
    getHeader,
    sendError,
    sendJson,
    validateRelaySecret,
    readJsonBody,
    parseTranslationPhase,
    parseTranslationModelFamily,
    resolveTranslationModel,
    isAllowedStage5TranslationModel,
    isClaudeModel,
    normalizeModelId,
    pruneTranslationJobs,
    translationJobs,
    upsertDurableRelayTranslationJob,
    processTranslationJob,
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
      service: STAGE5_RELAY_BILLING_SERVICES.TRANSLATION,
    },
  });
  if (!confirmResult.ok) {
    sendError(res, confirmResult.status, confirmResult.error || "Reservation confirmation failed");
    return;
  }

  const openaiKey = getHeader(req, "x-openai-key");
  const anthropicKey = getHeader(req, "x-anthropic-key");

  try {
    const parsed = await readJsonBody(req);

    let payload: any = null;
    if (Array.isArray(parsed?.messages)) {
      const messages = parsed.messages.map((m: any) => ({
        role: String(m?.role ?? ""),
        content: String(m?.content ?? ""),
      }));
      const translationPhase = parseTranslationPhase(parsed?.translationPhase);
      const modelFamily = parseTranslationModelFamily(parsed?.modelFamily);
      const qualityMode =
        typeof parsed?.qualityMode === "boolean"
          ? parsed.qualityMode
          : undefined;
      const model = resolveTranslationModel({
        rawModel:
          typeof parsed?.model === "string" && parsed.model.trim()
            ? parsed.model.trim()
            : DEFAULT_TRANSLATION_MODEL,
        messages,
        canUseAnthropic: Boolean(anthropicKey || process.env.ANTHROPIC_API_KEY),
        modelFamily,
        translationPhase,
        qualityMode,
      });
      payload = {
        mode: "chat",
        messages,
        model,
        modelFamily,
        reasoning: translationPhase === "review" ? undefined : parsed?.reasoning,
        translationPhase,
        qualityMode,
      };
    } else if (
      typeof parsed?.text === "string" &&
      parsed.text.trim() &&
      typeof parsed?.target_language === "string" &&
      parsed.target_language.trim()
    ) {
      const model = normalizeModelId(
        typeof parsed?.model === "string" && parsed.model.trim()
          ? parsed.model.trim()
          : DEFAULT_TRANSLATION_MODEL,
      );
      payload = {
        mode: "text",
        text: parsed.text,
        target_language: parsed.target_language,
        model,
      };
    }

    if (!payload) {
      sendError(
        res,
        400,
        "Invalid translation payload",
        "Expected messages[] or text/target_language",
      );
      return;
    }

    if (!isAllowedStage5TranslationModel(payload.model)) {
      sendError(res, 400, `Unsupported translation model: ${payload.model}`);
      return;
    }

    // Check for required API key based on model
    const modelIsClaud = isClaudeModel(payload.model);
    if (modelIsClaud && !anthropicKey && !process.env.ANTHROPIC_API_KEY) {
      console.log("❌ Missing Anthropic API key for Claude model");
      sendError(res, 401, "Unauthorized - missing Anthropic key");
      return;
    }
    if (!modelIsClaud && !openaiKey) {
      console.log("❌ Missing OpenAI API key for /translate");
      sendError(res, 401, "Unauthorized - missing OpenAI key");
      return;
    }

    pruneTranslationJobs();

    const job: RelayTranslationJob = {
      id: randomUUID(),
      status: "queued",
      createdAt: Date.now(),
      updatedAt: Date.now(),
      payload,
      openaiKey: openaiKey || "",
      anthropicKey: anthropicKey || undefined,
    };

    translationJobs.set(job.id, job);
    try {
      await upsertDurableRelayTranslationJob(
        CF_API_BASE,
        RELAY_SECRET,
        job.id,
        "queued",
      );
    } catch (persistErr: any) {
      translationJobs.delete(job.id);
      console.error(
        "❌ Failed to persist queued relay translation job:",
        persistErr?.message || persistErr,
      );
      sendError(
        res,
        503,
        "Translation job persistence unavailable",
        persistErr?.message || String(persistErr),
      );
      return;
    }

    sendJson(res, { jobId: job.id, status: job.status }, 202);

    setImmediate(() => {
      processTranslationJob(job)
        .catch((err: unknown) => {
          const errMessage = err instanceof Error ? err.message : String(err);
          job.error = {
            message: errMessage,
          };
          job.status = "failed";
          job.updatedAt = Date.now();
          console.error(
            "❌ Translation job processing error:",
            job.error.message,
          );
        })
        .finally(() => {
          pruneTranslationJobs();
        });
    });
  } catch (error: any) {
    console.error("❌ Relay translation job submission error:", error.message);
    sendError(res, 500, "Translation job submission failed", error.message);
  }
}

async function handleTranslateResult(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext,
): Promise<void> {
  const {
    RELAY_SECRET,
    CF_API_BASE,
    RELAY_TRANSLATION_STALE_MS,
    sendError,
    sendJson,
    validateRelaySecret,
    getDurableRelayTranslationJob,
    resolveRelayPollJob,
    translationJobs,
  } = ctx;

  if (!validateRelaySecret(req, RELAY_SECRET)) {
    sendError(res, 401, "Unauthorized - invalid relay secret");
    return;
  }

  const jobId = decodeURIComponent(
    req.url!.split("?")[0].split("/").pop() || "",
  );
  let durableJob: DurableRelayJob | null = null;
  let durableLookupFailed = false;
  try {
    durableJob = await getDurableRelayTranslationJob(
      CF_API_BASE,
      RELAY_SECRET,
      jobId,
    );
  } catch (error: any) {
    durableLookupFailed = true;
    console.error(
      "❌ Failed to read durable relay translation job:",
      error?.message || error,
    );
  }

  const resolvedJob = resolveRelayPollJob(jobId, durableJob, translationJobs);
  if (!resolvedJob) {
    if (durableLookupFailed) {
      res.writeHead(202, {
        "Content-Type": "application/json",
        "Retry-After": "2",
      });
      res.end(
        JSON.stringify({
          status: "processing",
          retryable: true,
          reason: "durable_lookup_unavailable",
        }),
      );
      return;
    }
    sendError(res, 404, "Job not found");
    return;
  }
  const { job, source: jobSource, updatedMs: jobUpdatedMs } = resolvedJob;

  if (job.status === "completed") {
    sendJson(res, job.result ?? {});
    return;
  }

  if (job.status === "failed") {
    const failedMessage =
      typeof job.error === "string"
        ? job.error
        : job.error?.message || (job as DurableRelayJob).error;
    sendError(
      res,
      500,
      failedMessage || "Translation failed"
    );
    return;
  }

  // If a non-terminal job remains queued/processing for too long, treat it
  // as missing so stage5-api can resubmit safely.
  if (
    jobUpdatedMs != null &&
    Date.now() - jobUpdatedMs > RELAY_TRANSLATION_STALE_MS
  ) {
    console.warn(
      `⚠️ Relay translation job ${jobId} is stale from ${jobSource} state (${Math.floor(
        (Date.now() - jobUpdatedMs) / 1000,
      )}s); returning 404 for resubmission`,
    );
    sendError(res, 404, "Job not found");
    return;
  }

  res.writeHead(202, {
    "Content-Type": "application/json",
    "Retry-After": "2",
  });
  res.end(JSON.stringify({ status: job.status }));
}

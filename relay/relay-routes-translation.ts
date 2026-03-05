import { randomUUID } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { DurableRelayJob } from "./relay-job-sync.js";
import type { RelayRoutesContext, RelayTranslationJob } from "./relay-routes.js";
import {
  authorizeRelayDevice,
  deductRelayCredits,
} from "./relay-billing-client.js";
import { STAGE5_RELAY_BILLING_SERVICES } from "./relay-billing-contract.js";

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

  // Step 2: Parse request and call OpenAI/Anthropic
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
    const reasoning = parsed?.reasoning;
    const useWebSearch = parseBooleanLike(parsed?.webSearch) === true;

    if (!isAllowedStage5TranslationModel(model)) {
      sendError(res, 400, `Unsupported translation model: ${model}`);
      return;
    }

    if (!Array.isArray(messages) || messages.length === 0) {
      sendError(res, 400, "Messages array required");
      return;
    }

    console.log(
      `🌐 Translating with ${model}${useWebSearch ? " (web search enabled)" : ""}...`,
    );

    const modelIsClaud = isClaudeModel(model);
    let result: any;
    let promptTokens = 0;
    let completionTokens = 0;

    if (modelIsClaud) {
      // Use Anthropic
      const anthropicKey = process.env.ANTHROPIC_API_KEY;
      if (!anthropicKey) {
        sendError(res, 500, "Anthropic not configured");
        return;
      }

      const effort = reasoning?.effort as "low" | "medium" | "high" | undefined;
      if (useWebSearch) {
        const response = await translateWithClaudeWebSearch({
          messages: messages as any,
          model,
          apiKey: anthropicKey,
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
          effort,
        });

        result = response;
        // translateWithClaude returns { usage: { prompt_tokens, completion_tokens } }
        promptTokens = (response.usage as any)?.prompt_tokens || 0;
        completionTokens = (response.usage as any)?.completion_tokens || 0;
      }
    } else {
      // Use OpenAI
      const openaiKey = process.env.OPENAI_API_KEY;
      if (!openaiKey) {
        sendError(res, 500, "OpenAI not configured");
        return;
      }

      if (useWebSearch) {
        const response = await translateWithOpenAiWebSearch({
          messages,
          model,
          apiKey: openaiKey,
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

    console.log(
      `🎯 Translation complete! Tokens: ${promptTokens}+${completionTokens}`,
    );

    // Step 3: Deduct credits
    console.log(`💳 Deducting credits...`);
    try {
      const deductResult = await deductRelayCredits({
        cfApiBase: CF_API_BASE,
        relaySecret: RELAY_SECRET,
        payload: {
          deviceId,
          service: STAGE5_RELAY_BILLING_SERVICES.TRANSLATION,
          promptTokens,
          completionTokens,
          model,
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
    sendJson(res, result);
  } catch (error: any) {
    console.error("❌ Translation error:", error.message);
    sendError(res, 500, "Translation failed", error.message);
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
        reasoning: parsed?.reasoning,
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

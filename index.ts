import { createServer } from "node:http";
import { Buffer } from "node:buffer";
import { makeOpenAI } from "./openai-config.js";
import { makeAnthropic, translateWithClaude } from "./anthropic-config.js";
import {
  CF_API_BASE,
  CLAUDE_OPUS_MODEL,
  DEFAULT_TRANSLATION_MODEL,
  isAllowedStage5TranslationModel,
  isClaudeModel,
  normalizeModelId,
} from "./constants.js";
import {
  transcribeWithScribe,
  synthesizeWithElevenLabs,
  dubWithElevenLabs,
} from "./elevenlabs-config.js";
import {
  getHeader,
  getCorsOrigin,
  sendError,
  sendJson,
  validateRelaySecret,
} from "./relay/relay-http.js";
import {
  getDurableRelayTranslationJob,
  type DurableRelayJob,
  type RelayJobStatus,
  resolveRelayPollJob,
  upsertDurableRelayTranslationJob,
} from "./relay/relay-job-sync.js";
import {
  isLikelySubtitleDraftMessages,
  isLikelySubtitleReviewMessages,
  parseBooleanLike,
  parseTranslationModelFamily,
  parseTranslationPhase,
  type TranslationModelFamily,
} from "./relay/relay-translation-helpers.js";
import {
  handleRelayRequest,
  type RelayRoutesContext,
  type RelayChatJobPayload,
  type RelayTextJobPayload,
} from "./relay/relay-routes.js";

const PORT = process.env.PORT || 3000;

const MAX_TTS_CHARS_PER_CHUNK = 3_500;
const DUB_MAX_CONCURRENCY = Math.max(
  1,
  Number.parseInt(process.env.DUB_MAX_CONCURRENCY || "4", 10),
);
const DUB_MAX_RETRIES = Math.max(
  1,
  Number.parseInt(process.env.DUB_MAX_RETRIES || "3", 10),
);
const DUB_RETRY_BASE_DELAY_MS = Math.max(
  100,
  Number.parseInt(process.env.DUB_RETRY_BASE_DELAY_MS || "500", 10),
);
const DUB_RETRY_MAX_DELAY_MS = Math.max(
  DUB_RETRY_BASE_DELAY_MS,
  Number.parseInt(process.env.DUB_RETRY_MAX_DELAY_MS || "4000", 10),
);
const DUB_MAX_TOTAL_CHARACTERS = Math.max(
  1,
  Number.parseInt(process.env.DUB_MAX_TOTAL_CHARACTERS || "90000", 10),
);
const DUB_MAX_SEGMENTS = Math.max(
  1,
  Number.parseInt(process.env.DUB_MAX_SEGMENTS || "240", 10),
);
const MAX_BODY_SIZE = 10 * 1024 * 1024; // 10MB limit for request bodies
const MAX_TRANSLATION_JOBS = 1000; // Memory leak prevention
const JOB_MAX_AGE_MS = 30 * 60 * 1000; // 30 minutes (reduced from 1 hour)
const R2_FETCH_TIMEOUT_MS = 120_000; // 2 minute timeout for R2 fetches
const RELAY_TRANSLATION_STALE_MS = Math.max(
  60_000,
  Number.parseInt(
    process.env.RELAY_TRANSLATION_STALE_MS || String(15 * 60 * 1000),
    10,
  ),
);
const WHISPER_TRANSCRIPTION_MODEL = "whisper-1";
const ELEVENLABS_TRANSCRIPTION_MODEL = "elevenlabs-scribe";
const WHISPER_MAX_FILE_SIZE_BYTES = Math.max(
  1,
  Number.parseInt(
    process.env.WHISPER_MAX_FILE_SIZE_BYTES || String(25 * 1024 * 1024),
    10,
  ),
);
const SCRIBE_MAX_RETRIES = Math.max(
  1,
  Number.parseInt(process.env.SCRIBE_MAX_RETRIES || "3", 10),
);
const SCRIBE_RETRY_BASE_DELAY_MS = Math.max(
  100,
  Number.parseInt(process.env.SCRIBE_RETRY_BASE_DELAY_MS || "600", 10),
);
const SCRIBE_RETRY_MAX_DELAY_MS = Math.max(
  SCRIBE_RETRY_BASE_DELAY_MS,
  Number.parseInt(process.env.SCRIBE_RETRY_MAX_DELAY_MS || "4000", 10),
);

// Fail fast if RELAY_SECRET is missing - this is critical for security
const RELAY_SECRET_ENV = process.env.RELAY_SECRET;
if (!RELAY_SECRET_ENV) {
  console.error("❌ FATAL: RELAY_SECRET environment variable is not set");
  process.exit(1);
}
const RELAY_SECRET: string = RELAY_SECRET_ENV;

// Allowed CORS origins (restrict in production)
const ALLOWED_ORIGINS = process.env.ALLOWED_ORIGINS
  ? process.env.ALLOWED_ORIGINS.split(",").map((o) => o.trim())
  : ["*"]; // Default to * for backwards compatibility

// Allowed R2 bucket hostnames for SSRF prevention
const ALLOWED_R2_HOSTS = process.env.ALLOWED_R2_HOSTS
  ? process.env.ALLOWED_R2_HOSTS.split(",").map((h) => h.trim().toLowerCase())
  : [];

type ChatJobPayload = RelayChatJobPayload;
type TextJobPayload = RelayTextJobPayload;
type TranslationJob = {
  id: string;
  status: RelayJobStatus;
  createdAt: number;
  updatedAt: number;
  payload: ChatJobPayload | TextJobPayload;
  openaiKey: string;
  anthropicKey?: string;
  result?: any;
  error?: { message: string; details?: string };
};

const translationJobs = new Map<string, TranslationJob>();

// ============================================================================
// Helper Functions (DRY consolidation)
// ============================================================================

/**
 * Validate R2 URL to prevent SSRF attacks.
 * Only allows URLs from configured R2 bucket hosts.
 */
function validateR2Url(urlString: string): { valid: boolean; error?: string } {
  try {
    const url = new URL(urlString);

    // Must be HTTPS
    if (url.protocol !== "https:") {
      return { valid: false, error: "R2 URL must use HTTPS" };
    }

    // If no allowed hosts configured, allow any (backwards compat)
    if (ALLOWED_R2_HOSTS.length === 0) {
      console.warn("⚠️ ALLOWED_R2_HOSTS not configured - allowing any R2 URL");
      return { valid: true };
    }

    // Check against whitelist
    const hostname = url.hostname.toLowerCase();
    if (
      !ALLOWED_R2_HOSTS.some(
        (h) => hostname === h || hostname.endsWith(`.${h}`),
      )
    ) {
      return {
        valid: false,
        error: `R2 URL hostname not in allowed list: ${hostname}`,
      };
    }

    return { valid: true };
  } catch {
    return { valid: false, error: "Invalid R2 URL format" };
  }
}

function resolveDirectTranscriptionQuality({
  explicitQualityRaw,
  modelHint,
  modelIdHint,
}: {
  explicitQualityRaw: unknown;
  modelHint?: string;
  modelIdHint?: string;
}): {
  useHighQuality: boolean;
  source: "explicit" | "model-hint" | "default";
} {
  const explicit = parseBooleanLike(explicitQualityRaw);
  if (typeof explicit === "boolean") {
    return { useHighQuality: explicit, source: "explicit" };
  }

  const combinedHints = [modelHint, modelIdHint]
    .map((v) => (typeof v === "string" ? v.trim().toLowerCase() : ""))
    .filter(Boolean)
    .join(" ");
  const hintTokens = combinedHints.split(/[^a-z0-9]+/).filter(Boolean);
  if (hintTokens.includes("whisper")) {
    return { useHighQuality: false, source: "model-hint" };
  }
  if (hintTokens.includes("scribe") || hintTokens.includes("elevenlabs")) {
    return { useHighQuality: true, source: "model-hint" };
  }

  return { useHighQuality: true, source: "default" };
}

function formatSizeMB(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(1);
}

function getWhisperFileSizeGuardMessage(fileSizeBytes: number): string | null {
  if (!Number.isFinite(fileSizeBytes) || fileSizeBytes <= 0) return null;
  if (fileSizeBytes <= WHISPER_MAX_FILE_SIZE_BYTES) return null;
  return `File is ${formatSizeMB(fileSizeBytes)}MB; Whisper supports up to ${formatSizeMB(WHISPER_MAX_FILE_SIZE_BYTES)}MB per request.`;
}

function toWhisperCompatibleScribeResult(result: any) {
  const segments = Array.isArray(result?.segments) ? result.segments : [];
  const duration =
    segments.length > 0
      ? Math.max(
          ...segments.map((segment: any) =>
            Number.isFinite(segment?.end) ? segment.end : 0,
          ),
        )
      : 0;

  return {
    text: String(result?.text ?? ""),
    language:
      typeof result?.language_code === "string"
        ? result.language_code
        : undefined,
    duration,
    approx_duration: duration,
    model: ELEVENLABS_TRANSCRIPTION_MODEL,
    segments: segments.map((segment: any, idx: number) => ({
      id: idx,
      start: Number.isFinite(segment?.start) ? segment.start : 0,
      end: Number.isFinite(segment?.end) ? segment.end : 0,
      text: String(segment?.text ?? ""),
      words: Array.isArray(segment?.words)
        ? segment.words.map((word: any) => ({
            word: String(word?.text ?? ""),
            start: Number.isFinite(word?.start) ? word.start : 0,
            end: Number.isFinite(word?.end) ? word.end : 0,
          }))
        : [],
    })),
    words: segments.flatMap((segment: any) =>
      Array.isArray(segment?.words)
        ? segment.words.map((word: any) => ({
            word: String(word?.text ?? ""),
            start: Number.isFinite(word?.start) ? word.start : 0,
            end: Number.isFinite(word?.end) ? word.end : 0,
          }))
        : [],
    ),
  };
}

async function transcribeWithWhisperFromPath({
  openaiKey,
  filePath,
  fileName,
  mimeType,
  language,
  prompt,
}: {
  openaiKey: string;
  filePath: string;
  fileName: string;
  mimeType: string;
  language?: string;
  prompt?: string;
}) {
  const client = makeOpenAI(openaiKey);
  const fs = await import("fs");
  const fileBuffer = await fs.promises.readFile(filePath);
  const fileBlob = new File([fileBuffer as unknown as BlobPart], fileName, {
    type: mimeType,
  });

  const whisperResult = (await client.audio.transcriptions.create({
    file: fileBlob,
    model: WHISPER_TRANSCRIPTION_MODEL,
    language: language || undefined,
    prompt: prompt || undefined,
    response_format: "verbose_json",
    timestamp_granularities: ["word", "segment"],
  })) as any;

  const durationFromSegments =
    Array.isArray(whisperResult?.segments) && whisperResult.segments.length > 0
      ? Math.max(
          ...whisperResult.segments.map((segment: any) =>
            Number.isFinite(segment?.end) ? segment.end : 0,
          ),
        )
      : 0;
  const duration =
    Number.isFinite(whisperResult?.duration) && whisperResult.duration > 0
      ? whisperResult.duration
      : durationFromSegments;

  return {
    ...whisperResult,
    model: WHISPER_TRANSCRIPTION_MODEL,
    duration,
    approx_duration:
      Number.isFinite(whisperResult?.approx_duration) &&
      whisperResult.approx_duration > 0
        ? whisperResult.approx_duration
        : duration,
  };
}

function resolveTranslationModel({
  rawModel,
  modelFamily,
  messages,
  canUseAnthropic,
  translationPhase,
  qualityMode,
}: {
  rawModel?: string;
  modelFamily?: TranslationModelFamily;
  messages?: Array<{ role: string; content: string }>;
  canUseAnthropic: boolean;
  translationPhase?: "draft" | "review";
  qualityMode?: boolean;
}): string {
  const normalized = normalizeModelId(rawModel || DEFAULT_TRANSLATION_MODEL);
  const requestedFamily =
    modelFamily && modelFamily !== "auto" ? modelFamily : undefined;
  const reviewByHeuristic = isLikelySubtitleReviewMessages(messages);
  const draftByHeuristic = isLikelySubtitleDraftMessages(messages);
  const isSubtitleWorkflow =
    translationPhase === "review" ||
    translationPhase === "draft" ||
    reviewByHeuristic ||
    draftByHeuristic;

  // Non-subtitle traffic keeps caller-selected model (e.g. summarization flows).
  if (!isSubtitleWorkflow) {
    return normalized;
  }

  const effectivePhase =
    translationPhase === "review"
      ? "review"
      : translationPhase === "draft"
        ? "draft"
        : qualityMode === false
          ? "draft"
          : reviewByHeuristic
            ? "review"
            : "draft";

  // Subtitle workflow in Stage5 credit mode is backend-authoritative:
  // - review => explicit model-family intent (gpt/claude), with Anthropic-aware fallback
  // - review (no family intent) => auto (Claude Opus when available, else GPT)
  // - draft => GPT
  const selectedModel =
    effectivePhase === "review"
      ? requestedFamily === "gpt"
        ? DEFAULT_TRANSLATION_MODEL
        : requestedFamily === "claude"
          ? canUseAnthropic
            ? CLAUDE_OPUS_MODEL
            : DEFAULT_TRANSLATION_MODEL
          : canUseAnthropic
            ? CLAUDE_OPUS_MODEL
            : DEFAULT_TRANSLATION_MODEL
      : DEFAULT_TRANSLATION_MODEL;

  if (selectedModel !== normalized) {
    console.log(
      `🧭 Server model authority selected ${selectedModel} (requested=${normalized}, phase=${
        translationPhase ?? effectivePhase
      }, modelFamily=${modelFamily ?? "auto"}, qualityMode=${
        typeof qualityMode === "boolean" ? qualityMode : "auto"
      })`,
    );
  }

  return selectedModel;
}

function mapMessagesToOpenAiResponsesInput(messages: any[]): any[] {
  return (messages || [])
    .map((msg: any) => {
      const roleRaw = typeof msg?.role === "string" ? msg.role : "user";
      const role =
        roleRaw === "system" || roleRaw === "assistant" || roleRaw === "user"
          ? roleRaw
          : "user";
      const text = typeof msg?.content === "string" ? msg.content.trim() : "";
      if (!text) return null;
      return {
        role,
        content: [
          {
            type: "input_text",
            text,
          },
        ],
      };
    })
    .filter(Boolean);
}

function extractTextFromOpenAiResponseObject(response: any): string {
  if (!response) return "";

  if (typeof response.output_text === "string" && response.output_text.trim()) {
    return response.output_text.trim();
  }

  if (Array.isArray(response.output_text)) {
    const joined = response.output_text
      .map((part: any) => (typeof part === "string" ? part : ""))
      .join("")
      .trim();
    if (joined) return joined;
  }

  if (Array.isArray(response.output)) {
    const chunks: string[] = [];
    for (const item of response.output) {
      const content = item?.content;
      if (!Array.isArray(content)) continue;
      for (const part of content) {
        if (typeof part?.text === "string" && part.text.trim()) {
          chunks.push(part.text.trim());
        }
      }
    }
    const joined = chunks.join("\n").trim();
    if (joined) return joined;
  }

  return "";
}

async function translateWithOpenAiWebSearch({
  messages,
  model,
  apiKey,
  reasoning,
}: {
  messages: any[];
  model: string;
  apiKey: string;
  reasoning?: { effort?: "low" | "medium" | "high" };
}): Promise<{
  model: string;
  content: string;
  usage: { prompt_tokens: number; completion_tokens: number };
}> {
  const client = makeOpenAI(apiKey);
  const payload: any = {
    model,
    input: mapMessagesToOpenAiResponsesInput(messages),
    tools: [{ type: "web_search_preview" }],
  };

  if (reasoning?.effort) {
    payload.reasoning = { effort: reasoning.effort };
  }

  const response: any = await client.responses.create(payload);
  const content = extractTextFromOpenAiResponseObject(response);
  if (!content) {
    throw new Error("OpenAI web search returned no text content.");
  }

  return {
    model: typeof response?.model === "string" ? response.model : model,
    content,
    usage: {
      prompt_tokens:
        Number(response?.usage?.input_tokens) ||
        Number(response?.usage?.prompt_tokens) ||
        0,
      completion_tokens:
        Number(response?.usage?.output_tokens) ||
        Number(response?.usage?.completion_tokens) ||
        0,
    },
  };
}

async function translateWithClaudeWebSearch({
  messages,
  model,
  apiKey,
  effort,
}: {
  messages: Array<{ role: string; content: string }>;
  model: string;
  apiKey: string;
  effort?: "low" | "medium" | "high";
}): Promise<{
  model: string;
  content: string;
  usage: { prompt_tokens: number; completion_tokens: number };
}> {
  const client = makeAnthropic(apiKey);
  let systemPrompt: string | undefined;
  const userMessages: Array<{ role: "user" | "assistant"; content: string }> =
    [];

  for (const msg of messages || []) {
    if (msg?.role === "system") {
      systemPrompt = msg?.content;
    } else if (msg?.role === "user" || msg?.role === "assistant") {
      userMessages.push({
        role: msg.role,
        content: String(msg?.content || ""),
      });
    }
  }

  if (userMessages.length === 0 || userMessages[0].role !== "user") {
    userMessages.unshift({ role: "user", content: "Please proceed." });
  }

  const thinkingBudget =
    effort === "high" ? 16_000 : effort === "medium" ? 8_000 : 0;
  const useExtendedThinking = thinkingBudget > 0;
  const requestParams: any = {
    model,
    max_tokens: useExtendedThinking ? 32_000 : 16_000,
    messages: userMessages,
    tools: [
      {
        type: "web_search_20250305",
        name: "web_search",
      },
    ],
  };

  if (systemPrompt && !useExtendedThinking) {
    requestParams.system = systemPrompt;
  } else if (systemPrompt && useExtendedThinking) {
    userMessages[0].content = `${systemPrompt}\n\n${userMessages[0].content}`;
  }

  if (useExtendedThinking) {
    requestParams.thinking = {
      type: "enabled",
      budget_tokens: thinkingBudget,
    };
  }

  const response: any = await client.messages.create(requestParams);
  let content = "";
  for (const block of response?.content || []) {
    if (block?.type === "text" && typeof block?.text === "string") {
      content += block.text;
    }
  }
  const normalizedContent = content.trim();
  if (!normalizedContent) {
    throw new Error("Anthropic web search returned no text content.");
  }

  return {
    model,
    content: normalizedContent,
    usage: {
      prompt_tokens: Number(response?.usage?.input_tokens) || 0,
      completion_tokens: Number(response?.usage?.output_tokens) || 0,
    },
  };
}

function readJsonBody(
  req: import("node:http").IncomingMessage,
): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    let body = "";
    let bodySize = 0;
    req.on("data", (chunk: Buffer) => {
      bodySize += chunk.length;
      if (bodySize > MAX_BODY_SIZE) {
        req.destroy();
        reject(new Error("Request body too large"));
        return;
      }
      body += chunk.toString();
    });
    req.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch (err) {
        reject(err);
      }
    });
    req.on("error", reject);
  });
}

function chunkLines(lines: string[], maxChars: number): string[] {
  const trimmed = lines.map((line) =>
    typeof line === "string" ? line.trim() : "",
  );
  const filtered = trimmed.filter(Boolean);
  const chunks: string[] = [];
  let current = "";

  const flush = () => {
    if (current.trim()) {
      chunks.push(current.trim());
    }
    current = "";
  };

  for (const line of filtered) {
    if (!line) continue;
    if (line.length > maxChars) {
      flush();
      for (let i = 0; i < line.length; i += maxChars) {
        const piece = line.slice(i, i + maxChars).trim();
        if (piece) {
          chunks.push(piece);
        }
      }
      continue;
    }

    if (!current) {
      current = line;
      continue;
    }

    if (current.length + 1 + line.length > maxChars) {
      flush();
      current = line;
    } else {
      current = `${current}\n${line}`;
    }
  }

  flush();
  return chunks;
}

function pruneTranslationJobs(maxAgeMs = JOB_MAX_AGE_MS) {
  const now = Date.now();

  // First pass: remove old completed/failed jobs
  for (const [jobId, job] of translationJobs) {
    if (
      (job.status === "completed" || job.status === "failed") &&
      now - job.createdAt > maxAgeMs
    ) {
      translationJobs.delete(jobId);
    }
  }

  // Memory leak prevention: if still over limit, remove oldest jobs
  if (translationJobs.size > MAX_TRANSLATION_JOBS) {
    const sortedJobs = [...translationJobs.entries()].sort(
      (a, b) => a[1].createdAt - b[1].createdAt,
    );

    const toRemove = sortedJobs.slice(
      0,
      translationJobs.size - MAX_TRANSLATION_JOBS,
    );
    for (const [jobId] of toRemove) {
      console.warn(
        `⚠️ Removing job ${jobId} due to job limit (${MAX_TRANSLATION_JOBS})`,
      );
      translationJobs.delete(jobId);
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function extractStatus(error: any): number | null {
  const direct =
    error?.status ?? error?.response?.status ?? error?.cause?.status;
  if (typeof direct === "number" && Number.isFinite(direct)) {
    return direct;
  }

  if (typeof direct === "string" && direct.trim()) {
    const parsed = Number.parseInt(direct, 10);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }

  if (typeof error?.message === "string") {
    const match = error.message.match(/\b(\d{3})\b/);
    if (match) {
      const parsed = Number.parseInt(match[1], 10);
      if (Number.isFinite(parsed)) {
        return parsed;
      }
    }
  }

  return null;
}

function shouldRetrySegmentError(error: any): boolean {
  const status = extractStatus(error);
  if (status != null) {
    if (status >= 200 && status < 400) {
      return false;
    }
    if ([408, 409, 425, 429, 500, 502, 503, 504, 522, 524].includes(status)) {
      return true;
    }
  }

  const message = String(error?.message || "").toLowerCase();
  if (
    /timeout|temporarily unavailable|connection reset|gateway|rate limit/.test(
      message,
    )
  ) {
    return true;
  }

  const code =
    typeof error?.code === "string" ? error.code.toUpperCase() : null;
  if (
    code &&
    [
      "ECONNRESET",
      "ETIMEDOUT",
      "EHOSTUNREACH",
      "ENETUNREACH",
      "ECONNABORTED",
    ].includes(code)
  ) {
    return true;
  }

  return status == null;
}

async function transcribeWithScribeWithRetries({
  filePath,
  apiKey,
  languageCode,
  idempotencyKey,
  contextLabel,
}: {
  filePath: string;
  apiKey: string;
  languageCode: string;
  idempotencyKey?: string;
  contextLabel: string;
}): Promise<{
  result: Awaited<ReturnType<typeof transcribeWithScribe>>;
  attempts: number;
}> {
  let lastError: any = null;
  let attempts = 0;

  for (let attempt = 1; attempt <= SCRIBE_MAX_RETRIES; attempt += 1) {
    attempts = attempt;
    try {
      const result = await transcribeWithScribe({
        filePath,
        apiKey,
        languageCode,
        idempotencyKey,
      });
      return { result, attempts };
    } catch (error: any) {
      lastError = error;
      const retryable = shouldRetrySegmentError(error);
      if (!retryable || attempt >= SCRIBE_MAX_RETRIES) {
        break;
      }

      const delay = Math.min(
        SCRIBE_RETRY_MAX_DELAY_MS,
        SCRIBE_RETRY_BASE_DELAY_MS * Math.pow(2, attempt - 1),
      );
      console.warn(
        `⚠️ ${contextLabel} ElevenLabs Scribe attempt ${attempt}/${SCRIBE_MAX_RETRIES} failed, retrying in ${delay}ms: ${
          error?.message || String(error)
        }`,
      );
      await sleep(delay);
    }
  }

  if (lastError && typeof lastError === "object") {
    (lastError as any).scribeAttempts = attempts;
  }
  throw (
    lastError ||
    new Error(`${contextLabel} ElevenLabs Scribe failed without a response`)
  );
}

async function processTranslationJob(job: TranslationJob): Promise<void> {
  job.status = "processing";
  job.updatedAt = Date.now();
  try {
    await upsertDurableRelayTranslationJob(
      CF_API_BASE,
      RELAY_SECRET,
      job.id,
      "processing",
    );
  } catch (persistErr: any) {
    console.error(
      "⚠️ Failed to persist relay translation job processing state:",
      persistErr?.message || persistErr,
    );
  }

  try {
    const payload = job.payload;
    const model =
      payload.mode === "chat"
        ? resolveTranslationModel({
            rawModel: payload.model,
            modelFamily: payload.modelFamily,
            messages: payload.messages,
            canUseAnthropic: Boolean(
              job.anthropicKey || process.env.ANTHROPIC_API_KEY,
            ),
            translationPhase: payload.translationPhase,
            qualityMode: payload.qualityMode,
          })
        : normalizeModelId(payload.model || DEFAULT_TRANSLATION_MODEL);

    if (!isAllowedStage5TranslationModel(model)) {
      throw new Error(`Unsupported translation model: ${model}`);
    }

    // Route to Claude for Claude models
    if (isClaudeModel(model)) {
      const anthropicKey =
        job.anthropicKey || process.env.ANTHROPIC_API_KEY || "";
      if (!anthropicKey) {
        throw new Error("No Anthropic API key available for Claude model");
      }

      if (payload.mode === "chat") {
        const { messages, reasoning } = payload;
        // Extract effort level from reasoning object (OpenAI format: { effort: 'high' })
        const effort = reasoning?.effort as
          | "low"
          | "medium"
          | "high"
          | undefined;
        const completion = await translateWithClaude({
          messages: messages as any,
          model,
          apiKey: anthropicKey,
          effort,
        });
        job.result = completion;
      } else {
        const { text, target_language } = payload;
        const completion = await translateWithClaude({
          messages: [
            {
              role: "user",
              content: `You are a professional translator. Translate the following text to ${target_language}. Only return the translated text, nothing else.\n\n${text}`,
            },
          ],
          model,
          apiKey: anthropicKey,
        });
        job.result = completion;
      }

      console.log(`🎯 Translation completed using Anthropic (model=${model})`);
      job.status = "completed";
      job.updatedAt = Date.now();
      try {
        await upsertDurableRelayTranslationJob(
          CF_API_BASE,
          RELAY_SECRET,
          job.id,
          "completed",
          {
            result: job.result ?? null,
            error: null,
          },
        );
      } catch (persistErr: any) {
        console.error(
          "⚠️ Failed to persist relay translation job completion:",
          persistErr?.message || persistErr,
        );
      }
      return;
    }

    // OpenAI path (existing logic)
    const client = makeOpenAI(job.openaiKey);

    if (payload.mode === "chat") {
      const { messages, reasoning } = payload;

      const request: any = {
        model,
        messages,
      };

      // Chat Completions API uses flat `reasoning_effort` parameter, not nested object
      if (reasoning?.effort) {
        request.reasoning_effort = reasoning.effort;
      }

      let completion;

      try {
        completion = await client.chat.completions.create(request);
      } catch (maybeReasoningError: any) {
        const status =
          maybeReasoningError?.status || maybeReasoningError?.response?.status;
        const msg = String(maybeReasoningError?.message || "").toLowerCase();
        // If reasoning_effort caused the error, retry without it
        if (
          reasoning?.effort &&
          (status === 400 || msg.includes("reasoning"))
        ) {
          const reqWithoutReasoning: any = { model, messages };
          completion =
            await client.chat.completions.create(reqWithoutReasoning);
        } else {
          throw maybeReasoningError;
        }
      }

      job.result = { ...completion, model };
    } else {
      const { text, target_language } = payload;

      const request: any = {
        model,
        messages: [
          {
            role: "system",
            content: `You are a professional translator. Translate the following text to ${target_language}. Only return the translated text, nothing else.`,
          },
          {
            role: "user",
            content: text,
          },
        ],
      };

      const completion = await client.chat.completions.create(request);
      job.result = { ...completion, model };
    }

    console.log(`🎯 Translation completed using OpenAI (model=${model})`);
    job.status = "completed";
    job.updatedAt = Date.now();
    try {
      await upsertDurableRelayTranslationJob(
        CF_API_BASE,
        RELAY_SECRET,
        job.id,
        "completed",
        {
          result: job.result ?? null,
          error: null,
        },
      );
    } catch (persistErr: any) {
      console.error(
        "⚠️ Failed to persist relay translation job completion:",
        persistErr?.message || persistErr,
      );
    }
  } catch (error: any) {
    job.error = {
      message: error?.message || String(error),
      details: error?.response?.data
        ? JSON.stringify(error.response.data)
        : undefined,
    };
    job.status = "failed";
    console.error("❌ Translation job failed:", job.error.message);
    try {
      await upsertDurableRelayTranslationJob(
        CF_API_BASE,
        RELAY_SECRET,
        job.id,
        "failed",
        {
          error: job.error.message,
        },
      );
    } catch (persistErr: any) {
      console.error(
        "❌ Failed to persist relay translation job failure:",
        persistErr?.message || persistErr,
      );
    }
  } finally {
    job.updatedAt = Date.now();
  }
}

const relayRoutesContext: RelayRoutesContext = {
  ALLOWED_ORIGINS,
  RELAY_SECRET,
  MAX_BODY_SIZE,
  ELEVENLABS_TRANSCRIPTION_MODEL,
  WHISPER_TRANSCRIPTION_MODEL,
  CF_API_BASE,
  R2_FETCH_TIMEOUT_MS,
  RELAY_TRANSLATION_STALE_MS,
  DUB_MAX_SEGMENTS,
  DUB_MAX_TOTAL_CHARACTERS,
  DUB_MAX_RETRIES,
  DUB_RETRY_BASE_DELAY_MS,
  DUB_RETRY_MAX_DELAY_MS,
  DUB_MAX_CONCURRENCY,
  MAX_TTS_CHARS_PER_CHUNK,
  SCRIBE_MAX_RETRIES,
  DEFAULT_TRANSLATION_MODEL,
  getCorsOrigin,
  getHeader,
  sendError,
  sendJson,
  validateRelaySecret,
  getDurableRelayTranslationJob,
  resolveRelayPollJob,
  upsertDurableRelayTranslationJob,
  parseBooleanLike,
  parseTranslationModelFamily,
  parseTranslationPhase,
  transcribeWithScribe,
  synthesizeWithElevenLabs,
  dubWithElevenLabs,
  transcribeWithWhisperFromPath,
  transcribeWithScribeWithRetries,
  resolveDirectTranscriptionQuality,
  getWhisperFileSizeGuardMessage,
  toWhisperCompatibleScribeResult,
  readJsonBody,
  resolveTranslationModel,
  isAllowedStage5TranslationModel,
  isClaudeModel,
  normalizeModelId,
  translateWithClaude,
  translateWithClaudeWebSearch,
  translateWithOpenAiWebSearch,
  makeOpenAI,
  translationJobs,
  pruneTranslationJobs,
  processTranslationJob,
  chunkLines,
  shouldRetrySegmentError,
  sleep,
  validateR2Url,
};

const server = createServer((req, res) =>
  handleRelayRequest(req, res, relayRoutesContext),
);

server.listen(PORT, () => {
  console.log(`🚀 OpenAI Relay server running on port ${PORT}`);
  console.log(`📡 Ready to process real transcriptions via OpenAI`);
});

setInterval(() => pruneTranslationJobs(), 60_000);

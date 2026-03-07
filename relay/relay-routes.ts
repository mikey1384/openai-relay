import type { IncomingMessage, ServerResponse } from "node:http";
import type { Buffer } from "node:buffer";
import type { ScribeResult, DubbingResult } from "../elevenlabs-config.js";
import type { TranslationModelFamily } from "./relay-translation-helpers.js";
import type {
  DurableRelayJob,
  MemoryRelayJobLike,
  RelayJobStatus,
} from "./relay-job-sync.js";
import { handleSpeechRoutes } from "./relay-routes-speech.js";
import { handleTranscriptionRoutes } from "./relay-routes-transcription.js";
import { handleTranslationRoutes } from "./relay-routes-translation.js";
import { handleDubbingRoutes } from "./relay-routes-dubbing.js";
import { enforceMinimumTranslatorVersion } from "./translator-version-gate.js";

export type RelayChatMessage = {
  role: string;
  content: string;
};

export type RelayChatJobPayload = {
  mode: "chat";
  messages: RelayChatMessage[];
  model: string;
  modelFamily?: TranslationModelFamily;
  reasoning?: any;
  translationPhase?: "draft" | "review";
  qualityMode?: boolean;
};

export type RelayTextJobPayload = {
  mode: "text";
  text: string;
  target_language: string;
  model: string;
};

export type RelayTranslationJob = {
  id: string;
  status: RelayJobStatus;
  createdAt: number;
  updatedAt: number;
  payload: RelayChatJobPayload | RelayTextJobPayload;
  openaiKey: string;
  anthropicKey?: string;
  result?: any;
  error?: {
    message: string;
    details?: string;
  };
};

export type WhisperCompatibleTranscriptionResult = any;

export interface RelayRoutesContext {
  ALLOWED_ORIGINS: string[];
  RELAY_SECRET: string;
  MAX_BODY_SIZE: number;
  ELEVENLABS_TRANSCRIPTION_MODEL: string;
  WHISPER_TRANSCRIPTION_MODEL: string;
  CF_API_BASE: string;
  R2_FETCH_TIMEOUT_MS: number;
  RELAY_TRANSLATION_STALE_MS: number;
  DUB_MAX_SEGMENTS: number;
  DUB_MAX_TOTAL_CHARACTERS: number;
  DUB_MAX_RETRIES: number;
  DUB_RETRY_BASE_DELAY_MS: number;
  DUB_RETRY_MAX_DELAY_MS: number;
  DUB_MAX_CONCURRENCY: number;
  MAX_TTS_CHARS_PER_CHUNK: number;
  SCRIBE_MAX_RETRIES: number;
  DEFAULT_TRANSLATION_MODEL: string;
  getCorsOrigin: (req: IncomingMessage, allowedOrigins: string[]) => string | null;
  getHeader: (req: IncomingMessage, name: string) => string | undefined;
  sendError: (
    res: ServerResponse,
    status: number,
    error: string,
    details?: string
  ) => void;
  sendJson: (res: ServerResponse, data: unknown, status?: number) => void;
  validateRelaySecret: (req: IncomingMessage, relaySecret: string) => boolean;
  getDurableRelayTranslationJob: (
    cfApiBase: string,
    relaySecret: string,
    jobId: string
  ) => Promise<DurableRelayJob | null>;
  resolveRelayPollJob: <T extends MemoryRelayJobLike>(
    jobId: string,
    durableJob: DurableRelayJob | null,
    memoryJobs: Map<string, T>
  ) =>
    | {
        source: "memory" | "durable";
        job: T | DurableRelayJob;
        updatedMs: number | null;
      }
    | null;
  upsertDurableRelayTranslationJob: (
    cfApiBase: string,
    relaySecret: string,
    jobId: string,
    status: RelayJobStatus,
    options?: {
      result?: unknown;
      error?: string | null;
    }
  ) => Promise<void>;
  parseBooleanLike: (raw: unknown) => boolean | undefined;
  parseTranslationModelFamily: (
    raw: unknown
  ) => TranslationModelFamily | undefined;
  parseTranslationPhase: (raw: unknown) => "draft" | "review" | undefined;
  transcribeWithScribe: (params: {
    filePath: string;
    apiKey: string;
    languageCode?: string;
    idempotencyKey?: string;
    signal?: AbortSignal;
  }) => Promise<ScribeResult>;
  synthesizeWithElevenLabs: (params: {
    text: string;
    voice?: string;
    modelId?: string;
    format?: string;
    apiKey: string;
    signal?: AbortSignal;
  }) => Promise<Buffer>;
  dubWithElevenLabs: (params: {
    fileBuffer: Buffer;
    fileName: string;
    mimeType: string;
    sourceLanguage?: string;
    targetLanguage: string;
    apiKey: string;
    numSpeakers?: number;
    dropBackgroundAudio?: boolean;
    pollIntervalMs?: number;
    maxWaitMs?: number;
    onProgress?: (status: string) => void;
  }) => Promise<DubbingResult>;
  transcribeWithWhisperFromPath: (params: {
    openaiKey: string;
    filePath: string;
    fileName: string;
    mimeType: string;
    language?: string;
    prompt?: string;
    signal?: AbortSignal;
  }) => Promise<WhisperCompatibleTranscriptionResult>;
  transcribeWithScribeWithRetries: (params: {
    filePath: string;
    apiKey: string;
    languageCode: string;
    idempotencyKey?: string;
    contextLabel: string;
    signal?: AbortSignal;
  }) => Promise<{
    result: ScribeResult;
    attempts: number;
  }>;
  resolveDirectTranscriptionQuality: (params: {
    explicitQualityRaw: unknown;
    modelHint?: string;
    modelIdHint?: string;
  }) => {
    useHighQuality: boolean;
    source: "explicit" | "model-hint" | "default";
  };
  getWhisperFileSizeGuardMessage: (fileSizeBytes: number) => string | null;
  toWhisperCompatibleScribeResult: (
    result: ScribeResult
  ) => WhisperCompatibleTranscriptionResult;
  readJsonBody: (req: IncomingMessage) => Promise<any>;
  resolveTranslationModel: (params: {
    rawModel?: string;
    modelFamily?: TranslationModelFamily;
    messages?: any[];
    canUseAnthropic: boolean;
    translationPhase?: "draft" | "review";
    qualityMode?: boolean;
  }) => string;
  resolveTranslationReservationMaxCompletionTokens: (params: {
    model: string;
    reasoning?: { effort?: "low" | "medium" | "high" } | null;
  }) => number;
  isAllowedStage5TranslationModel: (model: string) => boolean;
  isClaudeModel: (model: string) => boolean;
  normalizeModelId: (model: string) => string;
  translateWithClaude: (params: {
    messages: Array<{ role: "user" | "assistant"; content: string }>;
    model: string;
    apiKey: string;
    signal?: AbortSignal;
    maxTokens?: number;
    effort?: "low" | "medium" | "high";
  }) => Promise<{
    model: string;
    choices: Array<{ message: { role: string; content: string } }>;
    usage: { prompt_tokens: number; completion_tokens: number };
  }>;
  translateWithClaudeWebSearch: (params: {
    messages: RelayChatMessage[];
    model: string;
    apiKey: string;
    maxTokens?: number;
    effort?: "low" | "medium" | "high";
  }) => Promise<{
    model: string;
    content: string;
    usage: { prompt_tokens: number; completion_tokens: number };
  }>;
  translateWithOpenAiWebSearch: (params: {
    messages: any[];
    model: string;
    apiKey: string;
    maxOutputTokens?: number;
    reasoning?: any;
  }) => Promise<{
    model: string;
    content: string;
    usage: { prompt_tokens: number; completion_tokens: number };
  }>;
  makeOpenAI: (apiKey: string) => any;
  translationJobs: Map<string, RelayTranslationJob>;
  pruneTranslationJobs: (maxAgeMs?: number) => void;
  processTranslationJob: (job: RelayTranslationJob) => Promise<void>;
  chunkLines: (lines: string[], maxChars: number) => string[];
  shouldRetrySegmentError: (error: unknown) => boolean;
  sleep: (ms: number) => Promise<void>;
  validateR2Url: (urlString: string) => { valid: boolean; error?: string };
}

export async function handleRelayRequest(
  req: IncomingMessage,
  res: ServerResponse,
  ctx: RelayRoutesContext
): Promise<void> {
  const { ALLOWED_ORIGINS, getCorsOrigin, sendError } = ctx;

  console.log(
    `🔍 Relay received: ${req.method} ${req.url} from ${
      req.headers["user-agent"] || "unknown"
    }`
  );

  // Enable CORS with configurable origins
  const corsOrigin = getCorsOrigin(req, ALLOWED_ORIGINS);
  if (corsOrigin) {
    res.setHeader("Access-Control-Allow-Origin", corsOrigin);
    res.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
    res.setHeader(
      "Access-Control-Allow-Headers",
      "Content-Type, Authorization, Idempotency-Key, X-Idempotency-Key, X-Stage5-App-Version"
    );
    res.setHeader("Vary", "Origin");
  }

  // Handle preflight requests
  if (req.method === "OPTIONS") {
    console.log("✅ Responding to preflight request");
    res.writeHead(corsOrigin ? 200 : 403);
    res.end();
    return;
  }

  if (enforceMinimumTranslatorVersion({ req, res, sendJson: ctx.sendJson })) {
    return;
  }

  if (await handleSpeechRoutes(req, res, ctx)) {
    return;
  }

  if (await handleTranscriptionRoutes(req, res, ctx)) {
    return;
  }

  if (await handleTranslationRoutes(req, res, ctx)) {
    return;
  }

  if (await handleDubbingRoutes(req, res, ctx)) {
    return;
  }

  // Handle all other requests
  console.log(`❌ Unsupported request: ${req.method} ${req.url}`);
  sendError(res, 404, "Endpoint not found");
}

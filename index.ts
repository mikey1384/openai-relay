import { createServer } from "node:http";
import { Buffer } from "node:buffer";
import { IncomingForm } from "formidable";
import { randomUUID } from "node:crypto";
import { makeOpenAI } from "./openai-config.js";
import { translateWithClaude } from "./anthropic-config.js";
import { DEFAULT_TRANSLATION_MODEL, isClaudeModel } from "./constants.js";
import {
  transcribeWithScribe,
  synthesizeWithElevenLabs,
  dubWithElevenLabs,
} from "./elevenlabs-config.js";

const PORT = process.env.PORT || 3000;

const MAX_TTS_CHARS_PER_CHUNK = 3_500;
const DUB_MAX_CONCURRENCY = Math.max(
  1,
  Number.parseInt(process.env.DUB_MAX_CONCURRENCY || "4", 10)
);
const DUB_MAX_RETRIES = Math.max(
  1,
  Number.parseInt(process.env.DUB_MAX_RETRIES || "3", 10)
);
const DUB_RETRY_BASE_DELAY_MS = Math.max(
  100,
  Number.parseInt(process.env.DUB_RETRY_BASE_DELAY_MS || "500", 10)
);
const DUB_RETRY_MAX_DELAY_MS = Math.max(
  DUB_RETRY_BASE_DELAY_MS,
  Number.parseInt(process.env.DUB_RETRY_MAX_DELAY_MS || "4000", 10)
);
const DUB_MAX_TOTAL_CHARACTERS = Math.max(
  1,
  Number.parseInt(process.env.DUB_MAX_TOTAL_CHARACTERS || "90000", 10)
);
const DUB_MAX_SEGMENTS = Math.max(
  1,
  Number.parseInt(process.env.DUB_MAX_SEGMENTS || "240", 10)
);
const MAX_BODY_SIZE = 10 * 1024 * 1024; // 10MB limit for request bodies
const MAX_TRANSLATION_JOBS = 1000; // Memory leak prevention
const JOB_MAX_AGE_MS = 30 * 60 * 1000; // 30 minutes (reduced from 1 hour)
const R2_FETCH_TIMEOUT_MS = 120_000; // 2 minute timeout for R2 fetches

// Fail fast if RELAY_SECRET is missing - this is critical for security
const RELAY_SECRET = process.env.RELAY_SECRET;
if (!RELAY_SECRET) {
  console.error("❌ FATAL: RELAY_SECRET environment variable is not set");
  process.exit(1);
}

// Allowed CORS origins (restrict in production)
const ALLOWED_ORIGINS = process.env.ALLOWED_ORIGINS
  ? process.env.ALLOWED_ORIGINS.split(",").map((o) => o.trim())
  : ["*"]; // Default to * for backwards compatibility

// Allowed R2 bucket hostnames for SSRF prevention
const ALLOWED_R2_HOSTS = process.env.ALLOWED_R2_HOSTS
  ? process.env.ALLOWED_R2_HOSTS.split(",").map((h) => h.trim().toLowerCase())
  : [];

type TranslationJobStatus = "queued" | "processing" | "completed" | "failed";

type ChatJobPayload = {
  mode: "chat";
  messages: Array<{ role: string; content: string }>;
  model: string;
  reasoning?: any;
};

type TextJobPayload = {
  mode: "text";
  text: string;
  target_language: string;
  model: string;
};

type TranslationJob = {
  id: string;
  status: TranslationJobStatus;
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

import type { IncomingMessage, ServerResponse } from "node:http";

/**
 * Validate the relay secret from request headers.
 * Returns true if valid, false otherwise.
 */
function validateRelaySecret(req: IncomingMessage): boolean {
  const relaySecretHeader = req.headers["x-relay-secret"];
  const providedSecret = Array.isArray(relaySecretHeader)
    ? relaySecretHeader[0]
    : relaySecretHeader;
  return providedSecret === RELAY_SECRET;
}

/**
 * Extract a single header value (handles array headers).
 */
function getHeader(req: IncomingMessage, name: string): string | undefined {
  const value = req.headers[name.toLowerCase()];
  if (Array.isArray(value)) return value[0];
  return value || undefined;
}

/**
 * Send a JSON error response.
 */
function sendError(
  res: ServerResponse,
  status: number,
  error: string,
  details?: string
): void {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(details ? { error, details } : { error }));
}

/**
 * Send a JSON success response.
 */
function sendJson(res: ServerResponse, data: unknown, status = 200): void {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(data));
}

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
    if (!ALLOWED_R2_HOSTS.some((h) => hostname === h || hostname.endsWith(`.${h}`))) {
      return { valid: false, error: `R2 URL hostname not in allowed list: ${hostname}` };
    }

    return { valid: true };
  } catch {
    return { valid: false, error: "Invalid R2 URL format" };
  }
}

/**
 * Get the CORS origin to return based on the request origin.
 */
function getCorsOrigin(req: IncomingMessage): string {
  const requestOrigin = getHeader(req, "origin");

  // If wildcard is allowed, return wildcard
  if (ALLOWED_ORIGINS.includes("*")) {
    return "*";
  }

  // If request origin is in allowed list, return it
  if (requestOrigin && ALLOWED_ORIGINS.includes(requestOrigin)) {
    return requestOrigin;
  }

  // Default to first allowed origin
  return ALLOWED_ORIGINS[0] || "*";
}

function readJsonBody(req: any): Promise<any> {
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
    typeof line === "string" ? line.trim() : ""
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
    const sortedJobs = [...translationJobs.entries()]
      .sort((a, b) => a[1].createdAt - b[1].createdAt);

    const toRemove = sortedJobs.slice(0, translationJobs.size - MAX_TRANSLATION_JOBS);
    for (const [jobId] of toRemove) {
      console.warn(`⚠️ Removing job ${jobId} due to job limit (${MAX_TRANSLATION_JOBS})`);
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
      message
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

async function processTranslationJob(job: TranslationJob): Promise<void> {
  job.status = "processing";
  job.updatedAt = Date.now();

  try {
    const payload = job.payload;
    const model = payload.model || DEFAULT_TRANSLATION_MODEL;

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
        const effort = reasoning?.effort as 'low' | 'medium' | 'high' | undefined;
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
        if (reasoning?.effort && (status === 400 || msg.includes("reasoning"))) {
          const reqWithoutReasoning: any = { model, messages };
          completion = await client.chat.completions.create(
            reqWithoutReasoning
          );
        } else {
          throw maybeReasoningError;
        }
      }

      job.result = completion;
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
      job.result = completion;
    }

    console.log(`🎯 Translation completed using OpenAI (model=${model})`);
    job.status = "completed";
  } catch (error: any) {
    job.error = {
      message: error?.message || String(error),
      details: error?.response?.data
        ? JSON.stringify(error.response.data)
        : undefined,
    };
    job.status = "failed";
    console.error("❌ Translation job failed:", job.error.message);
  } finally {
    job.updatedAt = Date.now();
  }
}

const server = createServer(async (req, res) => {
  console.log(
    `🔍 Relay received: ${req.method} ${req.url} from ${
      req.headers["user-agent"] || "unknown"
    }`
  );

  // Enable CORS with configurable origins
  const corsOrigin = getCorsOrigin(req);
  res.setHeader("Access-Control-Allow-Origin", corsOrigin);
  res.setHeader("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
  res.setHeader(
    "Access-Control-Allow-Headers",
    "Content-Type, Authorization, X-Relay-Secret, X-OpenAI-Key, X-Anthropic-Key, X-ElevenLabs-Key"
  );
  if (corsOrigin !== "*") {
    res.setHeader("Vary", "Origin");
  }

  // Handle preflight requests
  if (req.method === "OPTIONS") {
    console.log("✅ Responding to preflight request");
    res.writeHead(200);
    res.end();
    return;
  }

  // Handle POST to /speech (text-to-speech synthesis)
  if (req.method === "POST" && req.url === "/speech") {
    console.log("🎤 Processing speech synthesis request...");

    if (!validateRelaySecret(req)) {
      console.log("❌ Invalid or missing relay secret for /speech");
      sendError(res, 401, "Unauthorized - invalid relay secret");
      return;
    }

    const openaiKey = getHeader(req, "x-openai-key");
    if (!openaiKey) {
      console.log("❌ Missing OpenAI API key for /speech");
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
      sendError(res, 500, "Request stream error", err.message);
    });

    req.on("end", async () => {
      if (rejected) return;
      try {
        const parsed = JSON.parse(body || "{}");
        const text = parsed.text;
        if (!text || typeof text !== "string") {
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

        sendJson(res, {
          audioBase64,
          voice,
          model,
          format,
          length: text.length,
        });
      } catch (error: any) {
        console.error(
          "❌ Relay speech synthesis error:",
          error.message || error
        );
        sendError(res, 500, "Speech synthesis failed", error?.message || String(error));
      }
    });

    return;
  }

  // Handle POST to /transcribe
  if (req.method === "POST" && req.url === "/transcribe") {
    console.log("📡 Processing transcribe request...");

    if (!validateRelaySecret(req)) {
      console.log("❌ Invalid or missing relay secret");
      sendError(res, 401, "Unauthorized - invalid relay secret");
      return;
    }

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

      const model = Array.isArray(fields.model)
        ? fields.model[0]
        : fields.model || "whisper-1";
      const language = Array.isArray(fields.language)
        ? fields.language[0]
        : fields.language;
      const prompt = Array.isArray(fields.prompt)
        ? fields.prompt[0]
        : fields.prompt;

      console.log(
        `🎵 Transcribing file: ${file.originalFilename} (${(file.size / 1024 / 1024).toFixed(1)}MB) with model: ${model}`
      );

      const openaiKey = getHeader(req, "x-openai-key");
      if (!openaiKey) {
        console.log("❌ Missing OpenAI API key");
        sendError(res, 401, "Unauthorized - missing OpenAI key");
        return;
      }
      const client = makeOpenAI(openaiKey);

      const fs = await import("fs");
      const fileBuffer = await fs.promises.readFile(file.filepath);
      const fileBlob = new File(
        [fileBuffer as unknown as BlobPart],
        file.originalFilename || "audio.webm",
        {
          type: file.mimetype || "audio/webm",
        }
      );

      const transcription = await client.audio.transcriptions.create({
        file: fileBlob,
        model: model,
        language: language || undefined,
        prompt: prompt || undefined,
        response_format: "verbose_json",
        timestamp_granularities: ["word", "segment"],
      });

      console.log(`🎯 Relay transcription completed successfully using OpenAI!`);
      sendJson(res, transcription);
    } catch (error: any) {
      console.error("❌ Relay transcription error:", error.message);
      sendError(res, 500, "Transcription failed", error.message);
    }

    return;
  }

  // Handle POST to /transcribe-elevenlabs (ElevenLabs Scribe)
  if (req.method === "POST" && req.url === "/transcribe-elevenlabs") {
    console.log("📡 Processing ElevenLabs Scribe transcription request...");

    if (!validateRelaySecret(req)) {
      console.log("❌ Invalid or missing relay secret");
      sendError(res, 401, "Unauthorized - invalid relay secret");
      return;
    }

    const elevenLabsKey = getHeader(req, "x-elevenlabs-key") || process.env.ELEVENLABS_API_KEY;
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

      const language = Array.isArray(fields.language)
        ? fields.language[0]
        : fields.language;

      console.log(
        `🎵 Transcribing with ElevenLabs Scribe: ${file.originalFilename} (${(file.size / 1024 / 1024).toFixed(1)}MB)`
      );

      const result = await transcribeWithScribe({
        filePath: file.filepath,
        apiKey: elevenLabsKey,
        languageCode: language || "auto",
      });

      // Convert to Whisper-compatible format
      const whisperFormat = {
        text: result.text,
        language: result.language_code,
        segments: result.segments.map((seg, idx) => ({
          id: idx,
          start: seg.start,
          end: seg.end,
          text: seg.text,
          words: seg.words?.map((w) => ({
            word: w.text,
            start: w.start,
            end: w.end,
          })),
        })),
        words: result.segments.flatMap(
          (seg) =>
            seg.words?.map((w) => ({
              word: w.text,
              start: w.start,
              end: w.end,
            })) || []
        ),
      };

      console.log(`🎯 ElevenLabs Scribe transcription completed!`);
      sendJson(res, whisperFormat);
    } catch (error: any) {
      console.error("❌ ElevenLabs Scribe error:", error.message);
      sendError(res, 500, "Transcription failed", error.message);
    }

    return;
  }

  // Handle POST to /transcribe-from-r2 (ElevenLabs Scribe from R2 URL)
  if (req.method === "POST" && req.url === "/transcribe-from-r2") {
    console.log("📡 Processing ElevenLabs Scribe from R2 URL...");

    if (!validateRelaySecret(req)) {
      console.log("❌ Invalid or missing relay secret for /transcribe-from-r2");
      sendError(res, 401, "Unauthorized - invalid relay secret");
      return;
    }

    const elevenLabsKey = getHeader(req, "x-elevenlabs-key") || process.env.ELEVENLABS_API_KEY;
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
      const { r2Url, language } = JSON.parse(body);

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

      // Fetch the file from R2 with timeout
      const abortController = new AbortController();
      const timeoutId = setTimeout(() => abortController.abort(), R2_FETCH_TIMEOUT_MS);

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

      console.log(`🎵 Transcribing with ElevenLabs Scribe (${fileSizeMB.toFixed(1)}MB from R2)`);

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
        });

        // Convert to Whisper-compatible format (same as /transcribe-elevenlabs)
        const allWords = result.segments.flatMap(
          (seg) =>
            seg.words?.map((w) => ({
              word: w.text,
              start: w.start,
              end: w.end,
            })) || []
        );

        // Calculate duration from segments
        const duration =
          result.segments.length > 0
            ? result.segments[result.segments.length - 1].end
            : 0;

        const whisperFormat = {
          text: result.text,
          language: result.language_code,
          segments: result.segments.map((seg, idx) => ({
            id: idx,
            start: seg.start,
            end: seg.end,
            text: seg.text,
            words: seg.words?.map((w) => ({
              word: w.text,
              start: w.start,
              end: w.end,
            })),
          })),
          words: allWords,
          duration,
          approx_duration: duration,
        };

        console.log(`🎯 ElevenLabs Scribe (R2) completed! Duration: ${duration.toFixed(1)}s`);
        sendJson(res, whisperFormat);
      } finally {
        // Cleanup temp file
        try {
          await fs.promises.unlink(tempFile);
        } catch (cleanupErr: any) {
          console.warn(`⚠️ Failed to cleanup temp file ${tempFile}:`, cleanupErr.message);
        }
      }
    } catch (error: any) {
      console.error("❌ ElevenLabs Scribe (R2) error:", error.message);
      sendError(res, 500, "Transcription from R2 failed", error.message);
    }

    return;
  }

  // Handle POST to /dub-elevenlabs (ElevenLabs TTS)
  if (req.method === "POST" && req.url === "/dub-elevenlabs") {
    console.log("🎬 Processing ElevenLabs TTS dub request...");

    if (!validateRelaySecret(req)) {
      console.log("❌ Invalid or missing relay secret for /dub-elevenlabs");
      sendError(res, 401, "Unauthorized - invalid relay secret");
      return;
    }

    const elevenLabsKey = getHeader(req, "x-elevenlabs-key") || process.env.ELEVENLABS_API_KEY;
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
              const text =
                typeof segment?.text === "string" ? segment.text.trim() : "";
              if (!text) return null;
              const index = Number.isFinite(segment?.index)
                ? Number(segment.index)
                : idx + 1;
              const targetDuration =
                typeof segment?.targetDuration === "number" && Number.isFinite(segment.targetDuration)
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

    return;
  }

  // Handle POST to /dub-video-elevenlabs (ElevenLabs Dubbing API with voice cloning)
  if (req.method === "POST" && req.url === "/dub-video-elevenlabs") {
    console.log("🎬 Processing ElevenLabs video dubbing request...");

    if (!validateRelaySecret(req)) {
      console.log("❌ Invalid or missing relay secret for /dub-video-elevenlabs");
      sendError(res, 401, "Unauthorized - invalid relay secret");
      return;
    }

    const elevenLabsKey = getHeader(req, "x-elevenlabs-key") || process.env.ELEVENLABS_API_KEY;
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
      const numSpeakers = numSpeakersRaw !== undefined && Number.isFinite(numSpeakersRaw) && numSpeakersRaw > 0
        ? numSpeakersRaw
        : undefined;
      const dropBackgroundAudioField = Array.isArray(
        fields.drop_background_audio
      )
        ? fields.drop_background_audio[0]
        : fields.drop_background_audio;
      const dropBackgroundAudio = dropBackgroundAudioField !== "false";

      console.log(
        `🎬 Dubbing video: ${file.originalFilename} (${(file.size / 1024 / 1024).toFixed(1)}MB) → ${targetLanguage}`
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
        onProgress: (status) => console.log(`   • ${status}`),
      });

      console.log(`🎯 ElevenLabs video dubbing completed!`);
      sendJson(res, result);
    } catch (error: any) {
      console.error("❌ ElevenLabs video dubbing error:", error.message);
      sendError(res, 500, "Video dubbing failed", error.message);
    }

    return;
  }

  // Handle POST to /translate via async job submission
  if (req.method === "POST" && req.url === "/translate") {
    console.log("🌐 Processing translate request (job)...");

    if (!validateRelaySecret(req)) {
      console.log("❌ Invalid or missing relay secret");
      sendError(res, 401, "Unauthorized - invalid relay secret");
      return;
    }

    const openaiKey = getHeader(req, "x-openai-key");
    const anthropicKey = getHeader(req, "x-anthropic-key");

    try {
      const parsed = await readJsonBody(req);

      let payload: ChatJobPayload | TextJobPayload | null = null;
      if (Array.isArray(parsed?.messages)) {
        const messages = parsed.messages.map((m: any) => ({
          role: String(m?.role ?? ""),
          content: String(m?.content ?? ""),
        }));
        const model =
          typeof parsed?.model === "string" && parsed.model.trim()
            ? parsed.model.trim()
            : DEFAULT_TRANSLATION_MODEL;
        payload = {
          mode: "chat",
          messages,
          model,
          reasoning: parsed?.reasoning,
        };
      } else if (
        typeof parsed?.text === "string" &&
        parsed.text.trim() &&
        typeof parsed?.target_language === "string" &&
        parsed.target_language.trim()
      ) {
        const model =
          typeof parsed?.model === "string" && parsed.model.trim()
            ? parsed.model.trim()
            : DEFAULT_TRANSLATION_MODEL;
        payload = {
          mode: "text",
          text: parsed.text,
          target_language: parsed.target_language,
          model,
        };
      }

      if (!payload) {
        sendError(res, 400, "Invalid translation payload", "Expected messages[] or text/target_language");
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

      const job: TranslationJob = {
        id: randomUUID(),
        status: "queued",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        payload,
        openaiKey: openaiKey || "",
        anthropicKey: anthropicKey || undefined,
      };

      translationJobs.set(job.id, job);

      sendJson(res, { jobId: job.id, status: job.status }, 202);

      setImmediate(() => {
        processTranslationJob(job)
          .catch((err) => {
            job.error = {
              message: err?.message || String(err),
            };
            job.status = "failed";
            job.updatedAt = Date.now();
            console.error(
              "❌ Translation job processing error:",
              job.error.message
            );
          })
          .finally(() => {
            pruneTranslationJobs();
          });
      });
    } catch (error: any) {
      console.error(
        "❌ Relay translation job submission error:",
        error.message
      );
      sendError(res, 500, "Translation job submission failed", error.message);
    }

    return;
  }
  if (
    req.method === "GET" &&
    req.url &&
    req.url.startsWith("/translate/result/")
  ) {
    if (!validateRelaySecret(req)) {
      sendError(res, 401, "Unauthorized - invalid relay secret");
      return;
    }

    const jobId = decodeURIComponent(
      req.url.split("?")[0].split("/").pop() || ""
    );
    const job = translationJobs.get(jobId);
    if (!job) {
      sendError(res, 404, "Job not found");
      return;
    }

    if (job.status === "completed") {
      sendJson(res, job.result ?? {});
    } else if (job.status === "failed") {
      sendError(res, 500, job.error?.message || "Translation failed", job.error?.details);
    } else {
      res.writeHead(202, {
        "Content-Type": "application/json",
        "Retry-After": "2",
      });
      res.end(JSON.stringify({ status: job.status }));
    }
    return;
  }
  // Handle POST to /dub (multi-chunk speech synthesis)
  if (req.method === "POST" && req.url === "/dub") {
    console.log("🎬 Processing dub synthesis request...");

    if (!validateRelaySecret(req)) {
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
          sendError(res, 413, "Dub request too large", `Received ${segmentsPayload.length} segments, limit is ${DUB_MAX_SEGMENTS}`);
          return;
        }

        const totalCharacters = segmentsPayload.reduce(
          (sum: number, seg: { text: string }) => sum + seg.text.length,
          0
        );

        if (totalCharacters > DUB_MAX_TOTAL_CHARACTERS) {
          sendError(res, 413, "Dub request too large", `Received ${totalCharacters} characters, limit is ${DUB_MAX_TOTAL_CHARACTERS}`);
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

                if (
                  attempt >= DUB_MAX_RETRIES ||
                  !shouldRetrySegmentError(segmentError)
                ) {
                  throw segmentError;
                }

                const delay = Math.min(
                  DUB_RETRY_MAX_DELAY_MS,
                  DUB_RETRY_BASE_DELAY_MS * Math.pow(2, attempt - 1)
                );
                console.warn(
                  `⚠️ Segment ${segIdx + 1}/${
                    segmentsPayload.length
                  } retry ${attempt}/${DUB_MAX_RETRIES} in ${delay}ms:`,
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
        const workerCount = Math.min(
          DUB_MAX_CONCURRENCY,
          segmentsPayload.length
        );

        const workers = Array.from(
          { length: workerCount },
          async (_, workerIdx) => {
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
                }/${segmentsPayload.length} (index=${seg.index}, ${
                  seg.text.length
                } chars)`
              );
              await synthesizeSegment(current);
              console.log(
                `     · Completed segment ${current + 1}/${
                  segmentsPayload.length
                }`
              );
            }
          }
        );

        try {
          await Promise.all(workers);
        } catch (segmentError: any) {
          if (requestClosed) {
            console.warn(
              "⚠️ Dub request aborted by upstream client while synthesizing segments"
            );
            return;
          }

          const details = segmentError?.response?.data ?? segmentError?.message;
          console.error("❌ Relay segment synthesis failed:", details);
          sendError(res, 500, "Dub synthesis failed", typeof details === "string" ? details : JSON.stringify(details));
          return;
        }

        if (requestClosed) {
          console.warn(
            "⚠️ Dub request closed before completion; skipping response"
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
      }

      if (!lines.length) {
        sendError(res, 400, "Invalid request: lines array required");
        return;
      }

      const totalCharacters = lines.reduce(
        (sum: number, line: string) => sum + line.length,
        0
      );
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
        console.log(
          `   • Chunk ${idx + 1}/${chunks.length} (${chunk.length} chars)`
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
      return;
    } catch (error: any) {
      console.error("❌ Relay dub synthesis error:", error?.message || error);
      sendError(res, 500, "Dub synthesis failed", error?.message || String(error));
      return;
    }
  }

  // Handle all other requests
  console.log(`❌ Unsupported request: ${req.method} ${req.url}`);
  sendError(res, 404, "Endpoint not found");
});

server.listen(PORT, () => {
  console.log(`🚀 OpenAI Relay server running on port ${PORT}`);
  console.log(`📡 Ready to process real transcriptions via OpenAI`);
});

setInterval(() => pruneTranslationJobs(), 60_000);

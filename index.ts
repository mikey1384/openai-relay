import { createServer } from "node:http";
import { Buffer } from "node:buffer";
import { IncomingForm } from "formidable";
import { randomUUID } from "node:crypto";
import { makeOpenAI, makeGroq } from "./openai-config.js";
import { DEFAULT_TRANSLATION_MODEL } from "./constants.js";

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

type TranslationJobStatus = "queued" | "processing" | "completed" | "failed";

type ChatJobPayload = {
  mode: "chat";
  messages: Array<{ role: string; content: string }>;
  model: string;
  temperature?: number;
  reasoning?: any;
};

type TextJobPayload = {
  mode: "text";
  text: string;
  target_language: string;
  model: string;
  temperature?: number;
};

type TranslationJob = {
  id: string;
  status: TranslationJobStatus;
  createdAt: number;
  updatedAt: number;
  payload: ChatJobPayload | TextJobPayload;
  openaiKey: string;
  result?: any;
  error?: { message: string; details?: string };
};

const translationJobs = new Map<string, TranslationJob>();

function readJsonBody(req: any): Promise<any> {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk: Buffer) => {
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

function pruneTranslationJobs(maxAgeMs = 1000 * 60 * 60) {
  const now = Date.now();
  for (const [jobId, job] of translationJobs) {
    if (
      (job.status === "completed" || job.status === "failed") &&
      now - job.createdAt > maxAgeMs
    ) {
      translationJobs.delete(jobId);
    }
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function extractStatus(error: any): number | null {
  const direct = error?.status ?? error?.response?.status ?? error?.cause?.status;
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

  const code = typeof error?.code === "string" ? error.code.toUpperCase() : null;
  if (
    code &&
    ["ECONNRESET", "ETIMEDOUT", "EHOSTUNREACH", "ENETUNREACH", "ECONNABORTED"].includes(
      code
    )
  ) {
    return true;
  }

  return status == null;
}

async function processTranslationJob(job: TranslationJob): Promise<void> {
  job.status = "processing";
  job.updatedAt = Date.now();

  try {
    const client = makeOpenAI(job.openaiKey);
    const payload = job.payload;
    const model = payload.model || DEFAULT_TRANSLATION_MODEL;
    const isGpt5 = String(model).startsWith("gpt-5");

    if (payload.mode === "chat") {
      const { messages, temperature, reasoning } = payload;

      const request: any = {
        model,
        messages,
      };

      if (reasoning !== undefined) {
        request.reasoning = reasoning;
      }

      if (
        !isGpt5 &&
        typeof temperature === "number" &&
        Number.isFinite(temperature)
      ) {
        request.temperature = temperature;
      }

      let completion;

      try {
        completion = await client.chat.completions.create(request);
      } catch (maybeReasoningError: any) {
        const status =
          maybeReasoningError?.status || maybeReasoningError?.response?.status;
        const msg = String(maybeReasoningError?.message || "").toLowerCase();
        if (reasoning && (status === 400 || msg.includes("reasoning"))) {
          const reqWithoutReasoning: any = { model, messages };
          if (
            !isGpt5 &&
            typeof temperature === "number" &&
            Number.isFinite(temperature)
          ) {
            reqWithoutReasoning.temperature = temperature;
          }
          completion = await client.chat.completions.create(
            reqWithoutReasoning
          );
        } else {
          throw maybeReasoningError;
        }
      }

      job.result = completion;
    } else {
      const { text, target_language, temperature } = payload;

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

      if (
        !isGpt5 &&
        typeof temperature === "number" &&
        Number.isFinite(temperature)
      ) {
        request.temperature = temperature;
      }

      const completion = await client.chat.completions.create(request);
      job.result = completion;
    }

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

  // Enable CORS
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader(
    "Access-Control-Allow-Headers",
    "Content-Type, Authorization, X-Relay-Secret, X-OpenAI-Key, X-Groq-Key"
  );

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

    const relaySecretHeader = req.headers["x-relay-secret"];
    const providedSecret = Array.isArray(relaySecretHeader)
      ? relaySecretHeader[0]
      : relaySecretHeader;
    const expectedSecret = process.env.RELAY_SECRET;
    if (
      !providedSecret ||
      !expectedSecret ||
      providedSecret !== expectedSecret
    ) {
      console.log("❌ Invalid or missing relay secret for /speech");
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Unauthorized - invalid relay secret" }));
      return;
    }

    const openaiKeyHeader = req.headers["x-openai-key"];
    const openaiKey = Array.isArray(openaiKeyHeader)
      ? openaiKeyHeader[0]
      : openaiKeyHeader;
    if (!openaiKey) {
      console.log("❌ Missing OpenAI API key for /speech");
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Unauthorized - missing OpenAI key" }));
      return;
    }

    try {
      let body = "";
      req.on("data", (chunk) => {
        body += chunk.toString();
      });

      req.on("end", async () => {
        try {
          const parsed = JSON.parse(body || "{}");
          const text = parsed.text;
          if (!text || typeof text !== "string") {
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(
              JSON.stringify({ error: "Invalid request: text is required" })
            );
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

          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({
              audioBase64,
              voice,
              model,
              format,
              length: text.length,
            })
          );
        } catch (error: any) {
          console.error(
            "❌ Relay speech synthesis error:",
            error.message || error
          );
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({
              error: "Speech synthesis failed",
              details: error?.message || String(error),
            })
          );
        }
      });
    } catch (error: any) {
      console.error("❌ Relay speech handler error:", error.message || error);
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          error: "Speech synthesis failed",
          details: error?.message || String(error),
        })
      );
    }

    return;
  }

  // Handle POST to /transcribe
  if (req.method === "POST" && req.url === "/transcribe") {
    console.log("📡 Processing transcribe request...");

    // Validate relay secret
    const relaySecretHeader = req.headers["x-relay-secret"];
    const providedSecret = Array.isArray(relaySecretHeader)
      ? relaySecretHeader[0]
      : relaySecretHeader;
    const expectedSecret = process.env.RELAY_SECRET;
    if (
      !providedSecret ||
      !expectedSecret ||
      providedSecret !== expectedSecret
    ) {
      console.log("❌ Invalid or missing relay secret");
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Unauthorized - invalid relay secret" }));
      return;
    }

    console.log("🎯 Relay secret validated, processing transcription...");

    try {
      // Parse the multipart form data
      const form = new IncomingForm();
      const [fields, files] = await form.parse(req);

      const file = Array.isArray(files.file) ? files.file[0] : files.file;
      if (!file) {
        console.log("❌ No file provided");
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "No file provided" }));
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
        `🎵 Transcribing file: ${file.originalFilename} (${file.size} bytes) with model: ${model}`
      );

      // Determine which key and client to use based on model
      let client;
      let usedProvider = "";
      if (model === "whisper-large-v3" || model === "whisper-large-v3-turbo") {
        const groqKey = req.headers["x-groq-key"] as string;
        if (!groqKey) {
          console.log("❌ Missing Groq API key for whisper-large-v3");
          res.writeHead(401, { "Content-Type": "application/json" });
          res.end(JSON.stringify({ error: "Unauthorized - missing Groq key" }));
          return;
        }
        client = makeGroq(groqKey);
        usedProvider = "Groq";
      } else {
        const openaiKey = req.headers["x-openai-key"] as string;
        if (!openaiKey) {
          console.log("❌ Missing OpenAI API key");
          res.writeHead(401, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({ error: "Unauthorized - missing OpenAI key" })
          );
          return;
        }
        client = makeOpenAI(openaiKey);
        usedProvider = "OpenAI";
      }

      // Read the file and create a proper File object
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

      console.log(
        `🎯 Relay transcription completed successfully using ${usedProvider}!`
      );

      // Send the real transcription result
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(transcription));
    } catch (error: any) {
      console.error("❌ Relay transcription error:", error.message);
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          error: "Transcription failed",
          details: error.message,
        })
      );
    }

    return;
  }

  // Handle POST to /translate via async job submission
  if (req.method === "POST" && req.url === "/translate") {
    console.log("🌐 Processing translate request (job)...");

    const relaySecretHeader = req.headers["x-relay-secret"];
    const providedSecret = Array.isArray(relaySecretHeader)
      ? relaySecretHeader[0]
      : relaySecretHeader;
    const expectedSecret = process.env.RELAY_SECRET;
    if (
      !providedSecret ||
      !expectedSecret ||
      providedSecret !== expectedSecret
    ) {
      console.log("❌ Invalid or missing relay secret");
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Unauthorized - invalid relay secret" }));
      return;
    }

    const openaiKeyHeader = req.headers["x-openai-key"];
    const openaiKey = Array.isArray(openaiKeyHeader)
      ? openaiKeyHeader[0]
      : openaiKeyHeader;
    if (!openaiKey) {
      console.log("❌ Missing OpenAI API key for /translate");
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Unauthorized - missing OpenAI key" }));
      return;
    }

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
          temperature: parsed?.temperature,
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
          temperature: parsed?.temperature,
        };
      }

      if (!payload) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            error: "Invalid translation payload",
            details: "Expected messages[] or text/target_language",
          })
        );
        return;
      }

      pruneTranslationJobs();

      const job: TranslationJob = {
        id: randomUUID(),
        status: "queued",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        payload,
        openaiKey,
      };

      translationJobs.set(job.id, job);

      res.writeHead(202, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ jobId: job.id, status: job.status }));

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
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          error: "Translation job submission failed",
          details: error.message,
        })
      );
    }

    return;
  }
  if (
    req.method === "GET" &&
    req.url &&
    req.url.startsWith("/translate/result/")
  ) {
    const relaySecretHeader = req.headers["x-relay-secret"];
    const providedSecret = Array.isArray(relaySecretHeader)
      ? relaySecretHeader[0]
      : relaySecretHeader;
    const expectedSecret = process.env.RELAY_SECRET;
    if (
      !providedSecret ||
      !expectedSecret ||
      providedSecret !== expectedSecret
    ) {
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Unauthorized - invalid relay secret" }));
      return;
    }

    const jobId = decodeURIComponent(
      req.url.split("?")[0].split("/").pop() || ""
    );
    const job = translationJobs.get(jobId);
    if (!job) {
      res.writeHead(404, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Job not found" }));
      return;
    }

    if (job.status === "completed") {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify(job.result ?? {}));
    } else if (job.status === "failed") {
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          error: job.error?.message || "Translation failed",
          details: job.error?.details,
        })
      );
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

    const relaySecretHeader = req.headers["x-relay-secret"];
    const providedSecret = Array.isArray(relaySecretHeader)
      ? relaySecretHeader[0]
      : relaySecretHeader;
    const expectedSecret = process.env.RELAY_SECRET;
    if (
      !providedSecret ||
      !expectedSecret ||
      providedSecret !== expectedSecret
    ) {
      console.log("❌ Invalid or missing relay secret for /dub");
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Unauthorized - invalid relay secret" }));
      return;
    }

    const openaiKeyHeader = req.headers["x-openai-key"];
    const openaiKey = Array.isArray(openaiKeyHeader)
      ? openaiKeyHeader[0]
      : openaiKeyHeader;
    if (!openaiKey) {
      console.log("❌ Missing OpenAI API key for /dub");
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Unauthorized - missing OpenAI key" }));
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
          res.writeHead(413, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({
              error: "Dub request too large",
              details: `Received ${segmentsPayload.length} segments, limit is ${DUB_MAX_SEGMENTS}`,
            })
          );
          return;
        }

        const totalCharacters = segmentsPayload.reduce(
          (sum: number, seg: { text: string }) => sum + seg.text.length,
          0
        );

        if (totalCharacters > DUB_MAX_TOTAL_CHARACTERS) {
          res.writeHead(413, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({
              error: "Dub request too large",
              details: `Received ${totalCharacters} characters, limit is ${DUB_MAX_TOTAL_CHARACTERS}`,
            })
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

        let nextIndex = 0;
        const workerCount = Math.min(DUB_MAX_CONCURRENCY, segmentsPayload.length);

        const workers = Array.from({ length: workerCount }, async (_, workerIdx) => {
          while (true) {
            if (requestClosed) {
              return;
            }

            const current = nextIndex++;
            if (current >= segmentsPayload.length) {
              return;
            }

            const seg = segmentsPayload[current];
            console.log(
              `   • Worker ${workerIdx + 1}/${workerCount} segment ${current + 1}/${segmentsPayload.length} (index=${seg.index}, ${seg.text.length} chars)`
            );
            await synthesizeSegment(current);
            console.log(
              `     · Completed segment ${current + 1}/${segmentsPayload.length}`
            );
          }
        });

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
          res.writeHead(500, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({
              error: "Dub synthesis failed",
              details,
            })
          );
          return;
        }

        if (requestClosed) {
          console.warn(
            "⚠️ Dub request closed before completion; skipping response"
          );
          return;
        }

        const completedSegments = segmentResponses.filter(Boolean);

        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            voice,
            model,
            format,
            segmentCount: completedSegments.length,
            totalCharacters,
            segments: completedSegments,
          })
        );
        return;
      }

      if (!lines.length) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({ error: "Invalid request: lines array required" })
        );
        return;
      }

      const totalCharacters = lines.reduce(
        (sum: number, line: string) => sum + line.length,
        0
      );
      const chunks = chunkLines(lines, MAX_TTS_CHARS_PER_CHUNK);

      if (!chunks.length) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "No valid dialogue for dubbing" }));
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

      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          audioBase64,
          voice,
          model,
          format,
          chunkCount: chunks.length,
          totalCharacters,
        })
      );
      return;
    } catch (error: any) {
      console.error("❌ Relay dub synthesis error:", error?.message || error);
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          error: "Dub synthesis failed",
          details: error?.message || String(error),
        })
      );
      return;
    }
  }

  // Handle all other requests
  console.log(`❌ Unsupported request: ${req.method} ${req.url}`);
  res.writeHead(404, { "Content-Type": "application/json" });
  res.end(JSON.stringify({ error: "Endpoint not found" }));
});

server.listen(PORT, () => {
  console.log(`🚀 OpenAI Relay server running on port ${PORT}`);
  console.log(`📡 Ready to process real transcriptions via OpenAI`);
});

setInterval(() => pruneTranslationJobs(), 60_000);

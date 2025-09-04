import { createServer } from "node:http";
import { IncomingForm } from "formidable";
import { makeOpenAI, makeGroq } from "./openai-config.js";
import { DEFAULT_TRANSLATION_MODEL } from "./constants.js";

const PORT = process.env.PORT || 3000;

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
        [fileBuffer],
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

  // Handle POST to /translate (supports two modes):
  // 1) Simple text translation: { text, target_language, model, temperature }
  // 2) Chat style translation: { messages, model, temperature } (OpenAI chat.completions compatible)
  if (req.method === "POST" && req.url === "/translate") {
    console.log("🌐 Processing translate request...");

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

    // Get OpenAI API key from headers
    const openaiKey = req.headers["x-openai-key"] as string;
    if (!openaiKey) {
      console.log("❌ Missing OpenAI API key");
      res.writeHead(401, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "Unauthorized - missing OpenAI key" }));
      return;
    }

    console.log(
      "🎯 Relay secret and OpenAI key validated, processing translation..."
    );

    try {
      // Parse JSON body
      let body = "";
      req.on("data", (chunk) => {
        body += chunk.toString();
      });

      req.on("end", async () => {
        try {
          const parsed = JSON.parse(body);
          const openai = makeOpenAI(openaiKey);

          // Branch: chat-style translation passthrough
          if (Array.isArray(parsed?.messages)) {
            const { messages, model, temperature, reasoning } = parsed as any;

            let completion;
            try {
              completion = await openai.chat.completions.create({
                model: model || DEFAULT_TRANSLATION_MODEL,
                messages,
                temperature: typeof temperature === "number" ? temperature : 0.3,
                ...(reasoning ? { reasoning } : {}),
              });
            } catch (err: any) {
              const status = err?.status || err?.response?.status;
              const msg = String(err?.message || "").toLowerCase();
              if (reasoning && (status === 400 || msg.includes("reasoning"))) {
                completion = await openai.chat.completions.create({
                  model: model || DEFAULT_TRANSLATION_MODEL,
                  messages,
                  temperature: typeof temperature === "number" ? temperature : 0.3,
                });
              } else {
                throw err;
              }
            }

            console.log("🎯 Relay chat translation completed successfully!");
            res.writeHead(200, { "Content-Type": "application/json" });
            res.end(JSON.stringify(completion));
            return;
          }

          // Simple text translation mode
          const { text, target_language, model, temperature } = parsed as any;
          if (!text || !target_language) {
            console.log("❌ Missing required fields: text or target_language");
            res.writeHead(400, { "Content-Type": "application/json" });
            res.end(
              JSON.stringify({
                error: "Missing required fields: text or target_language",
              })
            );
            return;
          }

          console.log(
            `🌐 Translating to: ${target_language} with model: ${
              model || DEFAULT_TRANSLATION_MODEL
            }`
          );

          const completion = await openai.chat.completions.create({
            model: model || DEFAULT_TRANSLATION_MODEL,
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
            temperature: typeof temperature === "number" ? temperature : 0.3,
          });

          const translatedText = completion.choices[0]?.message?.content || "";

          console.log("🎯 Relay translation completed successfully!");
          res.writeHead(200, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({
              translated_text: translatedText,
              target_language: target_language,
              model: model || DEFAULT_TRANSLATION_MODEL,
            })
          );
        } catch (parseError: any) {
          console.error(
            "❌ Relay translation parse error:",
            parseError.message
          );
          res.writeHead(400, { "Content-Type": "application/json" });
          res.end(
            JSON.stringify({
              error: "Translation failed",
              details: parseError.message,
            })
          );
        }
      });
    } catch (error: any) {
      console.error("❌ Relay translation error:", error.message);
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(
        JSON.stringify({
          error: "Translation failed",
          details: error.message,
        })
      );
    }

    return;
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

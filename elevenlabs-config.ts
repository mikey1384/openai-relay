import { Buffer } from "node:buffer";
import * as fs from "node:fs/promises";

const ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1";
export const ELEVENLABS_TTS_MODEL_ID = "eleven_v3";
export const ELEVENLABS_TTS_MAX_TEXT_CHARACTERS = 5_000;

function usesElevenV3(modelId?: string): boolean {
  return (
    String(modelId || "")
      .trim()
      .toLowerCase() === ELEVENLABS_TTS_MODEL_ID
  );
}

export function assertElevenLabsTtsTextLength(text: string): void {
  if (text.length <= ELEVENLABS_TTS_MAX_TEXT_CHARACTERS) {
    return;
  }

  throw new Error(
    `ElevenLabs v3 accepts at most ${ELEVENLABS_TTS_MAX_TEXT_CHARACTERS} characters per segment`,
  );
}

export interface ScribeWord {
  text: string;
  start: number;
  end: number;
  type?: "word" | "spacing" | "audio_event";
  speaker_id?: string;
}

export interface ScribeUtterance {
  text: string;
  start: number;
  end: number;
  speaker_id?: string;
}

export interface ScribeResult {
  text: string;
  language_code: string;
  language_probability?: number;
  words?: ScribeWord[];
  utterances?: ScribeUtterance[];
  // Computed segments for Whisper compatibility
  segments: Array<{
    text: string;
    start: number;
    end: number;
    words?: Array<{ text: string; start: number; end: number }>;
  }>;
}

type RawScribeResponse = {
  text: string;
  language_code: string;
  language_probability?: number;
  words?: ScribeWord[];
  utterances?: ScribeUtterance[];
};

export function normalizeScribeResult(rawResult: RawScribeResponse): ScribeResult {
  // Convert words into Whisper-compatible segments.
  // This logic matches the BYO ElevenLabs path in ai-provider.ts.
  const words = (rawResult.words || []).filter((w) => w.type === "word");

  const segments: Array<{
    text: string;
    start: number;
    end: number;
    words?: Array<{ text: string; start: number; end: number }>;
  }> = [];

  if (words.length > 0) {
    const SENTENCE_ENDERS = /[.!?。！？]/;
    const MAX_SEGMENT_DURATION = 8; // seconds - keep segments short like Whisper

    let currentSegment: {
      words: typeof words;
      speakerId: string | undefined;
    } = { words: [], speakerId: undefined };

    for (const word of words) {
      const speakerChanged =
        currentSegment.speakerId !== undefined &&
        word.speaker_id !== currentSegment.speakerId;
      const sentenceEnded =
        currentSegment.words.length > 0 &&
        SENTENCE_ENDERS.test(
          currentSegment.words[currentSegment.words.length - 1]?.text || "",
        );
      const tooLong =
        currentSegment.words.length > 0 &&
        word.end - currentSegment.words[0].start > MAX_SEGMENT_DURATION;

      // Start new segment on speaker change, sentence end, or if too long.
      if (
        (speakerChanged || sentenceEnded || tooLong) &&
        currentSegment.words.length > 0
      ) {
        const segWords = currentSegment.words;
        segments.push({
          start: segWords[0].start,
          end: segWords[segWords.length - 1].end,
          text: segWords.map((w) => w.text).join(" "),
          words: segWords.map((w) => ({
            text: w.text,
            start: w.start,
            end: w.end,
          })),
        });
        currentSegment = { words: [], speakerId: word.speaker_id };
      }

      currentSegment.words.push(word);
      currentSegment.speakerId = word.speaker_id;
    }

    // Don't forget the last segment.
    if (currentSegment.words.length > 0) {
      const segWords = currentSegment.words;
      segments.push({
        start: segWords[0].start,
        end: segWords[segWords.length - 1].end,
        text: segWords.map((w) => w.text).join(" "),
        words: segWords.map((w) => ({
          text: w.text,
          start: w.start,
          end: w.end,
        })),
      });
    }
  } else if (rawResult.text) {
    // Last resort: one segment with full text when word timings are unavailable.
    segments.push({
      text: rawResult.text,
      start: 0,
      end: 0,
      words: undefined,
    });
  }

  return {
    text: rawResult.text,
    language_code: rawResult.language_code,
    language_probability: rawResult.language_probability,
    words: rawResult.words,
    utterances: rawResult.utterances,
    segments,
  };
}

/**
 * Transcribe audio using ElevenLabs Scribe API
 */
export async function transcribeWithScribe({
  filePath,
  apiKey,
  languageCode = "auto",
  idempotencyKey,
  signal,
}: {
  filePath: string;
  apiKey: string;
  languageCode?: string;
  idempotencyKey?: string;
  signal?: AbortSignal;
}): Promise<ScribeResult> {
  const fileBuffer = await fs.readFile(filePath);
  const fileName = filePath.split("/").pop() || "audio.webm";

  const formData = new FormData();
  formData.append(
    "file",
    new Blob([new Uint8Array(fileBuffer)], { type: "audio/webm" }),
    fileName,
  );
  formData.append("model_id", "scribe_v2");
  if (languageCode && languageCode !== "auto") {
    formData.append("language_code", languageCode);
  }
  // Request word-level timestamps and enable diarization for utterances
  formData.append("timestamps_granularity", "word");
  formData.append("diarize", "true");

  const response = await fetch(`${ELEVENLABS_API_BASE}/speech-to-text`, {
    method: "POST",
    headers: {
      "xi-api-key": apiKey,
      ...(idempotencyKey ? { "Idempotency-Key": idempotencyKey } : {}),
    },
    body: formData,
    signal,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `ElevenLabs Scribe API error: ${response.status} - ${errorText}`,
    );
  }

  return normalizeScribeResult((await response.json()) as RawScribeResponse);
}

export async function startAsyncTranscriptionWithScribe({
  apiKey,
  cloudStorageUrl,
  languageCode = "auto",
  idempotencyKey,
  webhookId,
  webhookMetadata,
}: {
  apiKey: string;
  cloudStorageUrl: string;
  languageCode?: string;
  idempotencyKey?: string;
  webhookId?: string;
  webhookMetadata?: Record<string, unknown>;
}): Promise<{ request_id: string }> {
  const formData = new FormData();
  formData.append("model_id", "scribe_v2");
  formData.append("cloud_storage_url", cloudStorageUrl);
  formData.append("webhook", "true");
  if (languageCode && languageCode !== "auto") {
    formData.append("language_code", languageCode);
  }
  formData.append("timestamps_granularity", "word");
  formData.append("diarize", "true");
  if (webhookId) {
    formData.append("webhook_id", webhookId);
  }
  if (webhookMetadata && Object.keys(webhookMetadata).length > 0) {
    formData.append("webhook_metadata", JSON.stringify(webhookMetadata));
  }

  const response = await fetch(`${ELEVENLABS_API_BASE}/speech-to-text`, {
    method: "POST",
    headers: {
      "xi-api-key": apiKey,
      ...(idempotencyKey ? { "Idempotency-Key": idempotencyKey } : {}),
    },
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `ElevenLabs async Scribe API error: ${response.status} - ${errorText}`,
    );
  }

  const result = (await response.json()) as { request_id?: string };
  const requestId = String(result?.request_id || "").trim();
  if (!requestId) {
    throw new Error("ElevenLabs async Scribe API returned no request_id");
  }

  return { request_id: requestId };
}

export interface TTSSegment {
  index: number;
  text: string;
  audioBase64: string;
  targetDuration?: number;
}

type ElevenLabsDubFormat = "mp3" | "opus" | "pcm" | "wav";

function resolveElevenLabsDubFormat(format?: string): {
  normalizedFormat: ElevenLabsDubFormat;
  apiOutputFormat: string;
  wrapPcmAsWav?: boolean;
} {
  const normalized = String(format || "mp3")
    .trim()
    .toLowerCase();
  switch (normalized) {
    case "mp3":
      return {
        normalizedFormat: "mp3",
        apiOutputFormat: "mp3_44100_128",
      };
    case "opus":
      return {
        normalizedFormat: "opus",
        apiOutputFormat: "opus_48000_32",
      };
    case "pcm":
      return {
        normalizedFormat: "pcm",
        apiOutputFormat: "pcm_44100",
      };
    case "wav":
      return {
        normalizedFormat: "wav",
        apiOutputFormat: "pcm_44100",
        wrapPcmAsWav: true,
      };
    default:
      throw new Error(
        `ElevenLabs does not support requested output format "${normalized}"`,
      );
  }
}

function wrapPcm16LeAsWav(
  pcmBuffer: Buffer,
  sampleRate = 44_100,
  channels = 1,
  bitsPerSample = 16,
): Buffer {
  const blockAlign = (channels * bitsPerSample) / 8;
  const byteRate = sampleRate * blockAlign;
  const wavHeader = Buffer.alloc(44);
  wavHeader.write("RIFF", 0);
  wavHeader.writeUInt32LE(36 + pcmBuffer.length, 4);
  wavHeader.write("WAVE", 8);
  wavHeader.write("fmt ", 12);
  wavHeader.writeUInt32LE(16, 16);
  wavHeader.writeUInt16LE(1, 20);
  wavHeader.writeUInt16LE(channels, 22);
  wavHeader.writeUInt32LE(sampleRate, 24);
  wavHeader.writeUInt32LE(byteRate, 28);
  wavHeader.writeUInt16LE(blockAlign, 32);
  wavHeader.writeUInt16LE(bitsPerSample, 34);
  wavHeader.write("data", 36);
  wavHeader.writeUInt32LE(pcmBuffer.length, 40);
  return Buffer.concat([wavHeader, pcmBuffer]);
}

/**
 * Synthesize speech using ElevenLabs TTS API
 */
export async function synthesizeWithElevenLabs({
  text,
  voice = "adam",
  modelId = ELEVENLABS_TTS_MODEL_ID,
  format = "mp3",
  apiKey,
  signal,
}: {
  text: string;
  voice?: string;
  modelId?: string;
  format?: string;
  apiKey: string;
  signal?: AbortSignal;
}): Promise<Buffer> {
  assertElevenLabsTtsTextLength(text);

  // ElevenLabs voice IDs - map common names to IDs
  const voiceIdMap: Record<string, string> = {
    adam: "pNInz6obpgDQGcFmaJgB",
    rachel: "21m00Tcm4TlvDq8ikWAM",
    domi: "AZnzlk1XvdvUeBnXmlld",
    bella: "EXAVITQu4vr4xnSDxMaL",
    antoni: "ErXwobaYiN019PkySvjV",
    elli: "MF3mGyEYCl7XYWbV9V6O",
    josh: "TxGEqnHWrfWFTfGW9XjX",
    arnold: "VR6AewLTigWG4xSOukaG",
    sam: "yoZ06aMxZJJ28mfd3POQ",
    // Additional ElevenLabs voices
    sarah: "EXAVITQu4vr4xnSDxMaL", // American, young, soft (same as bella)
    charlie: "IKne3meq5aSn9XLyUdCD", // Australian, middle-aged, casual
    emily: "LcfcDJNUP1GQjkzn1xUU", // American, young, calm
    matilda: "XrExE9yKIg1WjnnlVkGX", // American, middle-aged, warm
    brian: "nPczCjzI2devNBz1zQrb", // American, middle-aged, deep
    // OpenAI voice name mappings for compatibility
    alloy: "pNInz6obpgDQGcFmaJgB", // map to adam
    echo: "VR6AewLTigWG4xSOukaG", // map to arnold
    fable: "MF3mGyEYCl7XYWbV9V6O", // map to elli
    onyx: "TxGEqnHWrfWFTfGW9XjX", // map to josh
    nova: "21m00Tcm4TlvDq8ikWAM", // map to rachel
    shimmer: "EXAVITQu4vr4xnSDxMaL", // map to bella
  };

  const voiceId = voiceIdMap[voice.toLowerCase()] || voice;

  const outputSpec = resolveElevenLabsDubFormat(format);
  const requestBody: Record<string, unknown> = {
    text,
    model_id: modelId,
  };

  if (!usesElevenV3(modelId)) {
    requestBody.voice_settings = {
      stability: 0.5,
      similarity_boost: 0.75,
    };
  }

  const response = await fetch(
    `${ELEVENLABS_API_BASE}/text-to-speech/${voiceId}?output_format=${encodeURIComponent(
      outputSpec.apiOutputFormat,
    )}`,
    {
      method: "POST",
      headers: {
        "xi-api-key": apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
      signal,
    },
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `ElevenLabs TTS API error: ${response.status} - ${errorText}`,
    );
  }

  const audioBuffer = Buffer.from(await response.arrayBuffer());
  if (outputSpec.wrapPcmAsWav) {
    return wrapPcm16LeAsWav(audioBuffer);
  }

  return audioBuffer;
}

/**
 * Test if an ElevenLabs API key is valid
 */
export async function testElevenLabsKey(apiKey: string): Promise<boolean> {
  const response = await fetch(`${ELEVENLABS_API_BASE}/user`, {
    headers: {
      "xi-api-key": apiKey,
    },
  });
  return response.ok;
}

// --- ElevenLabs Dubbing API ---

// Map full language names to ISO 639-1 codes (ElevenLabs requires ISO codes)
const LANGUAGE_CODE_MAP: Record<string, string> = {
  // Common language name mappings
  english: "en",
  spanish: "es",
  french: "fr",
  german: "de",
  italian: "it",
  portuguese: "pt",
  dutch: "nl",
  polish: "pl",
  russian: "ru",
  japanese: "ja",
  korean: "ko",
  chinese: "zh",
  "chinese (simplified)": "zh",
  "chinese (traditional)": "zh",
  arabic: "ar",
  hindi: "hi",
  turkish: "tr",
  vietnamese: "vi",
  thai: "th",
  indonesian: "id",
  malay: "ms",
  filipino: "fil",
  tagalog: "fil",
  swedish: "sv",
  norwegian: "no",
  danish: "da",
  finnish: "fi",
  greek: "el",
  hebrew: "he",
  czech: "cs",
  hungarian: "hu",
  romanian: "ro",
  ukrainian: "uk",
  bulgarian: "bg",
  croatian: "hr",
  slovak: "sk",
  slovenian: "sl",
  serbian: "sr",
  tamil: "ta",
  telugu: "te",
  bengali: "bn",
  urdu: "ur",
  gujarati: "gu",
  kannada: "kn",
  malayalam: "ml",
  marathi: "mr",
  punjabi: "pa",
};

/**
 * Convert language name or code to ISO 639-1 code
 */
function normalizeLanguageCode(lang: string): string {
  if (!lang) return lang;
  const lower = lang.toLowerCase().trim();
  // If it's already a 2-3 character code, assume it's valid
  if (lower.length <= 3) return lower;
  // Look up in mapping
  return LANGUAGE_CODE_MAP[lower] || lower;
}

export interface DubbingJobResponse {
  dubbing_id: string;
  expected_duration_sec: number;
}

export interface DubbingStatusResponse {
  dubbing_id: string;
  name: string;
  status: "dubbing" | "dubbed" | "failed";
  target_languages: string[];
  error?: string;
}

export interface DubbingResult {
  audioBase64: string;
  transcript: string;
  format: string;
}

/**
 * Submit a dubbing job to ElevenLabs
 */
export async function submitDubbingJob({
  fileBuffer,
  fileName,
  mimeType,
  sourceLanguage,
  targetLanguage,
  apiKey,
  numSpeakers,
  dropBackgroundAudio = true,
}: {
  fileBuffer: Buffer;
  fileName: string;
  mimeType: string;
  sourceLanguage?: string;
  targetLanguage: string;
  apiKey: string;
  numSpeakers?: number;
  dropBackgroundAudio?: boolean;
}): Promise<DubbingJobResponse> {
  // Normalize language codes (convert full names like "korean" to ISO codes like "ko")
  const normalizedTargetLang = normalizeLanguageCode(targetLanguage);
  const normalizedSourceLang = sourceLanguage
    ? normalizeLanguageCode(sourceLanguage)
    : undefined;

  const formData = new FormData();
  formData.append(
    "file",
    new Blob([new Uint8Array(fileBuffer)], { type: mimeType }),
    fileName,
  );
  formData.append("target_lang", normalizedTargetLang);
  formData.append("mode", "automatic");
  formData.append("drop_background_audio", String(dropBackgroundAudio));

  if (normalizedSourceLang) {
    formData.append("source_lang", normalizedSourceLang);
  }
  if (numSpeakers && numSpeakers > 0) {
    formData.append("num_speakers", String(numSpeakers));
  }

  const response = await fetch(`${ELEVENLABS_API_BASE}/dubbing`, {
    method: "POST",
    headers: {
      "xi-api-key": apiKey,
    },
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `ElevenLabs Dubbing submit error: ${response.status} - ${errorText}`,
    );
  }

  return (await response.json()) as DubbingJobResponse;
}

/**
 * Get dubbing job status
 */
export async function getDubbingStatus(
  dubbingId: string,
  apiKey: string,
): Promise<DubbingStatusResponse> {
  const response = await fetch(`${ELEVENLABS_API_BASE}/dubbing/${dubbingId}`, {
    headers: {
      "xi-api-key": apiKey,
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `ElevenLabs Dubbing status error: ${response.status} - ${errorText}`,
    );
  }

  return (await response.json()) as DubbingStatusResponse;
}

/**
 * Get dubbed audio
 */
export async function getDubbedAudio(
  dubbingId: string,
  languageCode: string,
  apiKey: string,
): Promise<Buffer> {
  const normalizedLang = normalizeLanguageCode(languageCode);
  const response = await fetch(
    `${ELEVENLABS_API_BASE}/dubbing/${dubbingId}/audio/${normalizedLang}`,
    {
      headers: {
        "xi-api-key": apiKey,
      },
    },
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `ElevenLabs Dubbing audio error: ${response.status} - ${errorText}`,
    );
  }

  const arrayBuffer = await response.arrayBuffer();
  return Buffer.from(arrayBuffer);
}

/**
 * Get dubbed transcript (SRT format)
 */
export async function getDubbedTranscript(
  dubbingId: string,
  languageCode: string,
  apiKey: string,
): Promise<string> {
  const normalizedLang = normalizeLanguageCode(languageCode);
  const response = await fetch(
    `${ELEVENLABS_API_BASE}/dubbing/${dubbingId}/transcript/${normalizedLang}?format_type=srt`,
    {
      headers: {
        "xi-api-key": apiKey,
      },
    },
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `ElevenLabs Dubbing transcript error: ${response.status} - ${errorText}`,
    );
  }

  // ElevenLabs returns plain SRT text when format_type=srt is specified
  const text = await response.text();

  // If it looks like JSON, try to parse it
  if (text.startsWith("{")) {
    try {
      const data = JSON.parse(text);
      return data.srt || text;
    } catch {
      return text;
    }
  }

  return text;
}

/**
 * Full dubbing workflow: submit, poll, and get results
 */
export async function dubWithElevenLabs({
  fileBuffer,
  fileName,
  mimeType,
  sourceLanguage,
  targetLanguage,
  apiKey,
  numSpeakers,
  dropBackgroundAudio = true,
  pollIntervalMs = 5000,
  maxWaitMs = 600000, // 10 minutes
  onProgress,
}: {
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
}): Promise<DubbingResult> {
  // Submit job
  onProgress?.("Submitting dubbing job...");
  const { dubbing_id, expected_duration_sec } = await submitDubbingJob({
    fileBuffer,
    fileName,
    mimeType,
    sourceLanguage,
    targetLanguage,
    apiKey,
    numSpeakers,
    dropBackgroundAudio,
  });

  console.log(
    `🎬 Dubbing job submitted: ${dubbing_id} (expected: ${expected_duration_sec}s)`,
  );

  // Poll for completion
  const startTime = Date.now();
  let status: DubbingStatusResponse;

  while (true) {
    if (Date.now() - startTime > maxWaitMs) {
      throw new Error("Dubbing job timed out");
    }

    status = await getDubbingStatus(dubbing_id, apiKey);
    onProgress?.(`Dubbing status: ${status.status}`);

    if (status.status === "dubbed") {
      break;
    }

    if (status.status === "failed") {
      throw new Error(`Dubbing failed: ${status.error || "Unknown error"}`);
    }

    // Wait before next poll
    await new Promise((resolve) => setTimeout(resolve, pollIntervalMs));
  }

  // Get results
  onProgress?.("Retrieving dubbed audio...");
  const audioBuffer = await getDubbedAudio(dubbing_id, targetLanguage, apiKey);

  onProgress?.("Retrieving transcript...");
  const transcript = await getDubbedTranscript(
    dubbing_id,
    targetLanguage,
    apiKey,
  );

  return {
    audioBase64: audioBuffer.toString("base64"),
    transcript,
    format: "mp3",
  };
}

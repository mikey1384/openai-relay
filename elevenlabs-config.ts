import { Buffer } from "node:buffer";
import * as fs from "node:fs/promises";

const ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1";

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

/**
 * Transcribe audio using ElevenLabs Scribe API
 */
export async function transcribeWithScribe({
  filePath,
  apiKey,
  languageCode = "auto",
}: {
  filePath: string;
  apiKey: string;
  languageCode?: string;
}): Promise<ScribeResult> {
  const fileBuffer = await fs.readFile(filePath);
  const fileName = filePath.split("/").pop() || "audio.webm";

  const formData = new FormData();
  formData.append(
    "file",
    new Blob([new Uint8Array(fileBuffer)], { type: "audio/webm" }),
    fileName
  );
  formData.append("model_id", "scribe_v1");
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
    },
    body: formData,
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `ElevenLabs Scribe API error: ${response.status} - ${errorText}`
    );
  }

  const rawResult = (await response.json()) as {
    text: string;
    language_code: string;
    language_probability?: number;
    words?: ScribeWord[];
    utterances?: ScribeUtterance[];
  };

  // Convert words into Whisper-compatible segments
  // This logic matches the BYO ElevenLabs path in ai-provider.ts
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
          currentSegment.words[currentSegment.words.length - 1]?.text || ""
        );
      const tooLong =
        currentSegment.words.length > 0 &&
        word.end - currentSegment.words[0].start > MAX_SEGMENT_DURATION;

      // Start new segment on speaker change, sentence end, or if too long
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

    // Don't forget the last segment
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
    // Last resort: one segment with full text (no word-level data available)
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

export interface TTSSegment {
  index: number;
  text: string;
  audioBase64: string;
  targetDuration?: number;
}

/**
 * Synthesize speech using ElevenLabs TTS API
 */
export async function synthesizeWithElevenLabs({
  text,
  voice = "adam",
  modelId = "eleven_multilingual_v2",
  apiKey,
}: {
  text: string;
  voice?: string;
  modelId?: string;
  apiKey: string;
}): Promise<Buffer> {
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

  const response = await fetch(
    `${ELEVENLABS_API_BASE}/text-to-speech/${voiceId}`,
    {
      method: "POST",
      headers: {
        "xi-api-key": apiKey,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text,
        model_id: modelId,
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.75,
        },
      }),
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `ElevenLabs TTS API error: ${response.status} - ${errorText}`
    );
  }

  const arrayBuffer = await response.arrayBuffer();
  return Buffer.from(arrayBuffer);
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
    fileName
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
      `ElevenLabs Dubbing submit error: ${response.status} - ${errorText}`
    );
  }

  return (await response.json()) as DubbingJobResponse;
}

/**
 * Get dubbing job status
 */
export async function getDubbingStatus(
  dubbingId: string,
  apiKey: string
): Promise<DubbingStatusResponse> {
  const response = await fetch(`${ELEVENLABS_API_BASE}/dubbing/${dubbingId}`, {
    headers: {
      "xi-api-key": apiKey,
    },
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `ElevenLabs Dubbing status error: ${response.status} - ${errorText}`
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
  apiKey: string
): Promise<Buffer> {
  const normalizedLang = normalizeLanguageCode(languageCode);
  const response = await fetch(
    `${ELEVENLABS_API_BASE}/dubbing/${dubbingId}/audio/${normalizedLang}`,
    {
      headers: {
        "xi-api-key": apiKey,
      },
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `ElevenLabs Dubbing audio error: ${response.status} - ${errorText}`
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
  apiKey: string
): Promise<string> {
  const normalizedLang = normalizeLanguageCode(languageCode);
  const response = await fetch(
    `${ELEVENLABS_API_BASE}/dubbing/${dubbingId}/transcript/${normalizedLang}?format_type=srt`,
    {
      headers: {
        "xi-api-key": apiKey,
      },
    }
  );

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `ElevenLabs Dubbing transcript error: ${response.status} - ${errorText}`
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
    `🎬 Dubbing job submitted: ${dubbing_id} (expected: ${expected_duration_sec}s)`
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
    apiKey
  );

  return {
    audioBase64: audioBuffer.toString("base64"),
    transcript,
    format: "mp3",
  };
}

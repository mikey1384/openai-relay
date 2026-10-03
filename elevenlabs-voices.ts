// Stage5 managed dubbing is ElevenLabs-only: OpenAI's TTS models (tts-1,
// tts-1-hd, gpt-4o-mini-tts) shut down on 2027-01-06. This module has no
// imports so the voice/format normalization can be unit-tested in isolation.

/** Default ElevenLabs voice (what the old OpenAI default "alloy" maps to). */
export const ELEVENLABS_DEFAULT_DUB_VOICE = "adam";

/** ElevenLabs voice IDs by name. */
export const ELEVENLABS_VOICE_IDS: Record<string, string> = {
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
};

/**
 * Legacy OpenAI voice names that older Translator versions still send, mapped
 * onto the ElevenLabs voices the Translator offers (rachel, adam, josh, sarah,
 * charlie, emily, matilda, brian). stage5-api keeps an identical copy in
 * src/lib/constants.ts (OPENAI_TO_ELEVENLABS_VOICE); keep the two in sync.
 */
export const OPENAI_TO_ELEVENLABS_VOICE: Record<string, string> = {
  alloy: "adam",
  echo: "brian",
  fable: "emily",
  onyx: "josh",
  nova: "rachel",
  shimmer: "sarah",
};

/**
 * Resolve a requested voice to the ElevenLabs voice name we synthesize with.
 * Legacy OpenAI names are mapped, known ElevenLabs names are lower-cased, and
 * anything else (e.g. a raw ElevenLabs voice ID) passes through unchanged.
 */
export function resolveElevenLabsDubVoice(voice?: unknown): string {
  const raw = typeof voice === "string" ? voice.trim() : "";
  if (!raw) return ELEVENLABS_DEFAULT_DUB_VOICE;
  const key = raw.toLowerCase();
  if (OPENAI_TO_ELEVENLABS_VOICE[key]) return OPENAI_TO_ELEVENLABS_VOICE[key];
  if (ELEVENLABS_VOICE_IDS[key]) return key;
  return raw;
}

export function resolveElevenLabsVoiceId(voice?: unknown): string {
  const name = resolveElevenLabsDubVoice(voice);
  return ELEVENLABS_VOICE_IDS[name] || name;
}

export const ELEVENLABS_DUB_FORMATS = ["mp3", "opus", "pcm", "wav"] as const;
export type ElevenLabsDubFormatName = (typeof ELEVENLABS_DUB_FORMATS)[number];

/**
 * OpenAI also accepted aac/flac; ElevenLabs cannot produce them, so legacy
 * requests for an unsupported format get mp3 (the response's `format` field
 * tells the client what it received).
 */
export function resolveElevenLabsDubFormatName(
  format?: unknown,
): ElevenLabsDubFormatName {
  const normalized =
    typeof format === "string" ? format.trim().toLowerCase() : "";
  return (ELEVENLABS_DUB_FORMATS as readonly string[]).includes(normalized)
    ? (normalized as ElevenLabsDubFormatName)
    : "mp3";
}

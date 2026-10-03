// Stage5 transcription is ElevenLabs Scribe only. OpenAI whisper-1 (and
// gpt-4o-transcribe) shut down 2027-02-26 and their replacement returns no
// timestamps, so there is no OpenAI fallback: transient Scribe failures are
// retried here, and if Scribe stays unavailable the caller answers 502
// transcription-provider-unavailable.

/** Billing model id stage5-api prices transcription with. */
export const STAGE5_SCRIBE_BILLING_MODEL = "elevenlabs-scribe";

export const TRANSCRIPTION_PROVIDER_UNAVAILABLE_ERROR =
  "transcription-provider-unavailable";
export const TRANSCRIPTION_PROVIDER_UNAVAILABLE_MESSAGE =
  "Transcription is temporarily unavailable. Please try again in a few minutes.";

export const ELEVENLABS_KEY_REQUIRED_ERROR = "elevenlabs-key-required";
export const ELEVENLABS_KEY_REQUIRED_MESSAGE =
  "Transcription uses ElevenLabs. Add an ElevenLabs API key in Settings, or turn off API key mode to use Stage5 credits.";

const RETRYABLE_SCRIBE_STATUSES = new Set([408, 409, 425, 429]);
// Local file errors are not a provider problem and never heal on retry.
const LOCAL_FILE_ERROR_CODES = new Set(["ENOENT", "EACCES", "EISDIR", "EMFILE"]);

export function extractScribeErrorStatus(error: any): number | null {
  const direct = error?.status ?? error?.response?.status ?? error?.cause?.status;
  if (typeof direct === "number" && Number.isFinite(direct)) {
    return direct;
  }
  if (typeof direct === "string" && /^\d{3}$/.test(direct.trim())) {
    return Number.parseInt(direct.trim(), 10);
  }
  // transcribeWithScribe errors read "ElevenLabs Scribe API error: <status> - ...".
  const match = String(error?.message || "").match(/API error: (\d{3})\b/);
  return match ? Number.parseInt(match[1], 10) : null;
}

/**
 * Transient = worth retrying and, once retries are exhausted, reported as
 * "temporarily unavailable": 5xx, 408/409/425/429, timeouts and network
 * failures. Other 4xx (bad audio, bad key, quota) fail at once.
 */
export function isTransientScribeError(error: any): boolean {
  if (error?.name === "AbortError") return false;
  const status = extractScribeErrorStatus(error);
  if (status !== null) {
    return status >= 500 || RETRYABLE_SCRIBE_STATUSES.has(status);
  }
  if (LOCAL_FILE_ERROR_CODES.has(String(error?.code || "").toUpperCase())) {
    return false;
  }
  // No HTTP status means Scribe never answered (timeout, reset, DNS, ...).
  return true;
}

/** True once Scribe retries ran out on a transient failure. */
export function isScribeProviderUnavailableError(error: any): boolean {
  return Boolean(error && typeof error === "object" && error.scribeUnavailable === true);
}

export async function transcribeWithScribeRetrying<TResult>({
  transcribe,
  maxAttempts,
  baseDelayMs,
  maxDelayMs,
  sleep,
  contextLabel,
  signal,
}: {
  transcribe: () => Promise<TResult>;
  maxAttempts: number;
  baseDelayMs: number;
  maxDelayMs: number;
  sleep: (ms: number, signal?: AbortSignal) => Promise<void>;
  contextLabel: string;
  signal?: AbortSignal;
}): Promise<{ result: TResult; attempts: number }> {
  const attemptsAllowed = Math.max(1, Math.floor(maxAttempts));
  let lastError: any = null;
  let attempts = 0;

  for (let attempt = 1; attempt <= attemptsAllowed; attempt += 1) {
    attempts = attempt;
    if (signal?.aborted) break;
    try {
      return { result: await transcribe(), attempts };
    } catch (error: any) {
      lastError = error;
      if (signal?.aborted || error?.name === "AbortError") break;
      if (!isTransientScribeError(error) || attempt >= attemptsAllowed) break;

      const delay = Math.min(maxDelayMs, baseDelayMs * Math.pow(2, attempt - 1));
      console.warn(
        `⚠️ ${contextLabel} ElevenLabs Scribe attempt ${attempt}/${attemptsAllowed} failed, retrying in ${delay}ms: ${
          error?.message || String(error)
        }`,
      );
      await sleep(delay, signal);
    }
  }

  const error =
    lastError && typeof lastError === "object"
      ? lastError
      : new Error(`${contextLabel} ElevenLabs Scribe failed without a response`);
  (error as any).scribeAttempts = attempts;
  if (!signal?.aborted && error?.name !== "AbortError" && isTransientScribeError(error)) {
    (error as any).scribeUnavailable = true;
    console.error(
      `❌ ${contextLabel} ElevenLabs Scribe unavailable after ${attempts} attempt(s): ${
        error?.message || String(error)
      }`,
    );
  }
  throw error;
}

/**
 * Shape a Scribe result as the transcription response every Translator
 * version parses (text, segments with start/end/text/words, words, duration).
 */
export function toTranscriptionResponse(result: any) {
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
    model: STAGE5_SCRIBE_BILLING_MODEL,
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

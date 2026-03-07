import { spawn } from "node:child_process";

const FFPROBE_PATH = String(process.env.FFPROBE_PATH || "ffprobe").trim() || "ffprobe";
const AUDIO_PROBE_TIMEOUT_MS = Math.max(
  1_000,
  Number.parseInt(process.env.AUDIO_PROBE_TIMEOUT_MS || "15000", 10),
);

function asPositiveNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value) && value > 0) {
    return value;
  }
  if (typeof value === "string" && value.trim()) {
    const parsed = Number(value);
    if (Number.isFinite(parsed) && parsed > 0) {
      return parsed;
    }
  }
  return null;
}

function extractProbeDurationSeconds(payload: any): number | null {
  const candidates: number[] = [];

  const formatDuration = asPositiveNumber(payload?.format?.duration);
  if (formatDuration !== null) {
    candidates.push(formatDuration);
  }

  const streams = Array.isArray(payload?.streams) ? payload.streams : [];
  for (const stream of streams) {
    const streamDuration = asPositiveNumber(stream?.duration);
    if (streamDuration !== null) {
      candidates.push(streamDuration);
    }
  }

  if (!candidates.length) {
    return null;
  }

  return Math.max(...candidates);
}

export async function probeMediaDurationSeconds(filePath: string): Promise<number> {
  return new Promise((resolve, reject) => {
    const args = [
      "-v",
      "error",
      "-print_format",
      "json",
      "-show_entries",
      "format=duration:stream=duration",
      filePath,
    ];
    const child = spawn(FFPROBE_PATH, args, {
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    const timeoutId = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error("Audio duration probe timed out"));
    }, AUDIO_PROBE_TIMEOUT_MS);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", (error) => {
      clearTimeout(timeoutId);
      reject(new Error(`Failed to start ffprobe: ${error.message}`));
    });
    child.on("close", (code) => {
      clearTimeout(timeoutId);
      if (code !== 0) {
        reject(
          new Error(
            stderr.trim() || `ffprobe exited with code ${String(code)}`
          )
        );
        return;
      }

      try {
        const payload = JSON.parse(stdout || "{}");
        const durationSeconds = extractProbeDurationSeconds(payload);
        if (durationSeconds === null) {
          reject(new Error("Unable to determine audio duration from media metadata"));
          return;
        }
        resolve(durationSeconds);
      } catch (error: any) {
        reject(
          new Error(
            `Failed to parse ffprobe output: ${error?.message || String(error)}`
          )
        );
      }
    });
  });
}

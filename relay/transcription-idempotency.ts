import crypto from "node:crypto";
import { createReadStream } from "node:fs";
import { STAGE5_RELAY_BILLING_SERVICES } from "./relay-billing-contract.js";
import { buildRelayRequestKey } from "./relay-billing-helpers.js";

async function hashFilePathContents(filePath: string): Promise<string> {
  const hasher = crypto.createHash("sha256");
  await new Promise<void>((resolve, reject) => {
    const stream = createReadStream(filePath);
    stream.on("data", (chunk) => {
      hasher.update(chunk);
    });
    stream.on("error", reject);
    stream.on("end", resolve);
  });
  return hasher.digest("hex").slice(0, 32);
}

export async function buildDirectRelayTranscriptionRequestKey({
  deviceId,
  clientIdempotencyKey,
  filePath,
  language,
  prompt,
  modelHint,
  modelIdHint,
  qualityMode,
}: {
  deviceId: string;
  clientIdempotencyKey?: string;
  filePath: string;
  language: string | null;
  prompt: string | null;
  modelHint: string | null;
  modelIdHint: string | null;
  qualityMode: string | null;
}): Promise<string> {
  const trimmedIdempotencyKey = String(clientIdempotencyKey || "").trim();
  if (!trimmedIdempotencyKey) {
    return buildRelayRequestKey({
      service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
      deviceId,
      clientIdempotencyKey: undefined,
      payload: null,
    });
  }

  // Match direct transcription retries on file contents plus request semantics,
  // not temp filenames or other upload metadata.
  const fileFingerprint = await hashFilePathContents(filePath);
  return buildRelayRequestKey({
    service: STAGE5_RELAY_BILLING_SERVICES.TRANSCRIPTION,
    deviceId,
    clientIdempotencyKey: trimmedIdempotencyKey,
    payload: {
      fileFingerprint,
      language,
      prompt,
      modelHint,
      modelIdHint,
      qualityMode,
    },
  });
}

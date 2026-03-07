import crypto from "node:crypto";
import type { IncomingMessage } from "node:http";

function normalizeValue(value: unknown): unknown {
  if (value == null) return null;
  if (
    typeof value === "string" ||
    typeof value === "boolean" ||
    typeof value === "number"
  ) {
    return value;
  }
  if (Array.isArray(value)) {
    return value.map((item) => normalizeValue(item));
  }
  if (typeof value === "object") {
    const source = value as Record<string, unknown>;
    return Object.fromEntries(
      Object.entries(source)
        .filter(([, entry]) => entry !== undefined)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, entry]) => [key, normalizeValue(entry)])
    );
  }
  return String(value);
}

function hashText(value: string, size = 24): string {
  return crypto.createHash("sha256").update(value).digest("hex").slice(0, size);
}

export function buildRelayRequestKey({
  service,
  deviceId,
  clientIdempotencyKey,
  payload,
}: {
  service: string;
  deviceId: string;
  clientIdempotencyKey?: string;
  payload: unknown;
}): string {
  const requestKey = String(clientIdempotencyKey || "").trim();
  if (!requestKey) {
    return `${service}:ephemeral:${crypto.randomUUID()}`;
  }

  const payloadHash = hashText(JSON.stringify(normalizeValue(payload)));
  const requestHash = hashText(requestKey);
  const deviceHash = hashText(deviceId);
  return `${service}:device:${deviceHash}:req:${requestHash}:payload:${payloadHash}`;
}

export function getInternalRelayBillingContext(
  req: IncomingMessage,
  getHeader: (req: IncomingMessage, name: string) => string | undefined
): { deviceId: string; requestKey: string } | null {
  const deviceId = String(getHeader(req, "x-stage5-device-id") || "").trim();
  const requestKey = String(getHeader(req, "x-stage5-request-key") || "").trim();
  if (!deviceId || !requestKey) {
    return null;
  }
  return { deviceId, requestKey };
}

import type { IncomingMessage, ServerResponse } from "node:http";

const TRANSLATOR_VERSION_HEADER = "x-stage5-app-version";
const DEFAULT_DOWNLOAD_URL = "https://stage5.tools";
const DEFAULT_UPDATE_REQUIRED_MESSAGE =
  "A newer version of Translator is required to continue. Please update the app.";
const DIRECT_CLIENT_PATHS = new Set([
  "/translate-direct",
  "/transcribe-direct",
  "/dub-direct",
]);

type EnforcementMode = "off" | "log" | "enforce";

type SendJson = (res: ServerResponse, data: unknown, status?: number) => void;

function normalizeVersion(raw: string | undefined | null): string {
  return String(raw || "").trim().replace(/^v/i, "");
}

function parseVersionParts(raw: string | undefined | null): number[] | null {
  const normalized = normalizeVersion(raw);
  if (!normalized) return null;
  const match = normalized.match(/^\d+(?:\.\d+){0,3}/);
  if (!match) return null;
  return match[0].split(".").map((part) => Number.parseInt(part, 10));
}

function compareVersionParts(a: number[], b: number[]): number {
  const maxLength = Math.max(a.length, b.length);
  for (let i = 0; i < maxLength; i += 1) {
    const left = a[i] ?? 0;
    const right = b[i] ?? 0;
    if (left > right) return 1;
    if (left < right) return -1;
  }
  return 0;
}

function resolveEnforcementMode(raw: string | undefined): EnforcementMode {
  const normalized = String(raw || "")
    .trim()
    .toLowerCase();
  if (normalized === "log") return "log";
  if (normalized === "enforce") return "enforce";
  return "off";
}

export function enforceMinimumTranslatorVersion(params: {
  req: IncomingMessage;
  res: ServerResponse;
  sendJson: SendJson;
}): boolean {
  const pathname = new URL(params.req.url || "/", "http://localhost").pathname;
  if (!DIRECT_CLIENT_PATHS.has(pathname)) {
    return false;
  }

  const minVersion = normalizeVersion(process.env.MIN_TRANSLATOR_VERSION);
  const mode = resolveEnforcementMode(
    process.env.MIN_TRANSLATOR_VERSION_ENFORCEMENT
  );
  if (!minVersion || mode === "off") {
    return false;
  }

  const minParts = parseVersionParts(minVersion);
  if (!minParts) {
    console.error(
      `[translator-version-gate] Invalid MIN_TRANSLATOR_VERSION: ${process.env.MIN_TRANSLATOR_VERSION}`
    );
    return false;
  }

  const clientVersion = normalizeVersion(
    Array.isArray(params.req.headers[TRANSLATOR_VERSION_HEADER])
      ? params.req.headers[TRANSLATOR_VERSION_HEADER]?.[0]
      : params.req.headers[TRANSLATOR_VERSION_HEADER]
  );
  const clientParts = parseVersionParts(clientVersion);
  if (clientParts && compareVersionParts(clientParts, minParts) >= 0) {
    return false;
  }

  console.warn(
    `[translator-version-gate] ${mode} path=${pathname} client=${clientVersion || "missing"} min=${minVersion}`
  );

  if (mode !== "enforce") {
    return false;
  }

  params.sendJson(
    params.res,
    {
      error: "update-required",
      message: DEFAULT_UPDATE_REQUIRED_MESSAGE,
      minVersion,
      clientVersion: clientVersion || undefined,
      downloadUrl: process.env.TRANSLATOR_DOWNLOAD_URL || DEFAULT_DOWNLOAD_URL,
      source: "relay",
    },
    426
  );
  return true;
}

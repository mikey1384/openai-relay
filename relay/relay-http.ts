import type { IncomingMessage, ServerResponse } from "node:http";

/**
 * Validate relay secret from request headers.
 */
export function validateRelaySecret(
  req: IncomingMessage,
  relaySecret: string
): boolean {
  const relaySecretHeader = req.headers["x-relay-secret"];
  const providedSecret = Array.isArray(relaySecretHeader)
    ? relaySecretHeader[0]
    : relaySecretHeader;
  return providedSecret === relaySecret;
}

/**
 * Extract a single header value (handles array headers).
 */
export function getHeader(
  req: IncomingMessage,
  name: string
): string | undefined {
  const value = req.headers[name.toLowerCase()];
  if (Array.isArray(value)) return value[0];
  return value || undefined;
}

/**
 * Send a JSON error response.
 */
export function sendError(
  res: ServerResponse,
  status: number,
  error: string,
  details?: string
): void {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(details ? { error, details } : { error }));
}

/**
 * Send a JSON success response.
 */
export function sendJson(
  res: ServerResponse,
  data: unknown,
  status = 200
): void {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(data));
}

/**
 * Get CORS origin based on request origin and allowlist.
 */
export function getCorsOrigin(
  req: IncomingMessage,
  allowedOrigins: string[]
): string {
  const requestOrigin = getHeader(req, "origin");

  if (allowedOrigins.includes("*")) {
    return "*";
  }

  if (requestOrigin && allowedOrigins.includes(requestOrigin)) {
    return requestOrigin;
  }

  return allowedOrigins[0] || "*";
}

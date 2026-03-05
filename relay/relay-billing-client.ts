import {
  buildStage5RelayBillingUrl,
  STAGE5_RELAY_BILLING_ENDPOINTS,
  STAGE5_RELAY_BILLING_SERVICES,
} from "./relay-billing-contract.js";

type RelayFailure = {
  ok: false;
  status: number;
  error: string;
};

type RelayAuthorizeSuccess = {
  ok: true;
  deviceId: string;
  creditBalance: number;
};

type RelayDeductSuccess = {
  ok: true;
};

export type RelayAuthorizeResult = RelayAuthorizeSuccess | RelayFailure;
export type RelayDeductResult = RelayDeductSuccess | RelayFailure;

type RelayBillingService =
  (typeof STAGE5_RELAY_BILLING_SERVICES)[keyof typeof STAGE5_RELAY_BILLING_SERVICES];

function formatTransportFailureMessage(
  fallback: string,
  error: unknown
): string {
  const detail =
    error instanceof Error
      ? error.message
      : typeof error === "string"
        ? error
        : "";
  if (!detail) return fallback;
  return `${fallback}: ${detail}`;
}

function extractErrorMessage(payload: unknown, fallback: string): string {
  if (payload && typeof payload === "object" && !Array.isArray(payload)) {
    const err = (payload as any).error;
    if (typeof err === "string" && err.trim()) return err.trim();
  }
  return fallback;
}

async function safeJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

export async function authorizeRelayDevice({
  cfApiBase,
  relaySecret,
  apiKey,
}: {
  cfApiBase: string;
  relaySecret: string;
  apiKey: string;
}): Promise<RelayAuthorizeResult> {
  let authRes: Response;
  try {
    authRes = await fetch(
      buildStage5RelayBillingUrl(
        cfApiBase,
        STAGE5_RELAY_BILLING_ENDPOINTS.AUTHORIZE
      ),
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Relay-Secret": relaySecret,
        },
        body: JSON.stringify({ apiKey }),
      }
    );
  } catch (error) {
    return {
      ok: false,
      status: 503,
      error: formatTransportFailureMessage(
        "Authorization backend unavailable",
        error
      ),
    };
  }

  const payload = await safeJson(authRes);
  if (!authRes.ok) {
    return {
      ok: false,
      status: authRes.status,
      error: extractErrorMessage(payload, "Authorization failed"),
    };
  }

  const deviceId = String((payload as any)?.deviceId || "").trim();
  const creditBalance = Number((payload as any)?.creditBalance || 0);
  if (!deviceId) {
    return {
      ok: false,
      status: 500,
      error: "Authorization failed",
    };
  }

  return {
    ok: true,
    deviceId,
    creditBalance: Number.isFinite(creditBalance) ? creditBalance : 0,
  };
}

export async function deductRelayCredits({
  cfApiBase,
  relaySecret,
  payload,
}: {
  cfApiBase: string;
  relaySecret: string;
  payload: {
    deviceId: string;
    service: RelayBillingService;
    seconds?: number;
    promptTokens?: number;
    completionTokens?: number;
    characters?: number;
    model?: string;
    idempotencyKey?: string;
  };
}): Promise<RelayDeductResult> {
  let deductRes: Response;
  try {
    deductRes = await fetch(
      buildStage5RelayBillingUrl(cfApiBase, STAGE5_RELAY_BILLING_ENDPOINTS.DEDUCT),
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Relay-Secret": relaySecret,
        },
        body: JSON.stringify(payload),
      }
    );
  } catch (error) {
    return {
      ok: false,
      status: 503,
      error: formatTransportFailureMessage(
        "Credit deduction backend unavailable",
        error
      ),
    };
  }

  if (!deductRes.ok) {
    const errPayload = await safeJson(deductRes);
    return {
      ok: false,
      status: deductRes.status,
      error: extractErrorMessage(errPayload, "Credit deduction failed"),
    };
  }

  return { ok: true };
}

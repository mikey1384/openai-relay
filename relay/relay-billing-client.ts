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

type RelayMutationSuccess = {
  ok: true;
  status?: string;
  reservationStatus?: "reserved" | "settled" | "released";
  reservationMeta?: unknown;
  reservationUpdatedAt?: string;
};

export type RelayAuthorizeResult = RelayAuthorizeSuccess | RelayFailure;
export type RelayMutationResult = RelayMutationSuccess | RelayFailure;

export type RelayReplayArtifactRef = {
  version: 1;
  storage: "r2";
  key: string;
  contentType: "application/json";
  sizeBytes: number;
};

type RelayBillingService =
  (typeof STAGE5_RELAY_BILLING_SERVICES)[keyof typeof STAGE5_RELAY_BILLING_SERVICES];

export type RelayBillingPayload = {
  deviceId: string;
  service: RelayBillingService;
  requestKey: string;
  seconds?: number;
  promptTokens?: number;
  maxCompletionTokens?: number;
  completionTokens?: number;
  webSearchCalls?: number;
  characters?: number;
  model?: string;
  meta?: unknown;
};

function isReplayArtifactRef(value: unknown): value is RelayReplayArtifactRef {
  return (
    !!value &&
    typeof value === "object" &&
    !Array.isArray(value) &&
    (value as any).version === 1 &&
    (value as any).storage === "r2" &&
    (value as any).contentType === "application/json" &&
    typeof (value as any).key === "string" &&
    (value as any).key.trim().length > 0 &&
    typeof (value as any).sizeBytes === "number" &&
    Number.isFinite((value as any).sizeBytes) &&
    (value as any).sizeBytes >= 0
  );
}

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

function asObject(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

async function mutateRelayBilling({
  cfApiBase,
  relaySecret,
  endpoint,
  payload,
  unavailableMessage,
  failureMessage,
}: {
  cfApiBase: string;
  relaySecret: string;
  endpoint: (typeof STAGE5_RELAY_BILLING_ENDPOINTS)[keyof typeof STAGE5_RELAY_BILLING_ENDPOINTS];
  payload: RelayBillingPayload;
  unavailableMessage: string;
  failureMessage: string;
}): Promise<RelayMutationResult> {
  let response: Response;
  try {
    response = await fetch(buildStage5RelayBillingUrl(cfApiBase, endpoint), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Relay-Secret": relaySecret,
      },
      body: JSON.stringify(payload),
    });
  } catch (error) {
    return {
      ok: false,
      status: 503,
      error: formatTransportFailureMessage(unavailableMessage, error),
    };
  }

  if (!response.ok) {
    const errPayload = await safeJson(response);
    return {
      ok: false,
      status: response.status,
      error: extractErrorMessage(errPayload, failureMessage),
    };
  }

  const payloadBody = await safeJson(response);
  const payloadObject = asObject(payloadBody);
  const reservationStatusRaw = payloadObject?.reservationStatus;
  const reservationStatus =
    reservationStatusRaw === "reserved" ||
    reservationStatusRaw === "settled" ||
    reservationStatusRaw === "released"
      ? reservationStatusRaw
      : undefined;
  const hasReservationMeta =
    payloadObject &&
    Object.prototype.hasOwnProperty.call(payloadObject, "reservationMeta");
  const reservationUpdatedAt =
    typeof payloadObject?.reservationUpdatedAt === "string" &&
    payloadObject.reservationUpdatedAt.trim()
      ? payloadObject.reservationUpdatedAt.trim()
      : undefined;
  return {
    ok: true,
    status: payloadObject ? String(payloadObject.status || "") : "",
    ...(reservationStatus ? { reservationStatus } : {}),
    ...(hasReservationMeta
      ? { reservationMeta: payloadObject?.reservationMeta }
      : {}),
    ...(reservationUpdatedAt ? { reservationUpdatedAt } : {}),
  };
}

export async function authorizeRelayDevice({
  cfApiBase,
  relaySecret,
  apiKey,
  service,
  clientIdempotencyKey,
  appVersion,
}: {
  cfApiBase: string;
  relaySecret: string;
  apiKey: string;
  service?: RelayBillingService;
  clientIdempotencyKey?: string;
  appVersion?: string;
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
          ...(appVersion
            ? { "X-Stage5-App-Version": appVersion }
            : {}),
        },
        body: JSON.stringify({
          apiKey,
          ...(service ? { service } : {}),
          ...(clientIdempotencyKey
            ? { clientIdempotencyKey }
            : {}),
          ...(appVersion ? { appVersion } : {}),
        }),
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

export async function reserveRelayCredits({
  cfApiBase,
  relaySecret,
  payload,
}: {
  cfApiBase: string;
  relaySecret: string;
  payload: RelayBillingPayload;
}): Promise<RelayMutationResult> {
  return mutateRelayBilling({
    cfApiBase,
    relaySecret,
    endpoint: STAGE5_RELAY_BILLING_ENDPOINTS.RESERVE,
    payload,
    unavailableMessage: "Credit reservation backend unavailable",
    failureMessage: "Credit reservation failed",
  });
}

export async function confirmRelayReservation({
  cfApiBase,
  relaySecret,
  payload,
}: {
  cfApiBase: string;
  relaySecret: string;
  payload: RelayBillingPayload;
}): Promise<RelayMutationResult> {
  return mutateRelayBilling({
    cfApiBase,
    relaySecret,
    endpoint: STAGE5_RELAY_BILLING_ENDPOINTS.CONFIRM,
    payload,
    unavailableMessage: "Reservation confirmation backend unavailable",
    failureMessage: "Reservation confirmation failed",
  });
}

export async function finalizeRelayCredits({
  cfApiBase,
  relaySecret,
  payload,
}: {
  cfApiBase: string;
  relaySecret: string;
  payload: RelayBillingPayload;
}): Promise<RelayMutationResult> {
  return mutateRelayBilling({
    cfApiBase,
    relaySecret,
    endpoint: STAGE5_RELAY_BILLING_ENDPOINTS.FINALIZE,
    payload,
    unavailableMessage: "Credit finalize backend unavailable",
    failureMessage: "Credit finalize failed",
  });
}

export async function persistRelayReservationMeta({
  cfApiBase,
  relaySecret,
  payload,
}: {
  cfApiBase: string;
  relaySecret: string;
  payload: RelayBillingPayload;
}): Promise<RelayMutationResult> {
  return mutateRelayBilling({
    cfApiBase,
    relaySecret,
    endpoint: STAGE5_RELAY_BILLING_ENDPOINTS.PERSIST,
    payload,
    unavailableMessage: "Replay persistence backend unavailable",
    failureMessage: "Replay persistence failed",
  });
}

export async function releaseRelayCredits({
  cfApiBase,
  relaySecret,
  payload,
}: {
  cfApiBase: string;
  relaySecret: string;
  payload: RelayBillingPayload;
}): Promise<RelayMutationResult> {
  return mutateRelayBilling({
    cfApiBase,
    relaySecret,
    endpoint: STAGE5_RELAY_BILLING_ENDPOINTS.RELEASE,
    payload,
    unavailableMessage: "Credit release backend unavailable",
    failureMessage: "Credit release failed",
  });
}

export async function storeRelayReplayArtifact({
  cfApiBase,
  relaySecret,
  payload,
}: {
  cfApiBase: string;
  relaySecret: string;
  payload: {
    deviceId: string;
    service: RelayBillingService;
    requestKey: string;
    body: unknown;
  };
}): Promise<
  | { ok: true; artifact: RelayReplayArtifactRef }
  | { ok: false; status: number; error: string }
> {
  let response: Response;
  try {
    response = await fetch(
      buildStage5RelayBillingUrl(
        cfApiBase,
        STAGE5_RELAY_BILLING_ENDPOINTS.REPLAY_STORE,
      ),
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Relay-Secret": relaySecret,
        },
        body: JSON.stringify({
          deviceId: payload.deviceId,
          service: payload.service,
          requestKey: payload.requestKey,
          payload: payload.body,
        }),
      }
    );
  } catch (error) {
    return {
      ok: false,
      status: 503,
      error: formatTransportFailureMessage(
        "Replay artifact backend unavailable",
        error,
      ),
    };
  }

  const body = await safeJson(response);
  if (!response.ok) {
    return {
      ok: false,
      status: response.status,
      error: extractErrorMessage(body, "Replay artifact store failed"),
    };
  }

  const artifact = (body as any)?.artifact;
  if (!isReplayArtifactRef(artifact)) {
    return {
      ok: false,
      status: 500,
      error: "Replay artifact store returned an invalid artifact reference",
    };
  }

  return { ok: true, artifact };
}

export async function loadRelayReplayArtifact({
  cfApiBase,
  relaySecret,
  artifact,
}: {
  cfApiBase: string;
  relaySecret: string;
  artifact: RelayReplayArtifactRef;
}): Promise<
  | { ok: true; body: unknown }
  | { ok: false; status: number; error: string }
> {
  let response: Response;
  try {
    response = await fetch(
      buildStage5RelayBillingUrl(
        cfApiBase,
        STAGE5_RELAY_BILLING_ENDPOINTS.REPLAY_LOAD,
      ),
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Relay-Secret": relaySecret,
        },
        body: JSON.stringify({ artifact }),
      }
    );
  } catch (error) {
    return {
      ok: false,
      status: 503,
      error: formatTransportFailureMessage(
        "Replay artifact backend unavailable",
        error,
      ),
    };
  }

  const body = await safeJson(response);
  if (!response.ok) {
    return {
      ok: false,
      status: response.status,
      error: extractErrorMessage(body, "Replay artifact load failed"),
    };
  }

  return {
    ok: true,
    body: (body as any)?.payload,
  };
}

export async function deleteRelayReplayArtifact({
  cfApiBase,
  relaySecret,
  artifact,
}: {
  cfApiBase: string;
  relaySecret: string;
  artifact: RelayReplayArtifactRef;
}): Promise<
  | { ok: true }
  | { ok: false; status: number; error: string }
> {
  let response: Response;
  try {
    response = await fetch(
      buildStage5RelayBillingUrl(
        cfApiBase,
        STAGE5_RELAY_BILLING_ENDPOINTS.REPLAY_DELETE,
      ),
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Relay-Secret": relaySecret,
        },
        body: JSON.stringify({ artifact }),
      }
    );
  } catch (error) {
    return {
      ok: false,
      status: 503,
      error: formatTransportFailureMessage(
        "Replay artifact backend unavailable",
        error,
      ),
    };
  }

  const body = await safeJson(response);
  if (!response.ok) {
    return {
      ok: false,
      status: response.status,
      error: extractErrorMessage(body, "Replay artifact delete failed"),
    };
  }

  return { ok: true };
}

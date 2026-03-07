import { randomUUID } from "node:crypto";
import {
  persistRelayReservationMeta,
  releaseRelayCredits,
} from "./relay-billing-client.js";
import { STAGE5_RELAY_BILLING_SERVICES } from "./relay-billing-contract.js";

type RelayBillingService =
  (typeof STAGE5_RELAY_BILLING_SERVICES)[keyof typeof STAGE5_RELAY_BILLING_SERVICES];

const DIRECT_REQUEST_LEASE_TIMEOUT_MS = Math.max(
  60_000,
  Number.parseInt(
    process.env.DIRECT_REQUEST_LEASE_TIMEOUT_MS || String(10 * 60 * 1_000),
    10,
  ),
);
const DIRECT_REQUEST_LEASE_HEARTBEAT_MS = Math.max(
  10_000,
  Math.min(
    DIRECT_REQUEST_LEASE_TIMEOUT_MS / 3,
    Number.parseInt(
      process.env.DIRECT_REQUEST_LEASE_HEARTBEAT_MS || String(60_000),
      10,
    ),
  ),
);
let relayInstanceId: string | null = null;

type DirectRequestLease = {
  version: 1;
  instanceId: string;
  ownerId: string;
  acquiredAt: string;
  lastHeartbeatAt: string;
  timeoutMs: number;
};

function asObject(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

function parseTimestampMs(value: unknown): number | null {
  if (typeof value !== "string" || !value.trim()) {
    return null;
  }
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function extractDirectRequestLease(
  reservationMeta: unknown,
): DirectRequestLease | null {
  const metaObject = asObject(reservationMeta);
  const leaseObject = asObject(metaObject?.directRequestLease);
  if (!leaseObject) {
    return null;
  }

  const version = Number(leaseObject.version);
  const instanceId =
    typeof leaseObject.instanceId === "string" ? leaseObject.instanceId.trim() : "";
  const ownerId =
    typeof leaseObject.ownerId === "string" ? leaseObject.ownerId.trim() : "";
  const acquiredAt =
    typeof leaseObject.acquiredAt === "string" ? leaseObject.acquiredAt.trim() : "";
  const lastHeartbeatAt =
    typeof leaseObject.lastHeartbeatAt === "string"
      ? leaseObject.lastHeartbeatAt.trim()
      : "";
  const timeoutMs = Number(leaseObject.timeoutMs);

  if (
    version !== 1 ||
    !instanceId ||
    !ownerId ||
    !acquiredAt ||
    !lastHeartbeatAt ||
    !Number.isFinite(timeoutMs) ||
    timeoutMs <= 0
  ) {
    return null;
  }

  return {
    version: 1,
    instanceId,
    ownerId,
    acquiredAt,
    lastHeartbeatAt,
    timeoutMs,
  };
}

function getLeaseHeartbeatAgeMs(lease: DirectRequestLease, now = Date.now()): number {
  const heartbeatMs = parseTimestampMs(lease.lastHeartbeatAt);
  if (heartbeatMs === null) {
    return Number.POSITIVE_INFINITY;
  }
  return now - heartbeatMs;
}

function isDirectRequestLeaseFresh({
  reservationMeta,
  reservationUpdatedAt,
  now = Date.now(),
}: {
  reservationMeta?: unknown;
  reservationUpdatedAt?: string;
  now?: number;
}): boolean {
  const lease = extractDirectRequestLease(reservationMeta);
  if (lease) {
    return getLeaseHeartbeatAgeMs(lease, now) <= lease.timeoutMs;
  }

  const updatedAtMs = parseTimestampMs(reservationUpdatedAt);
  if (updatedAtMs === null) {
    return false;
  }
  return now - updatedAtMs <= DIRECT_REQUEST_LEASE_TIMEOUT_MS;
}

export function normalizeRelayRecoveryFailureStatus(status: number): number {
  if (status === 402 || status === 409) {
    return status;
  }
  return 503;
}

export function createDirectRequestLease(): DirectRequestLease {
  const now = new Date().toISOString();
  if (!relayInstanceId) {
    relayInstanceId = `relay:${process.pid}:${randomUUID()}`;
  }
  return {
    version: 1,
    instanceId: relayInstanceId,
    ownerId: randomUUID(),
    acquiredAt: now,
    lastHeartbeatAt: now,
    timeoutMs: DIRECT_REQUEST_LEASE_TIMEOUT_MS,
  };
}

export function startDirectRequestLeaseHeartbeat({
  cfApiBase,
  relaySecret,
  deviceId,
  requestKey,
  service,
  lease,
}: {
  cfApiBase: string;
  relaySecret: string;
  deviceId: string;
  requestKey: string;
  service: RelayBillingService;
  lease: DirectRequestLease;
}): () => void {
  let active = true;
  let inFlight = false;
  let currentLease = lease;

  const persistHeartbeat = async (): Promise<void> => {
    if (!active || inFlight) {
      return;
    }
    inFlight = true;
    const nextLease: DirectRequestLease = {
      ...currentLease,
      lastHeartbeatAt: new Date().toISOString(),
    };
    try {
      const result = await persistRelayReservationMeta({
        cfApiBase,
        relaySecret,
        payload: {
          deviceId,
          requestKey,
          service,
          meta: {
            directRequestLease: nextLease,
          },
        },
      });
      if (result.ok) {
        currentLease = nextLease;
      }
    } finally {
      inFlight = false;
    }
  };

  const timer = setInterval(() => {
    void persistHeartbeat();
  }, DIRECT_REQUEST_LEASE_HEARTBEAT_MS);
  (timer as NodeJS.Timeout).unref?.();

  return () => {
    active = false;
    clearInterval(timer);
  };
}

export async function recoverOrRestartDuplicateReservation({
  cfApiBase,
  relaySecret,
  deviceId,
  requestKey,
  service,
  reservationMeta,
  reservationUpdatedAt,
}: {
  cfApiBase: string;
  relaySecret: string;
  deviceId: string;
  requestKey: string;
  service: RelayBillingService;
  reservationMeta?: unknown;
  reservationUpdatedAt?: string;
}): Promise<
  | { ok: true; action: "retry-reserve" }
  | { ok: true; action: "reservation-settled"; reservationMeta?: unknown }
  | { ok: false; status: number; error: string; details?: string }
> {
  if (
    isDirectRequestLeaseFresh({
      reservationMeta,
      reservationUpdatedAt,
    })
  ) {
    return {
      ok: false,
      status: 409,
      error: "Duplicate request is still in progress",
    };
  }

  const releaseResult = await releaseRelayCredits({
    cfApiBase,
    relaySecret,
    payload: {
      deviceId,
      requestKey,
      service,
      meta: {
        reason: "duplicate-reserved-without-replay-state",
      },
    },
  });
  if (!releaseResult.ok) {
    return {
      ok: false,
      status: normalizeRelayRecoveryFailureStatus(releaseResult.status),
      error:
        releaseResult.error || "Failed to recover duplicate reservation state",
    };
  }

  if (
    releaseResult.status === "released" ||
    releaseResult.reservationStatus === "released"
  ) {
    return { ok: true, action: "retry-reserve" };
  }

  if (releaseResult.reservationStatus === "settled") {
    return {
      ok: true,
      action: "reservation-settled",
      ...(typeof releaseResult.reservationMeta !== "undefined"
        ? { reservationMeta: releaseResult.reservationMeta }
        : {}),
    };
  }

  return {
    ok: false,
    status: 409,
    error: "Duplicate request is still in progress",
  };
}

export async function persistDirectReplayOrRelease({
  cfApiBase,
  relaySecret,
  deviceId,
  requestKey,
  service,
  replayResult,
  pendingFinalize,
}: {
  cfApiBase: string;
  relaySecret: string;
  deviceId: string;
  requestKey: string;
  service: RelayBillingService;
  replayResult: unknown;
  pendingFinalize: unknown;
}): Promise<
  | { ok: true }
  | { ok: false; status: number; error: string; details?: string }
> {
  const persistResult = await persistRelayReservationMeta({
    cfApiBase,
    relaySecret,
    payload: {
      deviceId,
      requestKey,
      service,
      meta: {
        directReplayResult: replayResult,
        pendingFinalize,
      },
    },
  });
  if (persistResult.ok) {
    return { ok: true };
  }

  let details: string | undefined;
  const releaseResult = await releaseRelayCredits({
    cfApiBase,
    relaySecret,
    payload: {
      deviceId,
      requestKey,
      service,
      meta: {
        reason: "replay-persist-failed",
        persistError: persistResult.error || "Replay persistence failed",
      },
    },
  });
  if (!releaseResult.ok) {
    details =
      releaseResult.error && releaseResult.error.trim()
        ? `Reservation release also failed: ${releaseResult.error.trim()}`
        : "Reservation release also failed";
  }

  return {
    ok: false,
    status: normalizeRelayRecoveryFailureStatus(persistResult.status),
    error: persistResult.error || "Replay persistence failed",
    ...(details ? { details } : {}),
  };
}

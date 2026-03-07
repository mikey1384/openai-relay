import {
  deleteRelayReplayArtifact,
  loadRelayReplayArtifact,
  storeRelayReplayArtifact,
  type RelayReplayArtifactRef,
} from "./relay-billing-client.js";
import { STAGE5_RELAY_BILLING_SERVICES } from "./relay-billing-contract.js";

type RelayBillingService =
  (typeof STAGE5_RELAY_BILLING_SERVICES)[keyof typeof STAGE5_RELAY_BILLING_SERVICES];

export type DirectReplayResult =
  | { kind: "success"; status: number; data: unknown }
  | { kind: "error"; status: number; error: string; details?: string };

export type StoredDirectReplayResult =
  | { kind: "success"; status: number; data: unknown }
  | { kind: "success"; status: number; artifact: RelayReplayArtifactRef }
  | { kind: "error"; status: number; error: string; details?: string };

function asObject(value: unknown): Record<string, unknown> | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null;
  }
  return value as Record<string, unknown>;
}

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

export function extractStoredDirectReplayResult(
  reservationMeta: unknown,
  defaultErrorMessage: string,
): StoredDirectReplayResult | null {
  const metaObject = asObject(reservationMeta);
  const replayObject = asObject(metaObject?.directReplayResult);
  if (!replayObject) {
    return null;
  }

  const kind = replayObject.kind;
  const status = replayObject.status;
  if (
    (kind !== "success" && kind !== "error") ||
    typeof status !== "number" ||
    !Number.isFinite(status)
  ) {
    return null;
  }

  if (kind === "success") {
    const artifact = isReplayArtifactRef(replayObject.artifact)
      ? replayObject.artifact
      : null;
    if (artifact) {
      return {
        kind,
        status,
        artifact,
      };
    }

    if (!Object.prototype.hasOwnProperty.call(replayObject, "data")) {
      return null;
    }

    return {
      kind,
      status,
      data: replayObject.data,
    };
  }

  return {
    kind,
    status,
    error:
      typeof replayObject.error === "string" && replayObject.error.trim()
        ? replayObject.error.trim()
        : defaultErrorMessage,
    ...(typeof replayObject.details === "string" && replayObject.details.trim()
      ? { details: replayObject.details.trim() }
      : {}),
  };
}

export async function materializeStoredDirectReplayResult({
  cfApiBase,
  relaySecret,
  storedReplay,
}: {
  cfApiBase: string;
  relaySecret: string;
  storedReplay: StoredDirectReplayResult | null;
}): Promise<DirectReplayResult | null> {
  if (!storedReplay) {
    return null;
  }

  if (storedReplay.kind === "success" && "artifact" in storedReplay) {
    const loaded = await loadRelayReplayArtifact({
      cfApiBase,
      relaySecret,
      artifact: storedReplay.artifact,
    });
    if (!loaded.ok) {
      return {
        kind: "error",
        status: loaded.status,
        error: loaded.error || "Failed to load replay artifact",
      };
    }

    return {
      kind: "success",
      status: storedReplay.status,
      data: loaded.body,
    };
  }

  return storedReplay;
}

export async function storeSuccessDirectReplayArtifact({
  cfApiBase,
  relaySecret,
  deviceId,
  requestKey,
  service,
  replay,
}: {
  cfApiBase: string;
  relaySecret: string;
  deviceId: string;
  requestKey: string;
  service: RelayBillingService;
  replay: Extract<DirectReplayResult, { kind: "success" }>;
}): Promise<
  | { ok: true; storedReplay: StoredDirectReplayResult }
  | { ok: false; status: number; error: string }
> {
  // Large direct replay bodies (verbose transcripts, base64 audio segments)
  // live in the worker's object store; reservation metadata keeps only a ref.
  const stored = await storeRelayReplayArtifact({
    cfApiBase,
    relaySecret,
    payload: {
      deviceId,
      service,
      requestKey,
      body: replay.data,
    },
  });
  if (!stored.ok) {
    return {
      ok: false,
      status: stored.status,
      error: stored.error,
    };
  }

  return {
    ok: true,
    storedReplay: {
      kind: "success",
      status: replay.status,
      artifact: stored.artifact,
    },
  };
}

export async function deleteStoredDirectReplayArtifact({
  cfApiBase,
  relaySecret,
  storedReplay,
}: {
  cfApiBase: string;
  relaySecret: string;
  storedReplay: StoredDirectReplayResult | null;
}): Promise<void> {
  if (
    !storedReplay ||
    storedReplay.kind !== "success" ||
    !("artifact" in storedReplay)
  ) {
    return;
  }

  await deleteRelayReplayArtifact({
    cfApiBase,
    relaySecret,
    artifact: storedReplay.artifact,
  });
}

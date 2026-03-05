export type RelayJobStatus = "queued" | "processing" | "completed" | "failed";

export type DurableRelayJob = {
  jobId: string;
  status: RelayJobStatus;
  result?: any;
  error?: string | null;
  createdAt?: string;
  updatedAt?: string;
};

export type MemoryRelayJobLike = {
  status: RelayJobStatus;
  updatedAt: number;
};

export type RelayPollJobSource = "memory" | "durable";

export async function upsertDurableRelayTranslationJob(
  cfApiBase: string,
  relaySecret: string,
  jobId: string,
  status: RelayJobStatus,
  options?: {
    result?: any;
    error?: string | null;
  }
): Promise<void> {
  const response = await fetch(`${cfApiBase}/auth/relay/translation-jobs/upsert`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Relay-Secret": relaySecret,
    },
    body: JSON.stringify({
      jobId,
      status,
      ...(options?.result !== undefined ? { result: options.result } : {}),
      ...(options?.error !== undefined ? { error: options.error } : {}),
    }),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(
      `Failed to upsert durable relay translation job (${response.status}): ${text || response.statusText}`
    );
  }
}

export async function getDurableRelayTranslationJob(
  cfApiBase: string,
  relaySecret: string,
  jobId: string
): Promise<DurableRelayJob | null> {
  const response = await fetch(
    `${cfApiBase}/auth/relay/translation-jobs/${encodeURIComponent(jobId)}`,
    {
      method: "GET",
      headers: {
        "X-Relay-Secret": relaySecret,
      },
    }
  );

  if (response.status === 404) return null;

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(
      `Failed to read durable relay translation job (${response.status}): ${text || response.statusText}`
    );
  }

  const payload = (await response.json().catch(() => null)) as
    | {
        jobId?: string;
        status?: RelayJobStatus;
        result?: any;
        error?: string | null;
        createdAt?: string;
        updatedAt?: string;
      }
    | null;

  if (!payload?.jobId || !payload?.status) return null;
  return {
    jobId: payload.jobId,
    status: payload.status,
    result: payload.result,
    error: payload.error ?? null,
    createdAt: payload.createdAt,
    updatedAt: payload.updatedAt,
  };
}

function parseTimestampMs(value: string | undefined): number | null {
  if (!value) return null;
  const normalized = value.trim();
  if (!normalized) return null;

  const d1UtcMatch = normalized.match(
    /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,6}))?$/
  );
  if (d1UtcMatch) {
    const [, year, month, day, hour, minute, second, fractionRaw = ""] =
      d1UtcMatch;
    const millisecond = Number((fractionRaw + "000").slice(0, 3));
    const utcMs = Date.UTC(
      Number(year),
      Number(month) - 1,
      Number(day),
      Number(hour),
      Number(minute),
      Number(second),
      millisecond
    );
    return Number.isFinite(utcMs) ? utcMs : null;
  }

  const parsed = Date.parse(normalized);
  return Number.isFinite(parsed) ? parsed : null;
}

function isTerminalRelayJobStatus(status: RelayJobStatus): boolean {
  return status === "completed" || status === "failed";
}

function getDurableJobUpdatedMs(job: DurableRelayJob): number | null {
  return parseTimestampMs(job.updatedAt) ?? parseTimestampMs(job.createdAt);
}

export function resolveRelayPollJob<T extends MemoryRelayJobLike>(
  jobId: string,
  durableJob: DurableRelayJob | null,
  memoryJobs: Map<string, T>
): {
  source: RelayPollJobSource;
  job: T | DurableRelayJob;
  updatedMs: number | null;
} | null {
  const memoryJob = memoryJobs.get(jobId) ?? null;
  if (!memoryJob && !durableJob) return null;
  if (memoryJob && !durableJob) {
    return {
      source: "memory",
      job: memoryJob,
      updatedMs: memoryJob.updatedAt,
    };
  }
  if (!memoryJob && durableJob) {
    return {
      source: "durable",
      job: durableJob,
      updatedMs: getDurableJobUpdatedMs(durableJob),
    };
  }

  const durable = durableJob as DurableRelayJob;
  const memory = memoryJob as T;
  const memoryTerminal = isTerminalRelayJobStatus(memory.status);
  const durableTerminal = isTerminalRelayJobStatus(durable.status);
  const durableUpdatedMs = getDurableJobUpdatedMs(durable);

  if (memoryTerminal && !durableTerminal) {
    return { source: "memory", job: memory, updatedMs: memory.updatedAt };
  }
  if (durableTerminal && !memoryTerminal) {
    return { source: "durable", job: durable, updatedMs: durableUpdatedMs };
  }

  if (durableUpdatedMs != null && durableUpdatedMs > memory.updatedAt) {
    return { source: "durable", job: durable, updatedMs: durableUpdatedMs };
  }
  return { source: "memory", job: memory, updatedMs: memory.updatedAt };
}

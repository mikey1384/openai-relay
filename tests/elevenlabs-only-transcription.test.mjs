// Managed transcription is ElevenLabs Scribe only (OpenAI whisper-1 and
// gpt-4o-transcribe shut down 2027-02-26; their replacement has no timestamps).
// These tests drive the real transcription route handlers over a local HTTP
// server (real multipart parsing and ffprobe duration probe) with mocked
// Stage5 billing and ElevenLabs HTTP calls. Nothing reaches a real vendor.
import test, { after, afterEach, before } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

let tsxApi;
try {
  tsxApi = await import('tsx/esm/api');
} catch {
  tsxApi = await import(
    new URL('../../translator/node_modules/tsx/dist/esm/api/index.mjs', import.meta.url).href
  );
}
tsxApi.register();

const { handleTranscriptionRoutes } = await import('../relay/relay-routes-transcription.ts');
const { transcribeWithScribe } = await import('../elevenlabs-config.ts');
const {
  ELEVENLABS_KEY_REQUIRED_MESSAGE,
  TRANSCRIPTION_PROVIDER_UNAVAILABLE_MESSAGE,
  toTranscriptionResponse,
  transcribeWithScribeRetrying,
} = await import('../relay/scribe-transcription.ts');

const CF = 'http://cf.test';
const SCRIBE_ATTEMPTS = 3;
const originalFetch = globalThis.fetch;
const originalElevenKey = process.env.ELEVENLABS_API_KEY;
const originalOpenAiKey = process.env.OPENAI_API_KEY;

let server;
let baseUrl;
let fixtureDir;
let audioBytes;

// 1 s of 16 kHz mono 16-bit silence: ffprobe reports exactly 1.0 s.
function silentWav(seconds = 1, sampleRate = 16_000) {
  const dataBytes = seconds * sampleRate * 2;
  const buf = Buffer.alloc(44 + dataBytes);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + dataBytes, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20);
  buf.writeUInt16LE(1, 22);
  buf.writeUInt32LE(sampleRate, 24);
  buf.writeUInt32LE(sampleRate * 2, 28);
  buf.writeUInt16LE(2, 32);
  buf.writeUInt16LE(16, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(dataBytes, 40);
  return buf;
}

function makeCtx() {
  return {
    CF_API_BASE: CF,
    RELAY_SECRET: 'relay-secret',
    ELEVENLABS_TRANSCRIPTION_MODEL: 'elevenlabs-scribe',
    MAX_BODY_SIZE: 1024 * 1024,
    getHeader: (req, name) => {
      const value = req.headers[name.toLowerCase()];
      return Array.isArray(value) ? value[0] : value || undefined;
    },
    sendJson: (res, data, status = 200) => {
      res.writeHead(status, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(data));
    },
    sendError: (res, status, error, details) => {
      res.writeHead(status, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(details ? { error, details } : { error }));
    },
    validateRelaySecret: (req, secret) => req.headers['x-relay-secret'] === secret,
    // The real Scribe client + real retry policy, with no real waiting.
    transcribeWithScribeWithRetries: ({ contextLabel, signal, ...params }) =>
      transcribeWithScribeRetrying({
        transcribe: () => transcribeWithScribe({ ...params, signal }),
        maxAttempts: SCRIBE_ATTEMPTS,
        baseDelayMs: 1,
        maxDelayMs: 1,
        sleep: async () => {},
        contextLabel,
        signal,
      }),
    toTranscriptionResponse,
    makeOpenAI: () => {
      throw new Error('OpenAI must not be used for transcription');
    },
  };
}

// Route handlers log heavily; keep that out of the test runner's stdout.
const originalConsole = { log: console.log, warn: console.warn, error: console.error };

before(async () => {
  console.log = () => {};
  console.warn = () => {};
  console.error = () => {};
  fixtureDir = mkdtempSync(join(os.tmpdir(), 'relay-scribe-only-'));
  audioBytes = silentWav();
  writeFileSync(join(fixtureDir, 'audio.wav'), audioBytes);
  const ctx = makeCtx();
  server = createServer((req, res) => {
    handleTranscriptionRoutes(req, res, ctx).then((handled) => {
      if (!handled) {
        res.writeHead(404);
        res.end();
      }
    });
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  baseUrl = `http://127.0.0.1:${server.address().port}`;
});

after(async () => {
  Object.assign(console, originalConsole);
  await new Promise((resolve) => server.close(resolve));
  rmSync(fixtureDir, { recursive: true, force: true });
});

afterEach(() => {
  globalThis.fetch = originalFetch;
  for (const [name, value] of [
    ['ELEVENLABS_API_KEY', originalElevenKey],
    ['OPENAI_API_KEY', originalOpenAiKey],
  ]) {
    if (value === undefined) delete process.env[name];
    else process.env[name] = value;
  }
});

const scribeOk = () =>
  Response.json({
    text: 'Hello there.',
    language_code: 'en',
    words: [
      { text: 'Hello', start: 0, end: 0.4, type: 'word', speaker_id: 's0' },
      { text: ' ', start: 0.4, end: 0.5, type: 'spacing', speaker_id: 's0' },
      { text: 'there.', start: 0.5, end: 0.9, type: 'word', speaker_id: 's0' },
    ],
  });

// Mocks every outbound call the handlers make. Requests to the local test
// server are not routed through here (tests post with originalFetch).
function installFetchMock({ scribe = scribeOk, reserve } = {}) {
  const calls = { billing: [], eleven: [], other: [] };
  globalThis.fetch = async (url, init = {}) => {
    const href = String(url);
    if (href.startsWith('https://api.elevenlabs.io/')) {
      const form = init.body instanceof FormData ? init.body : null;
      calls.eleven.push({ url: href, headers: init.headers, modelId: form?.get('model_id') });
      return scribe(calls.eleven.length);
    }
    if (href.startsWith(CF)) {
      const endpoint = href.slice(CF.length);
      const body = init.body ? JSON.parse(init.body) : undefined;
      calls.billing.push({ endpoint, body });
      switch (endpoint) {
        case '/auth/authorize':
          return Response.json({ deviceId: 'device-1', creditBalance: 1_000_000 });
        case '/auth/reserve':
          return reserve ? reserve(body) : Response.json({ status: 'reserved' });
        case '/auth/replay-store':
          return Response.json({
            artifact: { version: 1, storage: 'r2', contentType: 'application/json', key: 'replay-key', sizeBytes: 10 },
          });
        default:
          return Response.json({ status: 'ok' });
      }
    }
    calls.other.push({ url: href });
    return new Response('unexpected', { status: 599 });
  };
  return calls;
}

async function post(path, fields, headers) {
  const form = new FormData();
  form.append('file', new Blob([audioBytes], { type: 'audio/wav' }), 'audio.wav');
  for (const [key, value] of Object.entries(fields)) form.append(key, value);
  const response = await originalFetch(`${baseUrl}${path}`, {
    method: 'POST',
    headers,
    body: form,
  });
  return { status: response.status, body: await response.json() };
}

// Exactly what Translator <= 1.22.0 sends after a confirmed Whisper fallback.
const LEGACY_WHISPER_FIELDS = {
  model: 'whisper-1',
  model_id: 'whisper-1',
  qualityMode: 'false',
  language: 'en',
  prompt: 'speaker names',
};

function assertTranscriptShape(body) {
  assert.equal(body.text, 'Hello there.');
  assert.equal(body.model, 'elevenlabs-scribe');
  assert.ok(body.duration > 0);
  assert.equal(body.segments.length, 1);
  const [segment] = body.segments;
  assert.equal(segment.start, 0);
  assert.equal(segment.end, 0.9);
  assert.equal(segment.text, 'Hello there.');
  assert.deepEqual(
    segment.words.map((w) => w.word),
    ['Hello', 'there.'],
  );
  assert.equal(body.words.length, 2);
  assert.equal('fallback' in body, false);
}

test('/transcribe-direct: old client whisper-1 + qualityMode=false is served by Scribe and reserved/billed as Scribe', async () => {
  process.env.ELEVENLABS_API_KEY = 'eleven-relay-key';
  process.env.OPENAI_API_KEY = 'sk-must-not-be-used';
  const calls = installFetchMock();

  const { status, body } = await post('/transcribe-direct', LEGACY_WHISPER_FIELDS, {
    authorization: 'Bearer app-key',
    'idempotency-key': 'tx-direct-legacy-1',
  });
  assert.equal(status, 200, JSON.stringify(body));
  assertTranscriptShape(body);

  assert.equal(calls.eleven.length, 1);
  assert.match(calls.eleven[0].url, /\/v1\/speech-to-text$/);
  assert.equal(calls.eleven[0].modelId, 'scribe_v2');
  assert.equal(calls.eleven[0].headers['xi-api-key'], 'eleven-relay-key');
  assert.equal(calls.other.length, 0);

  const reserve = calls.billing.find((c) => c.endpoint === '/auth/reserve');
  const finalize = calls.billing.find((c) => c.endpoint === '/auth/finalize');
  assert.equal(reserve.body.model, 'elevenlabs-scribe');
  assert.equal(reserve.body.seconds, 3); // 1 s probed + 2 s padding
  assert.equal(finalize.body.model, 'elevenlabs-scribe');
  assert.ok(!calls.billing.some((c) => c.endpoint === '/auth/release'));
});

test('/transcribe-direct: transient Scribe failures are retried, then 502 transcription-provider-unavailable with the hold released and no OpenAI call', async () => {
  process.env.ELEVENLABS_API_KEY = 'eleven-relay-key';
  process.env.OPENAI_API_KEY = 'sk-must-not-be-used';
  const calls = installFetchMock({
    scribe: () => new Response('{"detail":"overloaded"}', { status: 503 }),
  });

  const { status, body } = await post('/transcribe-direct', LEGACY_WHISPER_FIELDS, {
    authorization: 'Bearer app-key',
    'idempotency-key': 'tx-direct-down-1',
  });
  assert.equal(status, 502);
  assert.deepEqual(body, {
    error: 'transcription-provider-unavailable',
    details: TRANSCRIPTION_PROVIDER_UNAVAILABLE_MESSAGE,
  });
  assert.equal(calls.eleven.length, SCRIBE_ATTEMPTS);
  assert.equal(calls.other.length, 0);
  const release = calls.billing.find((c) => c.endpoint === '/auth/release');
  assert.ok(release, 'reservation must be released');
  assert.equal(release.body.meta.reason, 'vendor-error');
  assert.ok(!calls.billing.some((c) => c.endpoint === '/auth/finalize'));
});

test('/transcribe-direct: a permanent Scribe 4xx is not retried and still releases the hold', async () => {
  process.env.ELEVENLABS_API_KEY = 'eleven-relay-key';
  const calls = installFetchMock({
    scribe: () => new Response('{"detail":"invalid audio"}', { status: 400 }),
  });

  const { status, body } = await post('/transcribe-direct', { language: 'en' }, {
    authorization: 'Bearer app-key',
    'idempotency-key': 'tx-direct-bad-audio-1',
  });
  assert.equal(status, 500);
  assert.equal(body.error, 'Transcription failed');
  assert.equal(calls.eleven.length, 1);
  assert.ok(calls.billing.some((c) => c.endpoint === '/auth/release'));
  assert.ok(!calls.billing.some((c) => c.endpoint === '/auth/finalize'));
});

test('/transcribe-direct: insufficient credits answer the normal 402 with no fallback confirmation and no vendor call', async () => {
  process.env.ELEVENLABS_API_KEY = 'eleven-relay-key';
  const calls = installFetchMock({
    reserve: () => Response.json({ error: 'Insufficient credits' }, { status: 402 }),
  });

  const { status, body } = await post('/transcribe-direct', LEGACY_WHISPER_FIELDS, {
    authorization: 'Bearer app-key',
    'idempotency-key': 'tx-direct-poor-1',
  });
  assert.equal(status, 402);
  assert.deepEqual(body, { error: 'Insufficient credits' });
  assert.equal(calls.eleven.length, 0);
  assert.equal(calls.other.length, 0);
});

test('/transcribe-direct: with only an OpenAI key configured it refuses (502) before holding credits', async () => {
  delete process.env.ELEVENLABS_API_KEY;
  process.env.OPENAI_API_KEY = 'sk-must-not-be-used';
  const calls = installFetchMock();

  const { status, body } = await post('/transcribe-direct', LEGACY_WHISPER_FIELDS, {
    authorization: 'Bearer app-key',
    'idempotency-key': 'tx-direct-nokey-1',
  });
  assert.equal(status, 502);
  assert.equal(body.error, 'transcription-provider-unavailable');
  assert.ok(!calls.billing.some((c) => c.endpoint === '/auth/reserve'));
  assert.equal(calls.eleven.length + calls.other.length, 0);
});

test('/transcribe: a caller with only an OpenAI key gets 400 elevenlabs-key-required', async () => {
  delete process.env.ELEVENLABS_API_KEY;
  const calls = installFetchMock();

  const { status, body } = await post('/transcribe', LEGACY_WHISPER_FIELDS, {
    'x-relay-secret': 'relay-secret',
    'x-openai-key': 'sk-openai-only',
    'x-stage5-device-id': 'device-1',
    'x-stage5-request-key': 'tx:req:openai-only',
  });
  assert.equal(status, 400);
  assert.deepEqual(body, {
    error: 'elevenlabs-key-required',
    details: ELEVENLABS_KEY_REQUIRED_MESSAGE,
  });
  assert.equal(
    ELEVENLABS_KEY_REQUIRED_MESSAGE,
    'Transcription uses ElevenLabs. Add an ElevenLabs API key in Settings, or turn off API key mode to use Stage5 credits.',
  );
  assert.equal(calls.billing.length + calls.eleven.length + calls.other.length, 0);
});

test('/transcribe (stage5-api path): legacy whisper-1 + qualityMode=false runs on Scribe and confirms the hold as Scribe', async () => {
  delete process.env.ELEVENLABS_API_KEY;
  const calls = installFetchMock();

  const { status, body } = await post('/transcribe', LEGACY_WHISPER_FIELDS, {
    'x-relay-secret': 'relay-secret',
    'x-openai-key': 'sk-old-api-also-sends-this',
    'x-elevenlabs-key': 'eleven-from-api',
    'x-stage5-device-id': 'device-1',
    'x-stage5-request-key': 'tx:req:legacy',
  });
  assert.equal(status, 200, JSON.stringify(body));
  assertTranscriptShape(body);
  assert.equal(calls.eleven.length, 1);
  assert.equal(calls.eleven[0].headers['xi-api-key'], 'eleven-from-api');
  assert.equal(calls.other.length, 0);
  const topUp = calls.billing.find((c) => c.endpoint === '/auth/confirm' && c.body.seconds);
  assert.equal(topUp.body.model, 'elevenlabs-scribe');
  assert.equal(topUp.body.seconds, 3);
});

test('/transcribe: Scribe outage after retries answers 502 transcription-provider-unavailable (stage5-api releases its hold)', async () => {
  process.env.ELEVENLABS_API_KEY = 'eleven-relay-key';
  const calls = installFetchMock({
    scribe: (n) => (n === 1 ? new Response('rate limited', { status: 429 }) : new Response('down', { status: 500 })),
  });

  const { status, body } = await post('/transcribe', { qualityMode: 'true' }, {
    'x-relay-secret': 'relay-secret',
    'x-stage5-device-id': 'device-1',
    'x-stage5-request-key': 'tx:req:down',
  });
  assert.equal(status, 502);
  assert.equal(body.error, 'transcription-provider-unavailable');
  assert.equal(calls.eleven.length, SCRIBE_ATTEMPTS);
  assert.equal(calls.other.length, 0);
});

test('a Scribe success on a later attempt returns the transcript with retry info and no fallback', async () => {
  process.env.ELEVENLABS_API_KEY = 'eleven-relay-key';
  const calls = installFetchMock({
    scribe: (n) => (n === 1 ? new Response('gateway', { status: 502 }) : scribeOk()),
  });

  const { status, body } = await post('/transcribe-direct', { language: 'en' }, {
    authorization: 'Bearer app-key',
    'idempotency-key': 'tx-direct-retry-1',
  });
  assert.equal(status, 200, JSON.stringify(body));
  assertTranscriptShape(body);
  assert.deepEqual(body.retry, { provider: 'elevenlabs-scribe', attempts: 2 });
  assert.equal(calls.eleven.length, 2);
});

test('no relay source calls OpenAI transcription or keeps a Whisper model', () => {
  const root = fileURLToPath(new URL('..', import.meta.url));
  const offenders = [];
  const walk = (dir) => {
    for (const name of readdirSync(dir)) {
      if (name === 'node_modules' || name === 'dist' || name.startsWith('.')) continue;
      const full = join(dir, name);
      if (statSync(full).isDirectory()) walk(full);
      else if (name.endsWith('.ts')) {
        const source = readFileSync(full, 'utf8');
        if (
          /audio\s*\.\s*transcriptions|audio\/transcriptions|["']whisper-1["']|transcription-fallback-confirmation-required/.test(
            source,
          )
        ) {
          offenders.push(full);
        }
      }
    }
  };
  walk(root);
  assert.deepEqual(offenders, []);
});

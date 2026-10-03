// Managed dubbing is ElevenLabs-only (OpenAI TTS shuts down 2027-01-06).
// These tests drive the real dubbing route handlers with mocked billing and
// ElevenLabs HTTP calls to prove legacy OpenAI-shaped requests are synthesized
// with ElevenLabs v4, use the shared voice map, and are billed as eleven_v4.
import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { readFileSync, readdirSync, statSync } from 'node:fs';
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

const { handleDubbingRoutes } = await import('../relay/relay-routes-dubbing.ts');
const { synthesizeWithElevenLabs } = await import('../elevenlabs-config.ts');
const {
  OPENAI_TO_ELEVENLABS_VOICE,
  ELEVENLABS_VOICE_IDS,
  resolveElevenLabsDubVoice,
  resolveElevenLabsDubFormatName,
} = await import('../elevenlabs-voices.ts');

const CF = 'http://cf.test';
const originalFetch = globalThis.fetch;
const originalElevenKey = process.env.ELEVENLABS_API_KEY;

afterEach(() => {
  globalThis.fetch = originalFetch;
  if (originalElevenKey === undefined) delete process.env.ELEVENLABS_API_KEY;
  else process.env.ELEVENLABS_API_KEY = originalElevenKey;
});

function installFetchMock() {
  const calls = { billing: [], eleven: [], other: [] };
  globalThis.fetch = async (url, init = {}) => {
    const href = String(url);
    const body = init.body ? JSON.parse(init.body) : undefined;
    if (href.startsWith('https://api.elevenlabs.io/')) {
      calls.eleven.push({ url: href, body, headers: init.headers });
      return new Response(new Uint8Array([0xff, 0xfb, 1, 2]));
    }
    if (href.startsWith(CF)) {
      const endpoint = href.slice(CF.length);
      calls.billing.push({ endpoint, body });
      switch (endpoint) {
        case '/auth/authorize':
          return Response.json({ deviceId: 'device-1', creditBalance: 1_000_000 });
        case '/auth/reserve':
          return Response.json({ status: 'reserved' });
        case '/auth/replay-store':
          return Response.json({
            artifact: { version: 1, storage: 'r2', contentType: 'application/json', key: 'replay-key', sizeBytes: 10 },
          });
        default:
          return Response.json({ status: 'ok' });
      }
    }
    calls.other.push({ url: href, body });
    return new Response('unexpected', { status: 599 });
  };
  return calls;
}

function makeReq(url, headers) {
  const req = new EventEmitter();
  req.method = 'POST';
  req.url = url;
  req.headers = headers;
  req.aborted = false;
  return req;
}

function makeRes() {
  const res = new EventEmitter();
  res.writableEnded = false;
  res.headersSent = false;
  res.writeHead = () => { res.headersSent = true; };
  res.write = () => true;
  res.end = () => { res.writableEnded = true; };
  return res;
}

function makeCtx(body, sent) {
  return {
    CF_API_BASE: CF,
    RELAY_SECRET: 'relay-secret',
    DUB_MAX_SEGMENTS: 240,
    DUB_MAX_TOTAL_CHARACTERS: 90_000,
    MAX_TTS_CHARS_PER_CHUNK: 3_500,
    getHeader: (req, name) => req.headers[name.toLowerCase()],
    sendJson: (res, data, status = 200) => { sent.push({ status, data }); res.writableEnded = true; },
    sendError: (res, status, error, details) => { sent.push({ status, error, details }); res.writableEnded = true; },
    validateRelaySecret: (req, secret) => req.headers['x-relay-secret'] === secret,
    readJsonBody: async () => body,
    synthesizeWithElevenLabs,
    chunkLines: (lines) => lines.filter(Boolean),
    shouldRetrySegmentError: () => false,
    makeOpenAI: () => { throw new Error('OpenAI must not be used for dubbing'); },
  };
}

test('voice map: OpenAI names map onto Translator ElevenLabs voices that all have voice IDs', () => {
  assert.deepEqual(OPENAI_TO_ELEVENLABS_VOICE, {
    alloy: 'adam', echo: 'brian', fable: 'emily', onyx: 'josh', nova: 'rachel', shimmer: 'sarah',
  });
  const translatorVoices = ['rachel', 'adam', 'josh', 'sarah', 'charlie', 'emily', 'matilda', 'brian'];
  for (const [openaiVoice, target] of Object.entries(OPENAI_TO_ELEVENLABS_VOICE)) {
    assert.ok(translatorVoices.includes(target), `${openaiVoice} -> ${target}`);
    assert.ok(ELEVENLABS_VOICE_IDS[target], `${target} needs a voice id`);
    assert.equal(resolveElevenLabsDubVoice(openaiVoice.toUpperCase()), target);
  }
  assert.equal(resolveElevenLabsDubVoice(undefined), 'adam');
  assert.equal(resolveElevenLabsDubVoice('Rachel'), 'rachel');
  assert.equal(resolveElevenLabsDubVoice('CustomVoiceId123'), 'CustomVoiceId123');
  assert.equal(resolveElevenLabsDubFormatName('aac'), 'mp3');
  assert.equal(resolveElevenLabsDubFormatName('WAV'), 'wav');
});

test('/dub-direct: legacy ttsProvider "openai" + voice "nova" is synthesized by ElevenLabs v4 as rachel and billed as eleven_v4', async () => {
  process.env.ELEVENLABS_API_KEY = 'eleven-test';
  const calls = installFetchMock();
  const sent = [];
  const body = {
    segments: [{ index: 1, text: 'Hello there' }, { index: 2, text: 'General Kenobi' }],
    voice: 'nova',
    model: 'tts-1',
    format: 'mp3',
    ttsProvider: 'openai',
  };
  const req = makeReq('/dub-direct', { authorization: 'Bearer app-key', 'idempotency-key': 'idem-1' });
  const handled = await handleDubbingRoutes(req, makeRes(), makeCtx(body, sent));

  assert.equal(handled, true);
  assert.equal(calls.other.length, 0);
  assert.equal(sent.length, 1);
  assert.equal(sent[0].status, 200, JSON.stringify(sent[0]));
  assert.equal(sent[0].data.model, 'eleven_v4');
  assert.equal(sent[0].data.voice, 'rachel');
  assert.equal(sent[0].data.segments.length, 2);

  assert.equal(calls.eleven.length, 2);
  for (const call of calls.eleven) {
    assert.match(call.url, /\/text-to-dialogue\?/);
    assert.equal(call.body.model_id, 'eleven_v4');
    assert.equal(call.body.inputs[0].voice_id, ELEVENLABS_VOICE_IDS.rachel);
  }

  const reserve = calls.billing.find((c) => c.endpoint === '/auth/reserve');
  const finalize = calls.billing.find((c) => c.endpoint === '/auth/finalize');
  assert.equal(reserve.body.model, 'eleven_v4');
  assert.equal(finalize.body.model, 'eleven_v4');
  assert.equal(reserve.body.characters, 'Hello there'.length + 'General Kenobi'.length);
  assert.equal(finalize.body.characters, reserve.body.characters);
});

test('/dub-direct: ElevenLabs failure releases the reservation and never falls back to OpenAI', async () => {
  process.env.ELEVENLABS_API_KEY = 'eleven-test';
  const calls = installFetchMock();
  const mocked = globalThis.fetch;
  globalThis.fetch = async (url, init) =>
    String(url).startsWith('https://api.elevenlabs.io/')
      ? new Response('boom', { status: 500 })
      : mocked(url, init);
  const sent = [];
  const req = makeReq('/dub-direct', { authorization: 'Bearer app-key', 'idempotency-key': 'idem-2' });
  await handleDubbingRoutes(
    req,
    makeRes(),
    makeCtx({ segments: [{ index: 1, text: 'Hi' }], voice: 'alloy', ttsProvider: 'openai' }, sent),
  );
  assert.equal(sent[0].status, 500);
  assert.ok(calls.billing.some((c) => c.endpoint === '/auth/release'));
  assert.ok(!calls.billing.some((c) => c.endpoint === '/auth/finalize'));
  assert.equal(calls.other.length, 0);
});

test('/dub (legacy stage5-api path): OpenAI-shaped request is served by ElevenLabs with the relay key', async () => {
  process.env.ELEVENLABS_API_KEY = 'eleven-relay-key';
  const calls = installFetchMock();
  const sent = [];
  const req = makeReq('/dub', {
    'x-relay-secret': 'relay-secret',
    'x-openai-key': 'sk-old-api-only-sends-this',
    'x-stage5-device-id': 'device-1',
    'x-stage5-request-key': 'tts:req:1',
  });
  await handleDubbingRoutes(
    req,
    makeRes(),
    makeCtx(
      {
        voice: 'alloy',
        model: 'tts-1',
        format: 'mp3',
        segments: [{ index: 1, text: 'Hello', start: 0, end: 1 }],
        lines: ['Hello'],
      },
      sent,
    ),
  );
  assert.equal(sent[0].status, 200, JSON.stringify(sent[0]));
  assert.equal(sent[0].data.model, 'eleven_v4');
  assert.equal(sent[0].data.voice, 'adam');
  assert.equal(sent[0].data.segments[0].targetDuration, 1);
  assert.equal(calls.eleven.length, 1);
  assert.equal(calls.eleven[0].headers['xi-api-key'], 'eleven-relay-key');
  assert.equal(calls.eleven[0].body.inputs[0].voice_id, ELEVENLABS_VOICE_IDS.adam);
  assert.equal(calls.other.length, 0);
});

test('no relay source calls OpenAI speech synthesis or serves /speech', () => {
  const root = fileURLToPath(new URL('..', import.meta.url));
  const offenders = [];
  const walk = (dir) => {
    for (const name of readdirSync(dir)) {
      if (name === 'node_modules' || name === 'dist' || name.startsWith('.')) continue;
      const full = join(dir, name);
      if (statSync(full).isDirectory()) walk(full);
      else if (name.endsWith('.ts')) {
        const source = readFileSync(full, 'utf8');
        if (/audio\s*\.\s*speech/.test(source) || /["']\/speech["']/.test(source)) offenders.push(full);
      }
    }
  };
  walk(root);
  assert.deepEqual(offenders, []);
});

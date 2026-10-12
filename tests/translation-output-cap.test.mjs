// The translation output cap is sized from the prompt so credit holds track
// real cost (a flat 16,000 held ~40x and refused users who could afford the
// job). The same value caps the provider call, so spend never exceeds the hold.
import test from 'node:test';
import assert from 'node:assert/strict';

let tsxApi;
try {
  tsxApi = await import('tsx/esm/api');
} catch {
  tsxApi = await import(
    new URL('../../translator/node_modules/tsx/dist/esm/api/index.mjs', import.meta.url).href
  );
}
const helpers = await tsxApi.tsImport('../relay/relay-translation-helpers.ts', import.meta.url);
const {
  resolveTranslationReservationMaxCompletionTokens: cap,
  DEFAULT_TRANSLATION_MAX_COMPLETION_TOKENS: DEFAULT,
  EXTENDED_TRANSLATION_MAX_COMPLETION_TOKENS: EXTENDED,
} = helpers;

test('plain translation parts get a cap sized from the prompt', () => {
  assert.equal(cap({ model: 'gpt-5.1', promptTokens: 958 }), 2 * 958 + 1024);
  assert.equal(cap({ model: 'claude-opus-4-8', promptTokens: 4890 }), 2 * 4890 + 1024);
  assert.equal(cap({ model: 'gpt-5.1', promptTokens: 958, reasoning: { effort: 'low' } }), 2 * 958 + 1024);
});

test('the cap covers the largest output ratio seen in production with headroom', () => {
  // Largest observed: output 0.75x prompt; largest absolute output 2,161 tokens.
  for (const prompt of [50, 719, 2079, 4890, 8456]) {
    // Very large prompts are clamped to the previous flat cap, as before.
    const expected = Math.min(DEFAULT, Math.ceil(prompt * 0.75 * 2.6));
    assert(cap({ model: 'gpt-5.1', promptTokens: prompt }) >= expected, String(prompt));
  }
  assert(cap({ model: 'claude-opus-4-8', promptTokens: 4890 }) > 2161 * 2);
});

test('the prompt-sized cap never exceeds the previous flat cap', () => {
  assert.equal(cap({ model: 'gpt-5.1', promptTokens: 100_000 }), DEFAULT);
});

test('reasoning, tools, web search and unknown prompts keep their previous caps', () => {
  assert.equal(cap({ model: 'claude-opus-4-8', promptTokens: 900, reasoning: { effort: 'high' } }), EXTENDED);
  assert.equal(cap({ model: 'gpt-5.1', promptTokens: 900, reasoning: { effort: 'medium' } }), DEFAULT);
  assert.equal(cap({ model: 'gpt-5.1', promptTokens: 900, usesToolsOrWebSearch: true }), DEFAULT);
  assert.equal(cap({ model: 'gpt-5.1' }), DEFAULT);
  assert.equal(cap({ model: 'gpt-5.1', promptTokens: 0 }), DEFAULT);
});

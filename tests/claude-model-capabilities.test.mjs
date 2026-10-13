// Each Claude model gets the request shape it accepts: new models must not
// fall into an older model's thinking or tool_choice settings (those are 400s
// on Opus 5.5 / Sonnet 5.5), and non-answers (refusal, truncation, missing
// required tool call) must surface as errors so the credit hold is released
// rather than billed. Anthropic HTTP is stubbed; nothing reaches the API.
import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';

let tsxApi;
try {
  tsxApi = await import('tsx/esm/api');
} catch {
  tsxApi = await import(
    new URL('../../translator/node_modules/tsx/dist/esm/api/index.mjs', import.meta.url).href
  );
}
const cfg = await tsxApi.tsImport('../anthropic-config.ts', import.meta.url);
const helpers = await tsxApi.tsImport('../relay/relay-translation-helpers.ts', import.meta.url);

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

function stubAnthropic(responseBody) {
  const requests = [];
  globalThis.fetch = async (url, init) => {
    requests.push(JSON.parse(init.body));
    return new Response(
      JSON.stringify({
        id: 'msg_test',
        type: 'message',
        role: 'assistant',
        model: 'stub',
        stop_reason: 'end_turn',
        stop_details: null,
        content: [{ type: 'text', text: 'ok' }],
        usage: { input_tokens: 10, output_tokens: 5 },
        ...responseBody,
      }),
      { status: 200, headers: { 'content-type': 'application/json' } }
    );
  };
  return requests;
}

const tool = {
  type: 'function',
  function: { name: 'submit', parameters: { type: 'object', properties: {} } },
};
const msgs = [
  { role: 'system', content: 'Translate.' },
  { role: 'user', content: 'Hello' },
];

test('existing models keep their request shapes', () => {
  const off = cfg.resolveClaudeThinkingConfig({ model: 'claude-opus-4-8', maxTokens: 3000 });
  assert.equal(off.thinkingOn, false);
  assert.equal(off.apply, undefined);
  assert.equal(off.maxTokens, 3000);

  const params = {};
  const on = cfg.resolveClaudeThinkingConfig({ model: 'claude-opus-4-8', maxTokens: 3000, effort: 'high' });
  on.apply(params);
  assert.deepEqual(params.thinking, { type: 'adaptive' });
  assert.equal(params.output_config.effort, 'high');
  assert(on.maxTokens >= 32000);

  const forced = cfg.resolveClaudeThinkingConfig({ model: 'claude-opus-4-8', maxTokens: 3000, effort: 'high', forceToolUse: true });
  assert.equal(forced.thinkingOn, false);
});

test('Sonnet 5 with effort uses adaptive thinking, never budget_tokens', () => {
  const params = {};
  cfg.resolveClaudeThinkingConfig({ model: 'claude-sonnet-5', maxTokens: 3000, effort: 'medium' }).apply(params);
  assert.deepEqual(params.thinking, { type: 'adaptive' });
});

test('unknown older models keep budget thinking', () => {
  const params = {};
  cfg.resolveClaudeThinkingConfig({ model: 'claude-haiku-4-5', maxTokens: 3000, effort: 'medium' }).apply(params);
  assert.deepEqual(params.thinking, { type: 'enabled', budget_tokens: 8000 });
});

test('Opus 5.5 always thinks, at an explicit low effort by default', () => {
  const params = {};
  const c = cfg.resolveClaudeThinkingConfig({ model: 'claude-opus-5-5', maxTokens: 3000 });
  c.apply(params);
  assert.equal(c.thinkingOn, true);
  assert.deepEqual(params.thinking, { type: 'adaptive' });
  assert.equal(params.output_config.effort, 'low');
  assert(c.maxTokens >= 32000);
  assert.equal(cfg.claudeModelAlwaysThinks('claude-opus-5-5'), true);
  assert.equal(cfg.claudeModelAlwaysThinks('claude-opus-4-8'), false);
});

test('Sonnet 5.5 turns thinking off with between_tools and nothing else', () => {
  const params = {};
  const c = cfg.resolveClaudeThinkingConfig({ model: 'claude-sonnet-5-5', maxTokens: 3000 });
  c.apply(params);
  assert.equal(c.thinkingOn, false);
  assert.deepEqual(params.thinking, { type: 'between_tools' });
  assert.equal(params.output_config, undefined);
  assert.equal(c.maxTokens, 3000);

  const deep = {};
  cfg.resolveClaudeThinkingConfig({ model: 'claude-sonnet-5-5', maxTokens: 3000, effort: 'high' }).apply(deep);
  assert.deepEqual(deep.thinking, { type: 'adaptive' });
  assert.equal(deep.output_config.effort, 'high');
});

test('forced tool use: tool_choice any where allowed, auto plus instruction on new models', async () => {
  let requests = stubAnthropic({ content: [{ type: 'tool_use', id: 't1', name: 'submit', input: {} }], stop_reason: 'tool_use' });
  await cfg.translateWithClaude({ messages: msgs, model: 'claude-opus-4-8', apiKey: 'k', tools: [tool], toolChoice: 'required' });
  assert.deepEqual(requests[0].tool_choice, { type: 'any' });

  requests = stubAnthropic({ content: [{ type: 'tool_use', id: 't1', name: 'submit', input: {} }], stop_reason: 'tool_use' });
  await cfg.translateWithClaude({ messages: msgs, model: 'claude-sonnet-5-5', apiKey: 'k', tools: [tool], toolChoice: 'required' });
  assert.equal(requests[0].tool_choice, undefined);
  assert.match(String(requests[0].system), /Respond by calling one of the provided tools/);
  assert.deepEqual(requests[0].thinking, { type: 'between_tools' });
});

test('a required tool call that never comes is an error, not a billed answer', async () => {
  stubAnthropic({ content: [{ type: 'text', text: 'here you go' }] });
  await assert.rejects(
    cfg.translateWithClaude({ messages: msgs, model: 'claude-sonnet-5-5', apiKey: 'k', tools: [tool], toolChoice: 'required' }),
    err => err.name === 'ClaudeIncompleteResponseError' && err.reason === 'missing_tool_call'
  );
});

test('refusals and truncated output are errors, so holds are released unbilled', async () => {
  stubAnthropic({ stop_reason: 'refusal', stop_details: { type: 'refusal', category: 'cyber' }, content: [] });
  await assert.rejects(
    cfg.translateWithClaude({ messages: msgs, model: 'claude-opus-4-8', apiKey: 'k' }),
    err => err.reason === 'refusal' && /cyber/.test(err.message)
  );
  stubAnthropic({ stop_reason: 'max_tokens' });
  await assert.rejects(
    cfg.translateWithClaude({ messages: msgs, model: 'claude-opus-5-5', apiKey: 'k' }),
    err => err.reason === 'max_tokens'
  );
  stubAnthropic({});
  const ok = await cfg.translateWithClaude({ messages: msgs, model: 'claude-opus-5-5', apiKey: 'k' });
  assert.equal(ok.choices[0].message.content, 'ok');
});

test('the output cap leaves always-thinking models room sized from trial data', () => {
  const { resolveTranslationReservationMaxCompletionTokens: cap, EXTENDED_TRANSLATION_MAX_COMPLETION_TOKENS: EXTENDED } = helpers;
  // Trial peak for Opus 5.5 incl. thinking: 0.38x the prompt, 1,321 tokens.
  assert.equal(cap({ model: 'claude-opus-5-5', promptTokens: 900 }), 2 * 900 + 4096);
  assert(cap({ model: 'claude-opus-5-5', promptTokens: 5019 }) > 5019 * 0.38 * 5);
  assert.equal(cap({ model: 'claude-opus-5-5', promptTokens: 100_000 }), EXTENDED);
  assert.equal(cap({ model: 'claude-opus-5-5', promptTokens: 900, reasoning: { effort: 'high' } }), EXTENDED);
  assert.equal(cap({ model: 'claude-sonnet-5-5', promptTokens: 900 }), 2 * 900 + 1024);
  assert.equal(cap({ model: 'claude-haiku-5-5', promptTokens: 900 }), 2 * 900 + 1024);
});

test('Haiku 5.5 turns thinking off explicitly and keeps forced tool use', async () => {
  const params = {};
  const c = cfg.resolveClaudeThinkingConfig({ model: 'claude-haiku-5-5', maxTokens: 3000 });
  c.apply(params);
  assert.equal(c.thinkingOn, false);
  assert.deepEqual(params.thinking, { type: 'disabled' });
  const requests = stubAnthropic({ content: [{ type: 'tool_use', id: 't1', name: 'submit', input: {} }], stop_reason: 'tool_use' });
  await cfg.translateWithClaude({ messages: msgs, model: 'claude-haiku-5-5', apiKey: 'k', tools: [tool], toolChoice: 'required' });
  assert.deepEqual(requests[0].tool_choice, { type: 'any' });
});

test('credit-mode subtitles use Haiku 5.5 drafts and Opus 5.5 review when Anthropic is available, GPT otherwise', () => {
  const review = { translationPhase: 'review', messages: [{ role: 'user', content: 'x' }] };
  assert.equal(helpers.resolveTranslationModel({ ...review, canUseAnthropic: true, modelFamily: 'gpt' }), 'claude-opus-5-5');
  assert.equal(helpers.resolveTranslationModel({ ...review, canUseAnthropic: true }), 'claude-opus-5-5');
  assert.equal(helpers.resolveTranslationModel({ ...review, canUseAnthropic: false, modelFamily: 'claude' }), 'gpt-5.5');
  assert.equal(helpers.resolveTranslationModel({ translationPhase: 'draft', messages: [], canUseAnthropic: true }), 'claude-haiku-5-5');
  assert.equal(helpers.resolveTranslationModel({ translationPhase: 'draft', messages: [], canUseAnthropic: false }), 'gpt-5.1');
});

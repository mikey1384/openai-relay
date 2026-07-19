// Guards the copied wire-protocol contract against drift. The canonical
// copy lives in stage5-api; this test compares exported name/value pairs
// (formatting- and quote-style-agnostic) whenever the sibling checkout
// exists next to this repo, which is how this workspace is laid out.
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const OWN_COPY = path.resolve(__dirname, '../relay/wire-protocol.ts');
const CANONICAL = path.resolve(
  __dirname,
  '../../stage5-api/src/shared/wire-protocol.ts'
);

function parseWireExports(filePath) {
  const source = fs.readFileSync(filePath, 'utf8');
  const out = {};
  const re =
    /export const (WIRE_\w+)\s*=\s*(?:'([^']*)'|"([^"]*)"|(\d+(?:\.\d+)?))\s*;/gs;
  let m;
  while ((m = re.exec(source))) {
    const [, name, single, double, num] = m;
    out[name] = num !== undefined ? Number(num) : (single ?? double ?? '');
  }
  return out;
}

test('wire-protocol copy matches the canonical stage5-api file', t => {
  if (!fs.existsSync(CANONICAL)) {
    t.skip('stage5-api checkout not present next to this repo');
    return;
  }
  const ours = parseWireExports(OWN_COPY);
  const canonical = parseWireExports(CANONICAL);
  assert.ok(Object.keys(ours).length >= 8, 'parser found too few exports');
  assert.deepEqual(ours, canonical);
});

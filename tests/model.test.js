'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const T = require('../js/tensor.js');
const M = require('../js/model.js');
const { createTokenizer } = require('../js/tokenizer.js');

test('config validation', () => {
  assert.throws(() => new M.GPT({ dModel: 30, nHead: 4 }), /divisible/);
  assert.throws(() => new M.GPT({ nLayer: 0 }));
  assert.doesNotThrow(() => new M.GPT({ dModel: 32, nHead: 4 }));
});

test('parameter breakdown matches actual parameter count', () => {
  const m = new M.GPT({ vocabSize: 50, nLayer: 2, nHead: 2, dModel: 16, blockSize: 8, tieWeights: false });
  const total = m.parameterList().reduce((s, p) => s + p.size, 0);
  assert.equal(m.parameterBreakdown().total, total);
  // hand count: wte 800 + wpe 128 + per layer (ln 64 + qkv 816 + o 272 + mlp 2128) + lnf 32 + head 800
  assert.equal(total, 800 + 128 + 2 * (64 + 16 * 48 + 48 + 16 * 16 + 16 + 16 * 64 + 64 + 64 * 16 + 16) + 32 + 800);
});

test('forward rejects sequences longer than the context window', () => {
  const m = new M.GPT({ vocabSize: 10, blockSize: 4, dModel: 8, nHead: 2 });
  assert.throws(() => m.forward([1, 2, 3, 4, 5]), /context length/);
  assert.equal(m.nextTokenDistribution([1, 2, 3, 4, 5, 6]).context.length, 4);
});

test('causal model: logits at position t do not depend on later tokens', () => {
  const m = new M.GPT({ vocabSize: 10, blockSize: 8, dModel: 16, nHead: 4, nLayer: 2 });
  const a = m.forward([1, 2, 3, 4, 5]).logits.data.slice(0, 20);
  const b = m.forward([1, 2, 7, 7, 7]).logits.data.slice(0, 20);
  assert.deepEqual(Array.from(a), Array.from(b));
  const c = m.forward([1, 2, 7, 7, 7], { causal: false }).logits.data.slice(0, 20);
  assert.notDeepEqual(Array.from(a), Array.from(c));
});

test('sampling knobs', () => {
  const logits = new Float32Array([2, 1, 0, -1]);
  const greedy = M.applySampling(logits, { temperature: 0 });
  assert.deepEqual(Array.from(greedy), [1, 0, 0, 0]);
  const k2 = M.applySampling(logits, { temperature: 1, topK: 2 });
  assert.equal(k2[2] + k2[3], 0);
  assert.ok(Math.abs(k2[0] + k2[1] - 1) < 1e-9);
  const hot = M.applySampling(logits, { temperature: 5 });
  const cold = M.applySampling(logits, { temperature: 0.2 });
  assert.ok(hot[0] < cold[0]);
  const p = M.applySampling(logits, { temperature: 1, topP: 0.5 });
  assert.equal(p.filter((v) => v > 0).length, 1);
});

test('training reduces loss on a repetitive corpus', () => {
  const corpus = 'abcdefgh'.repeat(40);
  const tok = createTokenizer('char', corpus);
  const data = tok.encode(corpus);
  const m = new M.GPT({ vocabSize: tok.vocabSize, nLayer: 1, nHead: 2, dModel: 16, blockSize: 8, seed: 1 });
  const opt = new M.Adam(m.parameterList(), { lr: 1e-2 });
  const rng = T.makeRng(1);
  const before = M.evaluate(m, data);
  for (let i = 0; i < 60; i++) M.trainStep(m, opt, data, { batchSize: 4, rng });
  const after = M.evaluate(m, data);
  assert.ok(after < before * 0.3, `loss ${before} -> ${after}`);
  // greedy generation should continue the pattern
  const gen = M.generate(m, tok.encode('abc'), 5, { temperature: 0 }, rng);
  assert.equal(tok.decode(gen), 'abcdefgh');
});

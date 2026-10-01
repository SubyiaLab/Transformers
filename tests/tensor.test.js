'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const T = require('../js/tensor.js');
const { GPT } = require('../js/model.js');

// Compare analytic gradients with central finite differences.
function gradCheck(buildLoss, params, { eps = 1e-3, tol = 2e-2, samples = 12 } = {}) {
  const rng = T.makeRng(7);
  params.forEach((p) => p.zeroGrad());
  const tape = new T.Tape(true);
  tape.backward(buildLoss(tape));
  for (const p of params) {
    for (let s = 0; s < samples; s++) {
      const i = Math.floor(rng.uniform() * p.size);
      const orig = p.data[i];
      p.data[i] = orig + eps;
      const lp = buildLoss(new T.Tape(false)).data[0];
      p.data[i] = orig - eps;
      const lm = buildLoss(new T.Tape(false)).data[0];
      p.data[i] = orig;
      const num = (lp - lm) / (2 * eps);
      const ana = p.grad[i];
      const denom = Math.max(1e-2, Math.abs(num) + Math.abs(ana));
      assert.ok(Math.abs(num - ana) / denom < tol, `grad mismatch idx ${i}: numeric ${num} vs analytic ${ana}`);
    }
  }
}

test('matmul / bias / gelu / layernorm / cross-entropy gradients', () => {
  const rng = T.makeRng(1);
  const x = T.randn(5, 6, 1, rng);
  const w = T.randn(6, 4, 0.5, rng);
  const b = T.randn(1, 4, 0.5, rng);
  const g = T.randn(1, 4, 1, rng);
  const lb = T.randn(1, 4, 0.5, rng);
  const targets = [0, 3, 1, 2, 3];
  const build = (tape) => {
    let h = T.addBias(tape, T.matmul(tape, x, w), b);
    h = T.gelu(tape, h);
    h = T.layerNorm(tape, h, g, lb);
    return T.crossEntropy(tape, h, targets);
  };
  gradCheck(build, [x, w, b, g, lb]);
});

test('matmulBT and embedding gradients', () => {
  const rng = T.makeRng(2);
  const table = T.randn(7, 4, 1, rng);
  const ids = [1, 3, 3, 6];
  const build = (tape) => T.crossEntropy(tape, T.matmulBT(tape, T.embed(tape, table, ids), table), [0, 2, 3, 6]);
  gradCheck(build, [table]);
});

for (const causal of [true, false]) {
  test(`attention gradients (causal=${causal})`, () => {
    const rng = T.makeRng(3);
    const qkv = T.randn(5, 12, 1, rng); // d=4, 2 heads
    const w = T.randn(4, 3, 1, rng);
    const build = (tape) => {
      const { out } = T.attention(tape, qkv, 2, { causal, scaleMul: 1.5 });
      return T.crossEntropy(tape, T.matmul(tape, out, w), [0, 1, 2, 0, 1]);
    };
    gradCheck(build, [qkv, w]);
  });
}

test('causal attention never looks at the future and rows sum to 1', () => {
  const rng = T.makeRng(4);
  const qkv = T.randn(6, 24, 1, rng);
  const { probs } = T.attention(new T.Tape(false), qkv, 4, { causal: true });
  for (const P of probs) {
    for (let i = 0; i < 6; i++) {
      let s = 0;
      for (let j = 0; j < 6; j++) {
        if (j > i) assert.equal(P[i * 6 + j], 0);
        s += P[i * 6 + j];
      }
      assert.ok(Math.abs(s - 1) < 1e-5);
    }
  }
});

test('full GPT gradient check (every parameter, both pos-encodings, untied head)', () => {
  for (const posEncoding of ['learned', 'sinusoidal']) {
    const model = new GPT({ vocabSize: 9, nLayer: 2, nHead: 2, dModel: 8, blockSize: 6, mlpRatio: 2, posEncoding, tieWeights: posEncoding === 'learned', seed: 5 });
    // bigger init so finite differences are well above float32 noise
    for (const p of model.parameterList()) for (let i = 0; i < p.size; i++) p.data[i] *= p.rows > 1 ? 10 : 1;
    const ids = [1, 4, 2, 8, 0];
    const targets = [4, 2, 8, 0, 3];
    gradCheck((tape) => model.forward(ids, { tape, targets }).loss, model.parameterList(), { samples: 4, tol: 5e-2 });
  }
});

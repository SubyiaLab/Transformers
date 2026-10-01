/*
 * model.js — a tiny GPT (decoder-only transformer), same block layout as GPT-2:
 *
 *   tokens -> token embedding (+ positional encoding)
 *          -> N x [ x + Attn(LN(x)) ; x + MLP(LN(x)) ]   (pre-LayerNorm blocks)
 *          -> final LayerNorm -> LM head (optionally tied to the token embedding)
 *          -> logits over the vocabulary
 *
 * Plus an Adam optimizer and the usual sampling knobs (temperature, top-k, top-p).
 */
(function (root, factory) {
  const T = (typeof require === 'function') ? require('./tensor.js') : root.TV.tensor;
  const mod = factory(T);
  if (typeof module === 'object' && module.exports) module.exports = mod;
  else root.TV = Object.assign(root.TV || {}, { model: mod });
})(typeof self !== 'undefined' ? self : this, function (T) {
  'use strict';

  const DEFAULTS = {
    vocabSize: 64,
    nLayer: 2,
    nHead: 4,
    dModel: 32,
    blockSize: 32,      // context length
    mlpRatio: 4,
    posEncoding: 'learned', // 'learned' | 'sinusoidal' | 'none'
    tieWeights: true,
    seed: 1337,
  };

  function validateConfig(cfg) {
    const errs = [];
    const int = (k, lo, hi) => {
      if (!Number.isInteger(cfg[k]) || cfg[k] < lo || cfg[k] > hi) errs.push(`${k} must be an integer in [${lo}, ${hi}]`);
    };
    int('vocabSize', 2, 4096);
    int('nLayer', 1, 8);
    int('nHead', 1, 16);
    int('dModel', 4, 256);
    int('blockSize', 2, 256);
    int('mlpRatio', 1, 8);
    if (cfg.dModel % cfg.nHead !== 0) errs.push(`d_model (${cfg.dModel}) must be divisible by n_head (${cfg.nHead})`);
    if (!['learned', 'sinusoidal', 'none'].includes(cfg.posEncoding)) errs.push('posEncoding invalid');
    return errs;
  }

  function sinusoidalTable(blockSize, d) {
    const pe = new Float32Array(blockSize * d);
    for (let pos = 0; pos < blockSize; pos++) {
      for (let i = 0; i < d; i += 2) {
        const freq = Math.pow(10000, -i / d);
        pe[pos * d + i] = Math.sin(pos * freq);
        if (i + 1 < d) pe[pos * d + i + 1] = Math.cos(pos * freq);
      }
    }
    return pe;
  }

  class GPT {
    constructor(config) {
      this.config = Object.assign({}, DEFAULTS, config);
      const errs = validateConfig(this.config);
      if (errs.length) throw new Error(errs.join('; '));
      const { vocabSize: V, nLayer, dModel: d, blockSize, mlpRatio, posEncoding, tieWeights, seed } = this.config;
      const rng = T.makeRng(seed);
      const std = 0.02;
      const projStd = 0.02 / Math.sqrt(2 * nLayer); // GPT-2 scaled init for residual projections

      this.params = {};
      const P = (name, t) => { this.params[name] = t; return t; };
      this.wte = P('wte', T.randn(V, d, std, rng));
      if (posEncoding === 'learned') this.wpe = P('wpe', T.randn(blockSize, d, std, rng));
      if (posEncoding === 'sinusoidal') this.peTable = sinusoidalTable(blockSize, d);
      this.blocks = [];
      for (let l = 0; l < nLayer; l++) {
        const h = mlpRatio * d;
        this.blocks.push({
          ln1g: P(`h${l}.ln1.g`, T.filled(1, d, 1)), ln1b: P(`h${l}.ln1.b`, T.filled(1, d, 0)),
          wqkv: P(`h${l}.attn.w_qkv`, T.randn(d, 3 * d, std, rng)), bqkv: P(`h${l}.attn.b_qkv`, T.filled(1, 3 * d, 0)),
          wo: P(`h${l}.attn.w_o`, T.randn(d, d, projStd, rng)), bo: P(`h${l}.attn.b_o`, T.filled(1, d, 0)),
          ln2g: P(`h${l}.ln2.g`, T.filled(1, d, 1)), ln2b: P(`h${l}.ln2.b`, T.filled(1, d, 0)),
          w1: P(`h${l}.mlp.w_fc`, T.randn(d, h, std, rng)), b1: P(`h${l}.mlp.b_fc`, T.filled(1, h, 0)),
          w2: P(`h${l}.mlp.w_proj`, T.randn(h, d, projStd, rng)), b2: P(`h${l}.mlp.b_proj`, T.filled(1, d, 0)),
        });
      }
      this.lnfg = P('ln_f.g', T.filled(1, d, 1));
      this.lnfb = P('ln_f.b', T.filled(1, d, 0));
      if (!tieWeights) this.wlm = P('lm_head', T.randn(d, V, std, rng));
    }

    parameterList() { return Object.values(this.params); }

    // Parameter count grouped by component — shown in the UI's "model card".
    parameterBreakdown() {
      const groups = { 'token embedding': 0, 'position embedding': 0, attention: 0, mlp: 0, layernorm: 0, 'lm head': 0 };
      for (const [name, t] of Object.entries(this.params)) {
        if (name === 'wte') groups['token embedding'] += t.size;
        else if (name === 'wpe') groups['position embedding'] += t.size;
        else if (name.includes('.attn.')) groups.attention += t.size;
        else if (name.includes('.mlp.')) groups.mlp += t.size;
        else if (name.includes('ln')) groups.layernorm += t.size;
        else if (name === 'lm_head') groups['lm head'] += t.size;
      }
      const total = Object.values(groups).reduce((a, b) => a + b, 0);
      return { groups, total };
    }

    // Approximate forward-pass FLOPs for a full context (2 * MACs).
    flopsPerForward(Tlen = this.config.blockSize) {
      const { nLayer, dModel: d, mlpRatio, vocabSize: V } = this.config;
      const perLayer = 2 * Tlen * (d * 3 * d + d * d + 2 * d * mlpRatio * d) + 2 * 2 * Tlen * Tlen * d;
      return nLayer * perLayer + 2 * Tlen * d * V;
    }

    /*
     * ids: token ids (length <= blockSize).
     * opts: { tape, causal, attnScale, targets }
     * Returns { logits, loss?, attn: [layer][head] Float32Array(T*T), residualNorms }
     */
    forward(ids, opts = {}) {
      const tape = opts.tape || new T.Tape(false);
      const Tn = ids.length;
      if (Tn === 0) throw new Error('forward: empty input');
      if (Tn > this.config.blockSize) throw new Error(`forward: ${Tn} tokens exceeds context length ${this.config.blockSize}`);
      const { nHead, dModel: d, posEncoding } = this.config;

      let x = T.embed(tape, this.wte, ids);
      const positions = Array.from({ length: Tn }, (_, i) => i);
      if (posEncoding === 'learned') x = T.add(tape, x, T.embed(tape, this.wpe, positions));
      else if (posEncoding === 'sinusoidal') {
        x = T.add(tape, x, new T.Tensor(Tn, d, this.peTable.slice(0, Tn * d)));
      }

      const attn = [];
      const residualNorms = [rowNormMean(x)];
      for (const b of this.blocks) {
        const h = T.layerNorm(tape, x, b.ln1g, b.ln1b);
        const qkv = T.addBias(tape, T.matmul(tape, h, b.wqkv), b.bqkv);
        const a = T.attention(tape, qkv, nHead, { causal: opts.causal !== false, scaleMul: opts.attnScale ?? 1 });
        attn.push(a.probs);
        x = T.add(tape, x, T.addBias(tape, T.matmul(tape, a.out, b.wo), b.bo));
        const h2 = T.layerNorm(tape, x, b.ln2g, b.ln2b);
        const m = T.gelu(tape, T.addBias(tape, T.matmul(tape, h2, b.w1), b.b1));
        x = T.add(tape, x, T.addBias(tape, T.matmul(tape, m, b.w2), b.b2));
        residualNorms.push(rowNormMean(x));
      }
      const xf = T.layerNorm(tape, x, this.lnfg, this.lnfb);
      const logits = this.config.tieWeights ? T.matmulBT(tape, xf, this.wte) : T.matmul(tape, xf, this.wlm);
      const result = { logits, attn, residualNorms };
      if (opts.targets) result.loss = T.crossEntropy(tape, logits, opts.targets);
      return result;
    }

    // Probability distribution for the token after `ids` (uses the last blockSize tokens).
    nextTokenDistribution(ids, opts = {}) {
      const ctx = ids.slice(-this.config.blockSize);
      const { logits, attn, residualNorms } = this.forward(ctx, opts);
      const V = logits.cols;
      const last = logits.data.slice((logits.rows - 1) * V, logits.rows * V);
      return { logits: last, attn, residualNorms, context: ctx };
    }
  }

  function rowNormMean(x) {
    let total = 0;
    for (let i = 0; i < x.rows; i++) {
      let s = 0;
      for (let j = 0; j < x.cols; j++) { const v = x.data[i * x.cols + j]; s += v * v; }
      total += Math.sqrt(s);
    }
    return total / x.rows;
  }

  // ---- sampling ------------------------------------------------------------
  // Turns raw logits into the distribution actually sampled from after the knobs.
  function applySampling(logits, { temperature = 1, topK = 0, topP = 1 } = {}) {
    const V = logits.length;
    const probs = new Float64Array(V);
    if (temperature <= 0) { // greedy
      let best = 0;
      for (let i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
      probs[best] = 1;
      return probs;
    }
    let mx = -Infinity;
    for (let i = 0; i < V; i++) mx = Math.max(mx, logits[i] / temperature);
    let sum = 0;
    for (let i = 0; i < V; i++) { probs[i] = Math.exp(logits[i] / temperature - mx); sum += probs[i]; }
    for (let i = 0; i < V; i++) probs[i] /= sum;

    const order = Array.from({ length: V }, (_, i) => i).sort((a, b) => probs[b] - probs[a]);
    const keep = new Uint8Array(V);
    let cum = 0;
    for (let r = 0; r < V; r++) {
      const i = order[r];
      if (topK > 0 && r >= topK) break;
      keep[i] = 1;
      cum += probs[i];
      if (topP < 1 && cum >= topP) break;
    }
    sum = 0;
    for (let i = 0; i < V; i++) { if (!keep[i]) probs[i] = 0; sum += probs[i]; }
    for (let i = 0; i < V; i++) probs[i] /= sum;
    return probs;
  }

  function sampleFrom(probs, rng) {
    const u = rng.uniform();
    let cum = 0;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i];
      if (u < cum) return i;
    }
    for (let i = probs.length - 1; i >= 0; i--) if (probs[i] > 0) return i;
    return 0;
  }

  function generate(model, ids, n, sampling, rng, opts = {}) {
    const out = ids.slice();
    for (let s = 0; s < n; s++) {
      const { logits } = model.nextTokenDistribution(out, opts);
      out.push(sampleFrom(applySampling(logits, sampling), rng));
    }
    return out;
  }

  // ---- training --------------------------------------------------------------
  class Adam {
    constructor(params, { lr = 3e-3, beta1 = 0.9, beta2 = 0.99, eps = 1e-8, weightDecay = 0, clip = 1.0 } = {}) {
      this.params = params;
      Object.assign(this, { lr, beta1, beta2, eps, weightDecay, clip });
      this.m = params.map((p) => new Float32Array(p.size));
      this.v = params.map((p) => new Float32Array(p.size));
      this.t = 0;
    }
    zeroGrad() { this.params.forEach((p) => p.zeroGrad()); }
    step() {
      this.t++;
      let sq = 0;
      for (const p of this.params) if (p.grad) for (let i = 0; i < p.size; i++) sq += p.grad[i] * p.grad[i];
      const gradNorm = Math.sqrt(sq);
      const clipScale = this.clip > 0 && gradNorm > this.clip ? this.clip / gradNorm : 1;
      const bc1 = 1 - Math.pow(this.beta1, this.t), bc2 = 1 - Math.pow(this.beta2, this.t);
      this.params.forEach((p, k) => {
        if (!p.grad) return;
        const m = this.m[k], v = this.v[k];
        const decay = p.rows > 1 ? this.weightDecay : 0; // no decay on biases / LN gains
        for (let i = 0; i < p.size; i++) {
          const g = p.grad[i] * clipScale;
          m[i] = this.beta1 * m[i] + (1 - this.beta1) * g;
          v[i] = this.beta2 * v[i] + (1 - this.beta2) * g * g;
          p.data[i] -= this.lr * ((m[i] / bc1) / (Math.sqrt(v[i] / bc2) + this.eps) + decay * p.data[i]);
        }
      });
      return gradNorm;
    }
  }

  // One optimizer step on a batch of random windows from `data` (token ids).
  function trainStep(model, optim, data, { batchSize = 8, causal = true, rng }) {
    const bs = model.config.blockSize;
    const L = Math.min(bs, data.length - 1);
    if (L < 1) throw new Error('Not enough tokens in the corpus to train (need at least 2).');
    optim.zeroGrad();
    let loss = 0;
    for (let b = 0; b < batchSize; b++) {
      const start = Math.floor(rng.uniform() * (data.length - L));
      const x = data.slice(start, start + L);
      const y = data.slice(start + 1, start + L + 1);
      const tape = new T.Tape(true);
      const out = model.forward(x, { tape, targets: y, causal });
      loss += out.loss.data[0];
      tape.backward(out.loss, 1 / batchSize);
    }
    const gradNorm = optim.step();
    return { loss: loss / batchSize, gradNorm };
  }

  // Mean loss over evenly spaced windows (no gradient) — used for a val estimate.
  function evaluate(model, data, { windows = 8, causal = true } = {}) {
    const L = Math.min(model.config.blockSize, data.length - 1);
    if (L < 1) return NaN;
    const span = data.length - L;
    let loss = 0;
    for (let w = 0; w < windows; w++) {
      const start = Math.floor((w / windows) * span);
      const out = model.forward(data.slice(start, start + L), { targets: data.slice(start + 1, start + L + 1), causal });
      loss += out.loss.data[0];
    }
    return loss / windows;
  }

  return { GPT, DEFAULTS, validateConfig, applySampling, sampleFrom, generate, Adam, trainStep, evaluate, sinusoidalTable };
});

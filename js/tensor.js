/*
 * tensor.js — minimal 2-D tensor + reverse-mode autograd.
 *
 * Everything is a row-major [rows x cols] Float32Array. Ops record a backward
 * closure on a Tape when recording is enabled; tape.backward(loss) replays
 * them in reverse. This is just enough to train a tiny GPT in the browser.
 */
(function (root, factory) {
  const mod = factory();
  if (typeof module === 'object' && module.exports) module.exports = mod;
  else root.TV = Object.assign(root.TV || {}, { tensor: mod });
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  class Tensor {
    constructor(rows, cols, data, requiresGrad = false) {
      this.rows = rows;
      this.cols = cols;
      this.data = data || new Float32Array(rows * cols);
      if (this.data.length !== rows * cols) {
        throw new Error(`Tensor size mismatch: ${this.data.length} != ${rows}x${cols}`);
      }
      this.requiresGrad = requiresGrad;
      this.grad = null;
    }
    get size() { return this.rows * this.cols; }
    ensureGrad() {
      if (!this.grad) this.grad = new Float32Array(this.size);
      return this.grad;
    }
    zeroGrad() { if (this.grad) this.grad.fill(0); }
  }

  // ---- deterministic RNG (mulberry32 + Box–Muller) ------------------------
  function makeRng(seed) {
    let a = (seed >>> 0) || 1;
    const uniform = () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    let spare = null;
    const normal = () => {
      if (spare !== null) { const s = spare; spare = null; return s; }
      let u = 0, v = 0;
      while (u === 0) u = uniform();
      v = uniform();
      const r = Math.sqrt(-2 * Math.log(u));
      spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    };
    return { uniform, normal };
  }

  function randn(rows, cols, std, rng) {
    const t = new Tensor(rows, cols, null, true);
    for (let i = 0; i < t.data.length; i++) t.data[i] = rng.normal() * std;
    return t;
  }
  function filled(rows, cols, value, requiresGrad = true) {
    const t = new Tensor(rows, cols, null, requiresGrad);
    t.data.fill(value);
    return t;
  }

  // ---- tape ---------------------------------------------------------------
  class Tape {
    constructor(recording = true) {
      this.recording = recording;
      this.ops = [];
    }
    track(out, inputs, backward) {
      if (!this.recording) return out;
      if (inputs.some((t) => t.requiresGrad)) {
        out.requiresGrad = true;
        this.ops.push(backward);
      }
      return out;
    }
    backward(loss, scale = 1) {
      if (loss.size !== 1) throw new Error('backward() expects a scalar loss');
      loss.ensureGrad()[0] += scale;
      for (let i = this.ops.length - 1; i >= 0; i--) this.ops[i]();
      this.ops = [];
    }
  }

  // ---- ops ----------------------------------------------------------------
  // out[n,m] = a[n,k] @ b[k,m]
  function matmul(tape, a, b) {
    if (a.cols !== b.rows) throw new Error(`matmul shape ${a.rows}x${a.cols} @ ${b.rows}x${b.cols}`);
    const n = a.rows, k = a.cols, m = b.cols;
    const out = new Tensor(n, m);
    const A = a.data, B = b.data, O = out.data;
    for (let i = 0; i < n; i++) {
      const oi = i * m;
      for (let p = 0; p < k; p++) {
        const av = A[i * k + p];
        if (av === 0) continue;
        const bp = p * m;
        for (let j = 0; j < m; j++) O[oi + j] += av * B[bp + j];
      }
    }
    return tape.track(out, [a, b], () => {
      if (!out.grad) return;
      const G = out.grad;
      if (a.requiresGrad) {
        const dA = a.ensureGrad();
        for (let i = 0; i < n; i++) {
          for (let p = 0; p < k; p++) {
            let s = 0;
            const bp = p * m, gi = i * m;
            for (let j = 0; j < m; j++) s += G[gi + j] * B[bp + j];
            dA[i * k + p] += s;
          }
        }
      }
      if (b.requiresGrad) {
        const dB = b.ensureGrad();
        for (let i = 0; i < n; i++) {
          const gi = i * m;
          for (let p = 0; p < k; p++) {
            const av = A[i * k + p];
            if (av === 0) continue;
            const bp = p * m;
            for (let j = 0; j < m; j++) dB[bp + j] += av * G[gi + j];
          }
        }
      }
    });
  }

  // out[n,m] = a[n,k] @ b[m,k]^T   (used for weight-tied LM head)
  function matmulBT(tape, a, b) {
    if (a.cols !== b.cols) throw new Error('matmulBT shape mismatch');
    const n = a.rows, k = a.cols, m = b.rows;
    const out = new Tensor(n, m);
    const A = a.data, B = b.data, O = out.data;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        let s = 0;
        for (let p = 0; p < k; p++) s += A[i * k + p] * B[j * k + p];
        O[i * m + j] = s;
      }
    }
    return tape.track(out, [a, b], () => {
      if (!out.grad) return;
      const G = out.grad;
      const dA = a.requiresGrad ? a.ensureGrad() : null;
      const dB = b.requiresGrad ? b.ensureGrad() : null;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < m; j++) {
          const g = G[i * m + j];
          if (g === 0) continue;
          if (dA) for (let p = 0; p < k; p++) dA[i * k + p] += g * B[j * k + p];
          if (dB) for (let p = 0; p < k; p++) dB[j * k + p] += g * A[i * k + p];
        }
      }
    });
  }

  function add(tape, a, b) {
    if (a.rows !== b.rows || a.cols !== b.cols) throw new Error('add shape mismatch');
    const out = new Tensor(a.rows, a.cols);
    for (let i = 0; i < out.size; i++) out.data[i] = a.data[i] + b.data[i];
    return tape.track(out, [a, b], () => {
      if (!out.grad) return;
      if (a.requiresGrad) { const g = a.ensureGrad(); for (let i = 0; i < out.size; i++) g[i] += out.grad[i]; }
      if (b.requiresGrad) { const g = b.ensureGrad(); for (let i = 0; i < out.size; i++) g[i] += out.grad[i]; }
    });
  }

  // out[n,m] = a[n,m] + bias[1,m] (broadcast over rows)
  function addBias(tape, a, bias) {
    if (bias.rows !== 1 || bias.cols !== a.cols) throw new Error('addBias shape mismatch');
    const n = a.rows, m = a.cols;
    const out = new Tensor(n, m);
    for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) out.data[i * m + j] = a.data[i * m + j] + bias.data[j];
    return tape.track(out, [a, bias], () => {
      if (!out.grad) return;
      if (a.requiresGrad) { const g = a.ensureGrad(); for (let i = 0; i < out.size; i++) g[i] += out.grad[i]; }
      if (bias.requiresGrad) {
        const g = bias.ensureGrad();
        for (let i = 0; i < n; i++) for (let j = 0; j < m; j++) g[j] += out.grad[i * m + j];
      }
    });
  }

  // Gather rows of table[V,d] for ids -> [T,d]
  function embed(tape, table, ids) {
    const T = ids.length, d = table.cols;
    const out = new Tensor(T, d);
    for (let t = 0; t < T; t++) {
      const id = ids[t];
      if (id < 0 || id >= table.rows) throw new Error(`embedding index ${id} out of range`);
      out.data.set(table.data.subarray(id * d, id * d + d), t * d);
    }
    return tape.track(out, [table], () => {
      if (!out.grad) return;
      const g = table.ensureGrad();
      for (let t = 0; t < T; t++) {
        const base = ids[t] * d;
        for (let j = 0; j < d; j++) g[base + j] += out.grad[t * d + j];
      }
    });
  }

  function layerNorm(tape, x, gain, bias, eps = 1e-5) {
    const n = x.rows, d = x.cols;
    const out = new Tensor(n, d);
    const xhat = new Float32Array(n * d);
    const rstd = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      let mean = 0;
      for (let j = 0; j < d; j++) mean += x.data[i * d + j];
      mean /= d;
      let v = 0;
      for (let j = 0; j < d; j++) { const c = x.data[i * d + j] - mean; v += c * c; }
      v /= d;
      const r = 1 / Math.sqrt(v + eps);
      rstd[i] = r;
      for (let j = 0; j < d; j++) {
        const h = (x.data[i * d + j] - mean) * r;
        xhat[i * d + j] = h;
        out.data[i * d + j] = h * gain.data[j] + bias.data[j];
      }
    }
    return tape.track(out, [x, gain, bias], () => {
      if (!out.grad) return;
      const G = out.grad;
      const dg = gain.requiresGrad ? gain.ensureGrad() : null;
      const db = bias.requiresGrad ? bias.ensureGrad() : null;
      const dx = x.requiresGrad ? x.ensureGrad() : null;
      for (let i = 0; i < n; i++) {
        let sumDh = 0, sumDhXh = 0;
        for (let j = 0; j < d; j++) {
          const g = G[i * d + j];
          const h = xhat[i * d + j];
          if (dg) dg[j] += g * h;
          if (db) db[j] += g;
          const dh = g * gain.data[j];
          sumDh += dh;
          sumDhXh += dh * h;
        }
        if (dx) {
          const r = rstd[i];
          for (let j = 0; j < d; j++) {
            const dh = G[i * d + j] * gain.data[j];
            dx[i * d + j] += (r / d) * (d * dh - sumDh - xhat[i * d + j] * sumDhXh);
          }
        }
      }
    });
  }

  // GELU, tanh approximation (as in GPT-2)
  const GELU_C = Math.sqrt(2 / Math.PI);
  function gelu(tape, x) {
    const out = new Tensor(x.rows, x.cols);
    for (let i = 0; i < x.size; i++) {
      const v = x.data[i];
      out.data[i] = 0.5 * v * (1 + Math.tanh(GELU_C * (v + 0.044715 * v * v * v)));
    }
    return tape.track(out, [x], () => {
      if (!out.grad) return;
      const g = x.ensureGrad();
      for (let i = 0; i < x.size; i++) {
        const v = x.data[i];
        const u = GELU_C * (v + 0.044715 * v * v * v);
        const th = Math.tanh(u);
        const du = GELU_C * (1 + 3 * 0.044715 * v * v);
        const deriv = 0.5 * (1 + th) + 0.5 * v * (1 - th * th) * du;
        g[i] += out.grad[i] * deriv;
      }
    });
  }

  /*
   * Fused multi-head self-attention core.
   * qkv: [T, 3d] laid out as [Q | K | V], each split into nHead heads of size d/nHead.
   * Returns { out: [T,d], probs: Array(nHead) of Float32Array(T*T) }.
   * opts.causal    — mask future positions.
   * opts.scaleMul  — multiplies the standard 1/sqrt(d_head) scale (attention "temperature").
   */
  function attention(tape, qkv, nHead, opts = {}) {
    const T = qkv.rows, d3 = qkv.cols, d = d3 / 3;
    if (!Number.isInteger(d) || d % nHead !== 0) throw new Error('attention: bad head config');
    const hs = d / nHead;
    const causal = opts.causal !== false;
    const scale = (opts.scaleMul ?? 1) / Math.sqrt(hs);
    const Q = qkv.data;
    const out = new Tensor(T, d);
    const probs = [];
    for (let h = 0; h < nHead; h++) {
      const P = new Float32Array(T * T);
      const qo = h * hs, ko = d + h * hs, vo = 2 * d + h * hs;
      for (let i = 0; i < T; i++) {
        const jmax = causal ? i : T - 1;
        let mx = -Infinity;
        for (let j = 0; j <= jmax; j++) {
          let s = 0;
          for (let c = 0; c < hs; c++) s += Q[i * d3 + qo + c] * Q[j * d3 + ko + c];
          s *= scale;
          P[i * T + j] = s;
          if (s > mx) mx = s;
        }
        let sum = 0;
        for (let j = 0; j <= jmax; j++) { const e = Math.exp(P[i * T + j] - mx); P[i * T + j] = e; sum += e; }
        for (let j = 0; j <= jmax; j++) P[i * T + j] /= sum;
        for (let j = jmax + 1; j < T; j++) P[i * T + j] = 0;
        for (let j = 0; j <= jmax; j++) {
          const p = P[i * T + j];
          if (p === 0) continue;
          for (let c = 0; c < hs; c++) out.data[i * d + h * hs + c] += p * Q[j * d3 + vo + c];
        }
      }
      probs.push(P);
    }
    tape.track(out, [qkv], () => {
      if (!out.grad) return;
      const G = out.grad;
      const dQKV = qkv.ensureGrad();
      const dP = new Float32Array(T);
      for (let h = 0; h < nHead; h++) {
        const P = probs[h];
        const qo = h * hs, ko = d + h * hs, vo = 2 * d + h * hs;
        for (let i = 0; i < T; i++) {
          const jmax = causal ? i : T - 1;
          // dP[j] = dO_i · V_j ; dV_j += P_ij * dO_i
          let dot = 0;
          for (let j = 0; j <= jmax; j++) {
            let s = 0;
            const p = P[i * T + j];
            for (let c = 0; c < hs; c++) {
              const g = G[i * d + h * hs + c];
              s += g * Q[j * d3 + vo + c];
              dQKV[j * d3 + vo + c] += p * g;
            }
            dP[j] = s;
            dot += s * p;
          }
          // softmax backward, then through the scaled dot product
          for (let j = 0; j <= jmax; j++) {
            const dS = P[i * T + j] * (dP[j] - dot) * scale;
            if (dS === 0) continue;
            for (let c = 0; c < hs; c++) {
              dQKV[i * d3 + qo + c] += dS * Q[j * d3 + ko + c];
              dQKV[j * d3 + ko + c] += dS * Q[i * d3 + qo + c];
            }
          }
        }
      }
    });
    return { out, probs };
  }

  // Mean cross-entropy over rows. Returns scalar tensor [1,1].
  function crossEntropy(tape, logits, targets) {
    const n = logits.rows, V = logits.cols;
    if (targets.length !== n) throw new Error('crossEntropy: targets length mismatch');
    const soft = new Float32Array(n * V);
    let loss = 0;
    for (let i = 0; i < n; i++) {
      let mx = -Infinity;
      for (let j = 0; j < V; j++) mx = Math.max(mx, logits.data[i * V + j]);
      let sum = 0;
      for (let j = 0; j < V; j++) { const e = Math.exp(logits.data[i * V + j] - mx); soft[i * V + j] = e; sum += e; }
      for (let j = 0; j < V; j++) soft[i * V + j] /= sum;
      loss -= Math.log(Math.max(soft[i * V + targets[i]], 1e-12));
    }
    const out = new Tensor(1, 1, new Float32Array([loss / n]));
    return tape.track(out, [logits], () => {
      if (!out.grad) return;
      const g = logits.ensureGrad();
      const s = out.grad[0] / n;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < V; j++) g[i * V + j] += s * soft[i * V + j];
        g[i * V + targets[i]] -= s;
      }
    });
  }

  return {
    Tensor, Tape, makeRng, randn, filled,
    matmul, matmulBT, add, addBias, embed, layerNorm, gelu, attention, crossEntropy,
  };
});

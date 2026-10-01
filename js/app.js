/* app.js — UI wiring: knobs -> tokenizer/model -> visualizations. */
(function () {
  'use strict';
  const { tensor: T, tokenizer: Tok, model: M, corpora: C, charts: CH } = window.TV;
  const $ = (id) => document.getElementById(id);

  const MAX_HEADS = 8;
  const VAL_EVERY = 25;
  const MAX_HISTORY = 20000;

  const state = {
    tokenizer: null, trainData: [], valData: [], model: null, optim: null,
    rng: T.makeRng(1), step: 0, history: { train: [], val: [] },
    running: false, analysis: null, selLayer: 0, selHead: 0, selQuery: null,
    lastVizUpdate: 0, genToken: 0,
  };

  // ---------------------------------------------------------------- knobs
  const dModelMap = $('dModel').dataset.map.split(',').map(Number);
  function knobs() {
    return {
      corpus: $('corpus').value,
      tokType: $('tokType').value,
      vocab: +$('vocab').value,
      nLayer: +$('nLayer').value,
      dModel: dModelMap[+$('dModel').value],
      nHead: +$('nHead').value,
      blockSize: +$('blockSize').value,
      mlpRatio: +$('mlpRatio').value,
      posEncoding: $('posEnc').value,
      tieWeights: $('tie').checked,
      seed: clampInt(+$('seed').value, 1, 999999, 1337),
      lr: Math.pow(10, +$('lr').value),
      batch: +$('batch').value,
      trainCausal: $('trainCausal').checked,
      temperature: +$('temp').value,
      topK: +$('topk').value,
      topP: +$('topp').value,
      attnScale: +$('ascale').value,
      inferCausal: $('inferCausal').checked,
      genLen: +$('genLen').value,
    };
  }
  function clampInt(v, lo, hi, dflt) { return Number.isFinite(v) ? Math.min(hi, Math.max(lo, Math.round(v))) : dflt; }

  function updateOutputs() {
    const k = knobs();
    $('vocabOut').textContent = k.vocab;
    $('nLayerOut').textContent = k.nLayer;
    $('dModelOut').textContent = k.dModel;
    $('blockOut').textContent = k.blockSize;
    $('mlpOut').textContent = `${k.mlpRatio}×`;
    $('lrOut').textContent = k.lr.toExponential(1);
    $('batchOut').textContent = k.batch;
    $('tempOut').textContent = k.temperature === 0 ? '0 (greedy)' : k.temperature.toFixed(2);
    $('topkOut').textContent = k.topK === 0 ? 'off' : k.topK;
    $('toppOut').textContent = k.topP >= 1 ? 'off' : k.topP.toFixed(2);
    $('ascaleOut').textContent = `${k.attnScale.toFixed(2)}×`;
    $('genOut').textContent = k.genLen;
    $('vocabWrap').hidden = k.tokType === 'char';
    $('customWrap').hidden = k.corpus !== 'custom';
    $('mergesBox').hidden = k.tokType !== 'bpe';
  }

  function fillHeadOptions() {
    const d = dModelMap[+$('dModel').value];
    const sel = $('nHead');
    const prev = +sel.value || 4;
    const opts = [];
    for (let h = 1; h <= MAX_HEADS; h++) if (d % h === 0) opts.push(h);
    sel.replaceChildren(...opts.map((h) => new Option(`${h}  (d_head = ${d / h})`, h)));
    sel.value = String(opts.includes(prev) ? prev : opts.filter((h) => h <= prev).pop());
  }

  // ---------------------------------------------------------------- banner
  let bannerTimer = null;
  function banner(msg, ms = 4000) {
    const b = $('banner');
    b.textContent = msg;
    b.hidden = false;
    clearTimeout(bannerTimer);
    if (ms) bannerTimer = setTimeout(() => { b.hidden = true; }, ms);
  }

  // ---------------------------------------------------------------- build
  function corpusText() {
    const k = knobs();
    if (k.corpus === 'custom') return $('customText').value.slice(0, C.MAX_CUSTOM_CHARS);
    return C.list.find((c) => c.id === k.corpus).text;
  }

  function rebuildTokenizer() {
    const text = corpusText();
    if (text.length < 50) {
      banner('Custom text needs at least 50 characters. Keeping the previous tokenizer.', 0);
      return false;
    }
    const k = knobs();
    const t0 = performance.now();
    state.tokenizer = Tok.createTokenizer(k.tokType, text, k.vocab);
    const ids = state.tokenizer.encode(text);
    const split = Math.floor(ids.length * 0.9);
    // Need enough tokens on each side to form at least a small window.
    if (ids.length - split > 8) {
      state.trainData = ids.slice(0, split);
      state.valData = ids.slice(split);
    } else {
      state.trainData = ids;
      state.valData = [];
    }
    state.tokBuildMs = performance.now() - t0;
    state.corpusChars = [...text].length;
    $('banner').hidden = true;
    return true;
  }

  function rebuildModel(reason) {
    stopTraining();
    const k = knobs();
    try {
      state.model = new M.GPT({
        vocabSize: state.tokenizer.vocabSize, nLayer: k.nLayer, nHead: k.nHead, dModel: k.dModel,
        blockSize: k.blockSize, mlpRatio: k.mlpRatio, posEncoding: k.posEncoding, tieWeights: k.tieWeights, seed: k.seed,
      });
    } catch (e) {
      banner(`Invalid configuration: ${e.message}`, 0);
      return;
    }
    state.optim = new M.Adam(state.model.parameterList(), { lr: k.lr });
    state.rng = T.makeRng(k.seed + 1);
    state.step = 0;
    state.history = { train: [], val: [] };
    state.selLayer = Math.min(state.selLayer, k.nLayer - 1);
    state.selHead = Math.min(state.selHead, k.nHead - 1);
    $('genOut2').replaceChildren();
    if (reason) banner(`${reason} Model re-initialised with random weights — train it to see structure appear.`);
    fillLayerHeadSelects();
    renderArchitecture();
    renderLoss();
    analyze();
  }

  // ---------------------------------------------------------------- training
  function startTraining() {
    if (state.running) return;
    state.running = true;
    $('trainBtn').textContent = 'Pause';
    requestAnimationFrame(trainLoop);
  }
  function stopTraining() {
    state.running = false;
    $('trainBtn').textContent = 'Train';
  }

  function trainLoop() {
    if (!state.running) return;
    const k = knobs();
    state.optim.lr = k.lr;
    const t0 = performance.now();
    try {
      do {
        const { loss, gradNorm } = M.trainStep(state.model, state.optim, state.trainData, { batchSize: k.batch, causal: k.trainCausal, rng: state.rng });
        if (!Number.isFinite(loss)) {
          stopTraining();
          banner('Loss became NaN/Infinity — learning rate is too high. Reset weights and lower it.', 0);
          return;
        }
        state.step++;
        state.lastGradNorm = gradNorm;
        if (state.history.train.length < MAX_HISTORY) state.history.train.push([state.step, loss]);
        if (state.valData.length > 1 && state.step % VAL_EVERY === 0) {
          state.history.val.push([state.step, M.evaluate(state.model, state.valData, { causal: k.trainCausal })]);
        }
      } while (performance.now() - t0 < 30);
    } catch (e) {
      stopTraining();
      banner(`Training stopped: ${e.message}`, 0);
      return;
    }
    renderLoss();
    const now = performance.now();
    if (now - state.lastVizUpdate > 400) { state.lastVizUpdate = now; analyze(); }
    requestAnimationFrame(trainLoop);
  }

  // ---------------------------------------------------------------- analysis
  function analyze() {
    const k = knobs();
    const text = $('prompt').value;
    const allIds = state.tokenizer.encode(text);
    state.analysis = null;
    if (allIds.length) {
      const res = state.model.nextTokenDistribution(allIds, { causal: k.inferCausal, attnScale: k.attnScale });
      state.analysis = { allIds, ...res };
      if (state.selQuery !== null && state.selQuery >= res.context.length) state.selQuery = null;
    }
    renderTokens();
    renderAttention();
    renderPrediction();
    renderResidual();
  }

  // ---------------------------------------------------------------- renderers
  const visible = (s) => s.replace(/ /g, '·').replace(/\n/g, '↵').replace(/\t/g, '→');
  const fmt = (n) => n >= 1e9 ? (n / 1e9).toFixed(2) + ' G' : n >= 1e6 ? (n / 1e6).toFixed(2) + ' M' : n >= 1e3 ? (n / 1e3).toFixed(1) + ' k' : String(Math.round(n));

  function el(tag, attrs = {}, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === 'class') e.className = v;
      else if (k === 'text') e.textContent = v;
      else e.setAttribute(k, v);
    }
    for (const kid of kids) e.append(kid);
    return e;
  }
  function kv(dl, pairs, inline = false) {
    dl.replaceChildren(...pairs.map(([a, b]) => {
      const dt = el('dt', { text: a }), dd = el('dd', { text: b });
      return inline ? el('div', {}, dt, dd) : [dt, dd];
    }).flat());
  }

  function renderArchitecture() {
    const cfg = state.model.config;
    const Tn = cfg.blockSize, d = cfg.dModel, V = cfg.vocabSize, hd = d / cfg.nHead;
    const stage = (title, shape, cls = '') => el('div', { class: 'stage ' + cls }, el('b', { text: title }), el('span', { class: 'shape', text: shape }));
    const arrow = () => el('span', { class: 'arrow', text: '→', 'aria-hidden': 'true' });
    const parts = [
      stage('Tokenizer', `${state.tokenizer.type}, V=${V}`), arrow(),
      stage('Token embed', `[${Tn}×${d}]`), arrow(),
    ];
    if (cfg.posEncoding !== 'none') parts.push(stage(`+ Position (${cfg.posEncoding})`, `[${Tn}×${d}]`), arrow());
    for (let l = 0; l < cfg.nLayer; l++) {
      const b = stage(`Block ${l + 1}`, `[${Tn}×${d}]`, 'block');
      b.append(el('div', { class: 'sub-ops', text: `LN → ${cfg.nHead}-head attn (${hd}/head) → +` }));
      b.append(el('div', { class: 'sub-ops', text: `LN → MLP ${d}→${d * cfg.mlpRatio}→${d} → +` }));
      parts.push(b, arrow());
    }
    parts.push(stage('LayerNorm', `[${Tn}×${d}]`), arrow(),
      stage(cfg.tieWeights ? 'LM head (tied)' : 'LM head', `[${Tn}×${V}]`), arrow(),
      stage('softmax', 'next token'));
    $('pipeline').replaceChildren(...parts);

    const { groups, total } = state.model.parameterBreakdown();
    const max = Math.max(...Object.values(groups));
    $('paramBars').replaceChildren(...Object.entries(groups).filter(([, v]) => v > 0).map(([name, v]) => {
      const fill = el('div', { class: 'fill' });
      fill.style.width = `${(v / max) * 100}%`;
      return el('div', { class: 'hbar' }, el('span', { class: 'lbl', text: name }), el('div', { class: 'track' }, fill),
        el('span', { class: 'val', text: `${fmt(v)} (${((v / total) * 100).toFixed(0)}%)` }));
    }));

    const tokens = state.trainData.length + state.valData.length;
    kv($('modelCard'), [
      ['Parameters', total.toLocaleString()],
      ['Weights (float32)', `${(total * 4 / 1024).toFixed(0)} KB`],
      ['FLOPs / forward', `${fmt(state.model.flopsPerForward())} (full context)`],
      ['Head size', `${hd}`],
      ['Corpus', `${state.corpusChars.toLocaleString()} chars → ${tokens.toLocaleString()} tokens`],
      ['Tokens per parameter', `${(tokens / total).toFixed(2)}  (GPT-scale models use ~20)`],
      ['For scale: GPT-2 small', '124 M params, 12 layers, 12 heads, d=768, ctx 1024'],
    ]);
  }

  function renderTokens() {
    const a = state.analysis;
    const box = $('tokens');
    const tok = state.tokenizer;
    if (!a) {
      box.replaceChildren(el('span', { class: 'hint', text: 'Type a prompt to see it tokenized.' }));
      kv($('tokStats'), [], true);
      return;
    }
    const dropped = a.allIds.length - a.context.length;
    const k = knobs();
    const P = state.selQuery !== null ? a.attn[state.selLayer]?.[state.selHead] : null;
    const n = a.context.length;
    const chips = a.allIds.map((id, idx) => {
      const ctxIdx = idx - dropped;
      const chip = el('span', { class: 'tok' + (id === tok.unkId ? ' unk' : '') + (ctxIdx < 0 ? ' dropped' : '') + (ctxIdx === state.selQuery ? ' selected' : ''),
        title: `id ${id}${id === tok.unkId ? ' (unknown → <unk>)' : ''}${ctxIdx < 0 ? ' — outside context window' : ''}`,
        text: visible(id === tok.unkId ? '<unk>' : tok.tokenString(id)) });
      if (ctxIdx >= 0) {
        chip.dataset.pos = ctxIdx;
        if (P && state.selQuery !== null) {
          const w = el('span', { class: 'w' });
          const v = P[state.selQuery * n + ctxIdx];
          w.style.opacity = String(Math.sqrt(v));
          chip.append(w);
        }
      }
      return chip;
    });
    box.replaceChildren(...chips);
    const chars = [...$('prompt').value].length;
    const unk = a.allIds.filter((i) => i === tok.unkId).length;
    const stats = [
      ['Tokens', String(a.allIds.length)],
      ['Chars / token', (chars / a.allIds.length).toFixed(2)],
      ['Vocab', String(tok.vocabSize)],
      ['Unknown', String(unk)],
    ];
    if (dropped > 0) stats.push(['Truncated', `first ${dropped} tokens fall outside the ${k.blockSize}-token context`]);
    kv($('tokStats'), stats, true);
    if (tok.type === 'bpe') {
      $('merges').replaceChildren(...tok.merges.slice(0, 60).map(([l, r, c]) => el('li', { text: `${visible(l)} + ${visible(r)} → ${visible(l + r)}  (${c})` })));
    }
  }

  function fillLayerHeadSelects() {
    const cfg = state.model.config;
    $('layerSel').replaceChildren(...Array.from({ length: cfg.nLayer }, (_, i) => new Option(String(i + 1), i)));
    $('headSel').replaceChildren(...Array.from({ length: cfg.nHead }, (_, i) => new Option(String(i + 1), i)));
    $('layerSel').value = state.selLayer;
    $('headSel').value = state.selHead;
    const grid = $('allHeads');
    grid.style.gridTemplateColumns = `repeat(${Math.min(cfg.nHead, 8)}, minmax(0, 72px))`;
  }

  function headStats(P, n) {
    let ent = 0, prev = 0, self = 0, first = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) { const p = P[i * n + j]; if (p > 0) ent -= p * Math.log2(p); }
      if (i > 0) prev += P[i * n + i - 1];
      self += P[i * n + i];
      if (i > 0) first += P[i * n];
    }
    const m = Math.max(1, n - 1);
    return { entropy: ent / n, prev: prev / m, self: self / n, first: first / m };
  }

  function renderAttention() {
    const a = state.analysis;
    const k = knobs();
    const canvas = $('heatmap');
    if (!a) {
      CH.heatmap(canvas, new Float32Array(0), 0, [], {});
      $('allHeads').replaceChildren();
      return;
    }
    const tok = state.tokenizer;
    const labels = a.context.map((id) => visible(tok.tokenString(id)));
    const n = a.context.length;
    const P = a.attn[state.selLayer][state.selHead];
    CH.heatmap(canvas, P, n, labels, { causal: k.inferCausal });
    if (state.selQuery !== null && canvas._cell) {
      const { ox, oy, cell } = canvas._cell;
      const ctx = canvas.getContext('2d');
      ctx.strokeStyle = CH.cssVar('--series-2'); ctx.lineWidth = 2;
      ctx.strokeRect(ox - 1, oy + state.selQuery * cell - 1, cell * n + 1, cell + 1);
    }

    const s = headStats(P, n);
    const maxEnt = Math.log2(Math.max(2, n));
    kv($('headStats'), [
      ['Avg entropy', `${s.entropy.toFixed(2)} bits`],
      ['→ previous token', `${(s.prev * 100).toFixed(0)}%`],
      ['→ itself', `${(s.self * 100).toFixed(0)}%`],
      ['→ first token', `${(s.first * 100).toFixed(0)}%`],
    ]);
    let hint = 'Mixed pattern.';
    if (s.prev > 0.5) hint = 'Previous-token head: mostly looks one step back. Classic early-layer pattern.';
    else if (s.first > 0.5) hint = 'Attention sink: parks weight on the first token when nothing else is useful.';
    else if (s.self > 0.5) hint = 'Self/diagonal head: mostly looks at its own position.';
    else if (s.entropy > 0.6 * maxEnt) hint = state.step < 50 ? 'Diffuse: weights spread almost evenly — typical of an untrained model.' : 'Diffuse: averages over many tokens.';
    if (!k.inferCausal) hint += ' Causal mask is OFF — tokens can see the future, which a trained GPT never had during training.';
    $('headHint').textContent = hint;

    const cells = [];
    for (let l = 0; l < a.attn.length; l++) {
      for (let h = 0; h < a.attn[l].length; h++) {
        const c = el('canvas');
        const box = el('div', { class: 'sm' + (l === state.selLayer && h === state.selHead ? ' active' : ''), role: 'button', tabindex: '0', 'aria-label': `Layer ${l + 1} head ${h + 1}` },
          c, el('span', { text: `L${l + 1} · H${h + 1}` }));
        box.dataset.l = l; box.dataset.h = h;
        CH.thumb(c, a.attn[l][h], n, { causal: k.inferCausal });
        cells.push(box);
      }
    }
    $('allHeads').replaceChildren(...cells);
  }

  function renderPrediction() {
    const a = state.analysis;
    const canvas = $('probChart');
    if (!a) { CH.probBars(canvas, []); kv($('probStats'), [], true); return; }
    const k = knobs();
    const probs = M.applySampling(a.logits, { temperature: k.temperature, topK: k.topK, topP: k.topP });
    const raw = M.applySampling(a.logits, { temperature: 1 });
    const order = Array.from(raw.keys()).sort((x, y) => raw[y] - raw[x]).slice(0, 15);
    const tok = state.tokenizer;
    CH.probBars(canvas, order.map((i) => ({ label: visible(i === tok.unkId ? '<unk>' : tok.tokenString(i)), p: probs[i], raw: raw[i] })));
    const H = (p) => { let h = 0; for (const v of p) if (v > 0) h -= v * Math.log2(v); return h; };
    const alive = probs.filter((v) => v > 0).length;
    kv($('probStats'), [
      ['Entropy raw', `${H(raw).toFixed(2)} bits`],
      ['Entropy after knobs', `${H(probs).toFixed(2)} bits`],
      ['Candidates left', `${alive} / ${probs.length}`],
      ['Uniform would be', `${Math.log2(probs.length).toFixed(2)} bits`],
    ], true);
  }

  function renderResidual() {
    const a = state.analysis;
    if (!a) { $('residBars').replaceChildren(); return; }
    const norms = a.residualNorms;
    const max = Math.max(...norms, 1e-9);
    $('residBars').replaceChildren(...norms.map((v, i) => {
      const fill = el('div', { class: 'fill' });
      fill.style.width = `${(v / max) * 100}%`;
      return el('div', { class: 'hbar' }, el('span', { class: 'lbl', text: i === 0 ? 'embeddings' : `after block ${i}` }),
        el('div', { class: 'track' }, fill), el('span', { class: 'val', text: v.toFixed(2) }));
    }));
  }

  function downsample(points, max = 600) {
    if (points.length <= max) return points;
    const k = Math.ceil(points.length / max);
    const out = [];
    for (let i = 0; i < points.length; i += k) {
      const chunk = points.slice(i, i + k);
      out.push([chunk[chunk.length - 1][0], chunk.reduce((s, p) => s + p[1], 0) / chunk.length]);
    }
    return out;
  }

  function renderLoss() {
    const s1 = CH.cssVar('--series-1'), s2 = CH.cssVar('--series-2');
    const train = downsample(state.history.train);
    const series = [{ name: 'train', color: s1, points: train }];
    if (state.history.val.length) series.push({ name: 'validation', color: s2, points: state.history.val, dots: true });
    CH.lossChart($('lossChart'), series, { refLine: Math.log(state.model.config.vocabSize) });
    const legend = [['train (batch loss)', s1]];
    if (state.history.val.length) legend.push(['validation (held-out 10%)', s2]);
    $('lossLegend').replaceChildren(...legend.map(([name, c]) => {
      const i = el('i'); i.style.background = c;
      return el('span', {}, i, name);
    }));
    const last = state.history.train.at(-1);
    const lastVal = state.history.val.at(-1);
    const note = $('lossNote');
    note.hidden = true;
    if (last && lastVal && state.history.val.length >= 3) {
      const recent = state.history.train.slice(-VAL_EVERY);
      const trainAvg = recent.reduce((t, p) => t + p[1], 0) / recent.length;
      const valMin = Math.min(...state.history.val.map((p) => p[1]));
      if (lastVal[1] - trainAvg > 1 && lastVal[1] > valMin + 0.2) {
        note.textContent = 'Overfitting: training loss keeps falling while validation loss rises. The model is memorising this small text instead of learning rules that generalise. Try a bigger corpus, a smaller model, or stop earlier.';
        note.hidden = false;
      }
    }
    $('trainStat').textContent = `step ${state.step}` + (last ? ` · loss ${last[1].toFixed(3)}` : '') +
      (lastVal ? ` · val ${lastVal[1].toFixed(3)}` : '') + (state.lastGradNorm && last ? ` · |grad| ${state.lastGradNorm.toFixed(2)}` : '');
  }

  // ---------------------------------------------------------------- generation
  function generate() {
    const k = knobs();
    const tok = state.tokenizer;
    const promptIds = tok.encode($('prompt').value);
    if (!promptIds.length) { banner('Type a prompt first.'); return; }
    const out = $('genOut2');
    const gen = el('span');
    out.replaceChildren(el('span', { class: 'p', text: tok.decode(promptIds) }), gen);
    const rng = T.makeRng((Math.random() * 2 ** 32) >>> 0);
    const ids = promptIds.slice();
    const myToken = ++state.genToken;
    let produced = 0;
    $('genBtn').disabled = true;
    const tick = () => {
      if (myToken !== state.genToken) return; // superseded
      const t0 = performance.now();
      while (produced < k.genLen && performance.now() - t0 < 25) {
        const { logits } = state.model.nextTokenDistribution(ids, { causal: k.inferCausal, attnScale: k.attnScale });
        const next = M.sampleFrom(M.applySampling(logits, k), rng);
        ids.push(next);
        produced++;
      }
      gen.textContent = tok.decode(ids.slice(promptIds.length));
      if (produced < k.genLen) requestAnimationFrame(tick);
      else $('genBtn').disabled = false;
    };
    requestAnimationFrame(tick);
  }

  // ---------------------------------------------------------------- events
  function on(id, ev, fn) { $(id).addEventListener(ev, fn); }
  function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }

  function init() {
    $('corpus').replaceChildren(...C.list.map((c) => new Option(c.name, c.id)));
    $('corpus').value = 'refinery';
    $('customText').maxLength = C.MAX_CUSTOM_CHARS;
    fillHeadOptions();
    updateOutputs();

    document.querySelectorAll('.controls input, .controls select').forEach((e) => e.addEventListener('input', updateOutputs));

    const retokenize = (why) => { if (rebuildTokenizer()) rebuildModel(why); };
    on('corpus', 'change', () => {
      if (knobs().corpus === 'custom' && $('customText').value.length < 50) {
        banner('Paste your text in the box below (min 50 characters).', 0);
        return;
      }
      retokenize('Training text changed.');
    });
    on('customText', 'input', debounce(() => {
      $('customCount').textContent = `${$('customText').value.length.toLocaleString()} / ${C.MAX_CUSTOM_CHARS.toLocaleString()}`;
      retokenize('Custom text changed.');
    }, 700));
    on('tokType', 'change', () => retokenize('Tokenizer changed.'));
    on('vocab', 'change', () => retokenize('Vocab size changed.'));

    on('dModel', 'input', fillHeadOptions);
    for (const id of ['nLayer', 'dModel', 'nHead', 'blockSize', 'mlpRatio', 'posEnc', 'tie', 'seed']) {
      on(id, 'change', () => rebuildModel('Architecture changed.'));
    }
    on('resetBtn', 'click', () => rebuildModel('Weights reset.'));
    on('trainBtn', 'click', () => (state.running ? stopTraining() : startTraining()));

    // live inference knobs
    for (const id of ['temp', 'topk', 'topp']) on(id, 'input', renderPrediction);
    for (const id of ['ascale', 'inferCausal']) on(id, 'input', analyze);
    on('prompt', 'input', debounce(() => { state.selQuery = null; analyze(); }, 120));

    on('layerSel', 'change', () => { state.selLayer = +$('layerSel').value; renderTokens(); renderAttention(); });
    on('headSel', 'change', () => { state.selHead = +$('headSel').value; renderTokens(); renderAttention(); });
    const pickHead = (e) => {
      const box = e.target.closest('.sm');
      if (!box) return;
      state.selLayer = +box.dataset.l; state.selHead = +box.dataset.h;
      $('layerSel').value = state.selLayer; $('headSel').value = state.selHead;
      renderTokens(); renderAttention();
    };
    on('allHeads', 'click', pickHead);
    on('allHeads', 'keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pickHead(e); } });
    on('tokens', 'click', (e) => {
      const chip = e.target.closest('.tok');
      if (!chip || chip.dataset.pos === undefined) return;
      const pos = +chip.dataset.pos;
      state.selQuery = state.selQuery === pos ? null : pos;
      renderTokens(); renderAttention();
    });
    on('heatmap', 'click', (e) => {
      const c = $('heatmap')._cell;
      if (!c) return;
      const r = $('heatmap').getBoundingClientRect();
      const i = Math.floor((e.clientY - r.top - c.oy) / c.cell);
      if (i >= 0 && i < c.n) { state.selQuery = state.selQuery === i ? null : i; renderTokens(); renderAttention(); }
    });
    on('genBtn', 'click', generate);

    CH.attachTooltip($('lossChart'), $('lossTip'));
    CH.attachTooltip($('heatmap'), $('heatTip'));
    CH.attachTooltip($('probChart'), $('probTip'));

    // theme
    try { const t = localStorage.getItem('tv-theme'); if (t === 'light' || t === 'dark') document.documentElement.dataset.theme = t; } catch (_) { /* storage blocked */ }
    on('themeToggle', 'click', () => {
      const cur = document.documentElement.dataset.theme ||
        (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
      const next = cur === 'dark' ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('tv-theme', next); } catch (_) { /* storage blocked */ }
      renderLoss(); renderAttention(); renderPrediction();
    });
    matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { renderLoss(); renderAttention(); renderPrediction(); });
    window.addEventListener('resize', debounce(() => { renderLoss(); renderAttention(); renderPrediction(); }, 150));

    rebuildTokenizer();
    rebuildModel();
  }

  init();
})();

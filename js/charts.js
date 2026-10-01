/*
 * charts.js — small canvas renderers (loss line, attention heatmap, probability bars).
 * Each renderer stores a hit-test function on the canvas (canvas._hit) that returns
 * tooltip text for a pointer position; attachTooltip() wires it to a tooltip element.
 */
(function (root) {
  'use strict';

  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function hexToRgb(hex) {
    const h = hex.replace('#', '');
    const n = parseInt(h.length === 3 ? h.split('').map((c) => c + c).join('') : h, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  function mix(a, b, t) {
    const A = hexToRgb(a), B = hexToRgb(b);
    return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(',')})`;
  }

  // Size the backing store for crisp drawing on HiDPI screens.
  function setup(canvas, cssHeight) {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(50, canvas.clientWidth || canvas.parentElement.clientWidth);
    const h = cssHeight ?? w;
    canvas.style.height = h + 'px';
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
    return { ctx, w, h };
  }

  function niceTicks(min, max, count = 4) {
    if (!(max > min)) return [min];
    const step0 = (max - min) / count;
    const mag = Math.pow(10, Math.floor(Math.log10(step0)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) || step0;
    const ticks = [];
    for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(10));
    return ticks;
  }

  // ---- loss line chart -----------------------------------------------------
  function lossChart(canvas, series, { refLine } = {}) {
    const { ctx, w, h } = setup(canvas, 220);
    const pad = { l: 40, r: 12, t: 10, b: 24 };
    const all = series.flatMap((s) => s.points);
    const ink2 = cssVar('--ink-2'), muted = cssVar('--muted'), grid = cssVar('--grid'), axis = cssVar('--axis');
    if (!all.length) {
      ctx.fillStyle = muted;
      ctx.textAlign = 'center';
      ctx.fillText('Press Train to start. Loss will appear here.', w / 2, h / 2);
      canvas._hit = null;
      return;
    }
    const xMax = Math.max(1, ...all.map((p) => p[0]));
    let yMax = Math.max(...all.map((p) => p[1]));
    if (refLine) yMax = Math.max(yMax, refLine);
    yMax *= 1.05;
    const X = (x) => pad.l + (x / xMax) * (w - pad.l - pad.r);
    const Y = (y) => pad.t + (1 - y / yMax) * (h - pad.t - pad.b);

    ctx.strokeStyle = grid; ctx.lineWidth = 1; ctx.fillStyle = muted; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    for (const t of niceTicks(0, yMax)) {
      ctx.beginPath(); ctx.moveTo(pad.l, Y(t) + 0.5); ctx.lineTo(w - pad.r, Y(t) + 0.5); ctx.stroke();
      ctx.fillText(t.toFixed(t < 10 ? 1 : 0), pad.l - 6, Y(t));
    }
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    for (const t of niceTicks(0, xMax, 5)) ctx.fillText(String(t), X(t), h - pad.b + 6);
    ctx.strokeStyle = axis;
    ctx.beginPath(); ctx.moveTo(pad.l, h - pad.b + 0.5); ctx.lineTo(w - pad.r, h - pad.b + 0.5); ctx.stroke();

    if (refLine) {
      ctx.save(); ctx.setLineDash([4, 4]); ctx.strokeStyle = ink2;
      ctx.beginPath(); ctx.moveTo(pad.l, Y(refLine)); ctx.lineTo(w - pad.r, Y(refLine)); ctx.stroke();
      ctx.restore();
    }
    for (const s of series) {
      if (!s.points.length) continue;
      ctx.strokeStyle = s.color; ctx.lineWidth = 2; ctx.lineJoin = 'round';
      ctx.beginPath();
      s.points.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
      ctx.stroke();
      if (s.dots) {
        ctx.fillStyle = s.color;
        for (const [x, y] of s.points) { ctx.beginPath(); ctx.arc(X(x), Y(y), 3, 0, Math.PI * 2); ctx.fill(); }
      }
    }
    canvas._hit = (mx) => {
      if (mx < pad.l || mx > w - pad.r) return null;
      const step = ((mx - pad.l) / (w - pad.l - pad.r)) * xMax;
      const lines = [];
      for (const s of series) {
        if (!s.points.length) continue;
        let best = s.points[0];
        for (const p of s.points) if (Math.abs(p[0] - step) < Math.abs(best[0] - step)) best = p;
        lines.push(`${s.name} @ step ${best[0]}: ${best[1].toFixed(3)}`);
      }
      return lines.join('\n');
    };
  }

  // ---- attention heatmap ----------------------------------------------------
  function heatmap(canvas, P, n, labels, { causal = true } = {}) {
    const labelW = Math.min(90, 10 + 7 * Math.max(1, ...labels.map((l) => l.length)));
    const width = Math.max(50, canvas.clientWidth);
    const cell = Math.max(4, Math.min(28, Math.floor((width - labelW - 4) / Math.max(1, n))));
    const size = labelW + cell * n + 4;
    const { ctx } = setup(canvas, size);
    const r0 = cssVar('--ramp-0'), r1 = cssVar('--ramp-1'), masked = cssVar('--masked'), ink2 = cssVar('--ink-2');
    const ox = labelW, oy = labelW;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const isMasked = causal && j > i;
        ctx.fillStyle = isMasked ? masked : mix(r0, r1, Math.sqrt(P[i * n + j])); // sqrt: small weights stay visible
        ctx.fillRect(ox + j * cell, oy + i * cell, cell - (cell > 6 ? 1 : 0), cell - (cell > 6 ? 1 : 0));
      }
    }
    if (cell >= 9) {
      ctx.fillStyle = ink2; ctx.font = `${Math.min(12, cell - 1)}px ui-monospace, Menlo, Consolas, monospace`;
      ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      labels.forEach((l, i) => ctx.fillText(clip(l, labelW), ox - 4, oy + i * cell + cell / 2));
      ctx.save();
      labels.forEach((l, j) => {
        ctx.save(); ctx.translate(ox + j * cell + cell / 2, oy - 4); ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'left'; ctx.fillText(clip(l, labelW), 0, 0); ctx.restore();
      });
      ctx.restore();
    }
    canvas._hit = (mx, my) => {
      const j = Math.floor((mx - ox) / cell), i = Math.floor((my - oy) / cell);
      if (i < 0 || j < 0 || i >= n || j >= n) return null;
      if (causal && j > i) return `${labels[i]} → ${labels[j]}\nmasked (future token)`;
      return `query ${i} ${labels[i]}  →  key ${j} ${labels[j]}\nweight ${(P[i * n + j] * 100).toFixed(1)}%`;
    };
    canvas._cell = { ox, oy, cell, n };
  }

  function clip(s, px) {
    const max = Math.max(2, Math.floor(px / 7) - 1);
    return s.length > max ? s.slice(0, max - 1) + '…' : s;
  }

  // Tiny thumbnail: one pixel per cell, scaled up by CSS.
  function thumb(canvas, P, n, { causal = true } = {}) {
    canvas.width = n; canvas.height = n;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(n, n);
    const A = hexToRgb(cssVar('--ramp-0')), B = hexToRgb(cssVar('--ramp-1')), M = hexToRgb(cssVar('--masked'));
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const k = (i * n + j) * 4;
      const t = Math.sqrt(P[i * n + j]);
      const c = causal && j > i ? M : A.map((v, c2) => v + (B[c2] - v) * t);
      img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }

  // ---- probability bars ------------------------------------------------------
  function probBars(canvas, rows) {
    // rows: [{label, p, raw}]
    const rowH = 18;
    const { ctx, w } = setup(canvas, Math.max(60, rows.length * rowH + 8));
    const labelW = 110, valW = 52;
    const barW = w - labelW - valW - 8;
    const maxP = Math.max(1e-9, ...rows.map((r) => Math.max(r.p, r.raw)));
    const s1 = cssVar('--series-1'), ink = cssVar('--ink'), ink2 = cssVar('--ink-2');
    ctx.textBaseline = 'middle';
    rows.forEach((r, i) => {
      const y = 4 + i * rowH;
      ctx.fillStyle = ink2; ctx.textAlign = 'right';
      ctx.font = '12px ui-monospace, Menlo, Consolas, monospace';
      ctx.fillText(clip(r.label, labelW), labelW - 6, y + rowH / 2);
      const bw = (r.p / maxP) * barW;
      if (bw > 0) {
        ctx.fillStyle = s1;
        roundRight(ctx, labelW, y + 3, Math.max(1, bw), rowH - 6, 4);
        ctx.fill();
      }
      ctx.strokeStyle = ink2; ctx.lineWidth = 1;
      const rw = (r.raw / maxP) * barW;
      ctx.strokeRect(labelW + 0.5, y + 3.5, Math.max(1, rw), rowH - 7);
      ctx.fillStyle = ink; ctx.textAlign = 'left';
      ctx.font = '11px system-ui, -apple-system, "Segoe UI", sans-serif';
      ctx.fillText((r.p * 100).toFixed(1) + '%', labelW + barW + 6, y + rowH / 2);
    });
    canvas._hit = (mx, my) => {
      const i = Math.floor((my - 4) / rowH);
      if (i < 0 || i >= rows.length) return null;
      const r = rows[i];
      return `${r.label}\nafter knobs ${(r.p * 100).toFixed(2)}%\nraw (T=1) ${(r.raw * 100).toFixed(2)}%`;
    };
  }

  function roundRight(ctx, x, y, w, h, r) {
    r = Math.min(r, w, h / 2);
    ctx.beginPath();
    ctx.moveTo(x, y); ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x, y + h); ctx.closePath();
  }

  function attachTooltip(canvas, tip) {
    tip.style.whiteSpace = 'pre';
    canvas.addEventListener('mousemove', (e) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left, my = e.clientY - rect.top;
      const text = canvas._hit ? canvas._hit(mx, my) : null;
      if (!text) { tip.hidden = true; return; }
      tip.textContent = text;
      tip.hidden = false;
      const tw = tip.offsetWidth;
      tip.style.left = Math.min(Math.max(0, mx + 12), rect.width - tw) + 'px';
      tip.style.top = (my + 14) + 'px';
    });
    canvas.addEventListener('mouseleave', () => { tip.hidden = true; });
  }

  root.TV = Object.assign(root.TV || {}, { charts: { lossChart, heatmap, thumb, probBars, attachTooltip, cssVar } });
})(typeof self !== 'undefined' ? self : this);

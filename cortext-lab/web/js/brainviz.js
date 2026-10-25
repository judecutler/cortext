// Neural activity visuals, all canvas, no dependencies:
//   HeadMap   - top-down scalp map interpolated from the electrodes (8 on the Unicorn), with brain regions labelled
//   Traces    - live EEG lines (filtered 0.5-20 Hz by the bridge)
//   ErpPlot   - average response to the attended answer vs the other options (the P300 "peak")
//   EEGFeed   - polls the bridge for trace/band-power/quality/evidence and the ERP
// Honest scope: 8 scalp electrodes cannot localise sources; the map interpolates between electrodes.
import { h } from './util.js';

export const POS = {
  Fp1: [-.3, .95], Fp2: [.3, .95], AF3: [-.25, .75], AF4: [.25, .75], AF7: [-.55, .8], AF8: [.55, .8], F7: [-.8, .6],
  F3: [-.4, .5], Fz: [0, .5], F4: [.4, .5], F8: [.8, .6], FC1: [-.25, .25], FCz: [0, .25], FC2: [.25, .25],
  T7: [-.95, 0], C3: [-.5, 0], Cz: [0, 0], C4: [.5, 0], T8: [.95, 0], CP1: [-.25, -.25], CPz: [0, -.25], CP2: [.25, -.25],
  P7: [-.8, -.6], P3: [-.4, -.5], Pz: [0, -.5], P4: [.4, -.5], P8: [.8, -.6], PO7: [-.55, -.75], PO3: [-.25, -.75],
  POz: [0, -.75], PO4: [.25, -.75], PO8: [.55, -.75], O1: [-.3, -.95], Oz: [0, -.95], O2: [.3, -.95], TP9: [-1, -.3], TP10: [1, -.3],
};

export const REGIONS = [
  { name: 'Frontal', role: 'attention · control', y: 0.62, ch: ['Fz', 'F3', 'F4', 'Fp1', 'Fp2', 'AF3', 'AF4', 'AF7', 'AF8', 'FCz'] },
  { name: 'Central', role: 'sensorimotor', y: 0.16, ch: ['C3', 'Cz', 'C4', 'FC1', 'FC2', 'CP1', 'CP2'] },
  { name: 'Parietal', role: 'P300 · attention', y: -0.36, ch: ['Pz', 'P3', 'P4', 'CPz', 'P7', 'P8', 'TP9', 'TP10'] },   // TP9/TP10: the Muse's P300 sites
  { name: 'Occipital', role: 'vision', y: -0.88, ch: ['PO7', 'POz', 'PO8', 'O1', 'Oz', 'O2', 'PO3', 'PO4'] },
];

export const BAND_INFO = {
  p300: { label: 'P300 response', unit: 'µV', hint: 'Answer flashes minus other flashes, 250–450 ms (this session)' },
  alpha: { label: 'Alpha 8–13 Hz', unit: '%', hint: 'Relaxed / visual idling — strongest at the back of the head' },
  beta: { label: 'Beta 13–30 Hz', unit: '%', hint: 'Active thinking, muscle tension' },
  theta: { label: 'Theta 4–8 Hz', unit: '%', hint: 'Drowsiness, memory load' },
  delta: { label: 'Delta 1–4 Hz', unit: '%', hint: 'Slow waves; high values usually mean movement or poor contact' },
  signal: { label: 'Signal size', unit: 'µV', hint: 'Standard deviation of the filtered signal (5–25 µV is healthy)' },
};

// deep space -> cyan -> violet -> magenta -> white
const STOPS = [[0, [5, 8, 22]], [0.25, [11, 61, 145]], [0.5, [0, 229, 255]], [0.72, [168, 85, 247]], [0.9, [255, 43, 214]], [1, [255, 255, 255]]];
export function colormap(v) {
  v = Math.max(0, Math.min(1, v));
  for (let i = 1; i < STOPS.length; i++) {
    if (v <= STOPS[i][0]) {
      const [a, ca] = STOPS[i - 1], [b, cb] = STOPS[i];
      const t = (v - a) / (b - a);
      return ca.map((c, k) => Math.round(c + (cb[k] - c) * t));
    }
  }
  return STOPS[STOPS.length - 1][1];
}
const QCOL = { good: '#00e5ff', noisy: '#ffb020', flat: '#ff3b5c' };

function fitCanvas(c) {
  const dpr = window.devicePixelRatio || 1;
  const r = c.getBoundingClientRect();
  const w = Math.max(50, Math.round(r.width)), hgt = Math.max(50, Math.round(r.height));
  if (c.width !== w * dpr || c.height !== hgt * dpr) { c.width = w * dpr; c.height = hgt * dpr; }
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return [ctx, w, hgt];
}

/** values: {label: number}; quality: [{label,status}]; returns the normalised range used. */
export class HeadMap {
  constructor(canvas, { regions = true, compact = false } = {}) {
    this.c = canvas; this.regions = regions; this.compact = compact;
    this.grid = document.createElement('canvas'); this.grid.width = this.grid.height = 72;
  }

  draw(values, { quality = [], range = null, title = '' } = {}) {
    const [ctx, w, hgt] = fitCanvas(this.c);
    ctx.clearRect(0, 0, w, hgt);
    const R = this.compact ? Math.min(w, hgt) * 0.42 : Math.min(w * 0.30, hgt * 0.38);
    const cx = this.compact || !this.regions ? w / 2 : Math.max(w / 2, R * 1.12 + 120), cy = hgt / 2 + (this.compact ? 0 : 6);
    const pts = Object.entries(values || {}).filter(([l, v]) => POS[l] && Number.isFinite(v));
    const qmap = Object.fromEntries((quality || []).map((q) => [q.label, q.status]));
    let lo = range ? range[0] : Math.min(...pts.map((p) => p[1])), hi = range ? range[1] : Math.max(...pts.map((p) => p[1]));
    if (!(hi > lo)) { hi = (lo || 0) + 1; }
    // glow
    const g = ctx.createRadialGradient(cx, cy, R * 0.6, cx, cy, R * 1.25);
    g.addColorStop(0, 'rgba(0,229,255,0.10)'); g.addColorStop(1, 'rgba(0,229,255,0)');
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, R * 1.25, 0, 7); ctx.fill();
    // interpolated field (inverse-distance weighting on a small grid, then smoothed up)
    if (pts.length) {
      const G = this.grid.width, gctx = this.grid.getContext('2d'), img = gctx.createImageData(G, G);
      for (let j = 0; j < G; j++) {
        for (let i = 0; i < G; i++) {
          const x = (i / (G - 1)) * 2 - 1, y = 1 - (j / (G - 1)) * 2;
          const k = (j * G + i) * 4;
          if (x * x + y * y > 1.02) { img.data[k + 3] = 0; continue; }
          let num = 0, den = 0, exact = null;
          for (const [l, v] of pts) {
            const dx = x - POS[l][0], dy = y - POS[l][1], d2 = dx * dx + dy * dy;
            if (d2 < 1e-6) { exact = v; break; }
            const wgt = 1 / (d2 * d2 + 0.002);
            num += wgt * v; den += wgt;
          }
          const val = exact ?? num / den;
          const [r, gg, b] = colormap((val - lo) / (hi - lo));
          img.data[k] = r; img.data[k + 1] = gg; img.data[k + 2] = b; img.data[k + 3] = 235;
        }
      }
      gctx.putImageData(img, 0, 0);
      ctx.save();
      ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.clip();
      ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(this.grid, cx - R, cy - R, 2 * R, 2 * R);
      ctx.restore();
    }
    // brain outline: two hemispheres + midline + central sulcus hint
    ctx.save();
    ctx.strokeStyle = 'rgba(220,240,255,0.28)'; ctx.lineWidth = 1.2;
    for (const s of [-1, 1]) {
      ctx.beginPath();
      ctx.ellipse(cx + s * R * 0.43, cy + R * 0.02, R * 0.44, R * 0.86, 0, 0, 7);
      ctx.stroke();
    }
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(cx - R * 0.8, cy - R * 0.02); ctx.quadraticCurveTo(cx, cy - R * 0.18, cx + R * 0.8, cy - R * 0.02); ctx.stroke();
    ctx.restore();
    // head, nose, ears
    ctx.save();
    ctx.shadowColor = '#00e5ff'; ctx.shadowBlur = 14; ctx.strokeStyle = '#7df9ff'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(cx, cy, R, 0, 7); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(cx - R * 0.1, cy - R * 0.99); ctx.lineTo(cx, cy - R * 1.13); ctx.lineTo(cx + R * 0.1, cy - R * 0.99); ctx.stroke();
    for (const s of [-1, 1]) { ctx.beginPath(); ctx.ellipse(cx + s * R * 1.03, cy, R * 0.06, R * 0.16, 0, 0, 7); ctx.stroke(); }
    ctx.restore();
    // region labels
    if (this.regions && !this.compact) {
      ctx.save();
      ctx.font = '600 11px "JetBrains Mono", ui-monospace, monospace'; ctx.textAlign = 'right';
      for (const r of REGIONS) {
        const y = cy - r.y * R * 0.92;
        ctx.fillStyle = 'rgba(180,220,255,0.75)';
        ctx.fillText(r.name.toUpperCase(), cx - R * 1.12, y);
        ctx.fillStyle = 'rgba(160,180,210,0.5)'; ctx.font = '10px "JetBrains Mono", ui-monospace, monospace';
        ctx.fillText(r.role, cx - R * 1.12, y + 13);
        ctx.font = '600 11px "JetBrains Mono", ui-monospace, monospace';
        ctx.strokeStyle = 'rgba(125,249,255,0.25)'; ctx.beginPath(); ctx.moveTo(cx - R * 1.1, y - 4); ctx.lineTo(cx - R * 0.98 * Math.sqrt(Math.max(0, 1 - (r.y * 0.92) ** 2)), y - 4); ctx.stroke();
      }
      ctx.restore();
    }
    // electrodes
    ctx.save();
    ctx.font = `600 ${this.compact ? 9 : 11}px "JetBrains Mono", ui-monospace, monospace`; ctx.textAlign = 'center';
    for (const [l, v] of Object.entries(values || {})) {
      if (!POS[l]) continue;
      const x = cx + POS[l][0] * R, y = cy - POS[l][1] * R;
      const col = QCOL[qmap[l]] || '#cfe9ff';
      ctx.shadowColor = col; ctx.shadowBlur = 10;
      ctx.fillStyle = 'rgba(5,8,22,0.85)'; ctx.strokeStyle = col; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, this.compact ? 5 : 7, 0, 7); ctx.fill(); ctx.stroke();
      ctx.shadowBlur = 0;
      ctx.fillStyle = '#e8f6ff';
      ctx.fillText(l, x, y - (this.compact ? 8 : 12));
    }
    ctx.restore();
    if (title) {
      ctx.fillStyle = 'rgba(200,230,255,.8)'; ctx.font = '600 12px "JetBrains Mono", ui-monospace, monospace'; ctx.textAlign = 'left';
      ctx.fillText(title, 8, 16);
    }
    return [lo, hi];
  }
}

export class Traces {
  constructor(canvas, { uv = 50 } = {}) { this.c = canvas; this.uv = uv; }
  draw(trace) {
    const [ctx, w, hgt] = fitCanvas(this.c);
    ctx.clearRect(0, 0, w, hgt);
    if (!trace || !trace.data || !trace.data.length) {
      ctx.fillStyle = 'rgba(160,180,210,.6)'; ctx.font = '13px ui-monospace, monospace';
      ctx.fillText('no EEG signal', 12, 22);
      return;
    }
    const n = trace.data.length, left = 46, rowH = hgt / n;
    const q = Object.fromEntries((trace.quality || []).map((x) => [x.label, x.status]));
    ctx.font = '600 11px "JetBrains Mono", ui-monospace, monospace';
    trace.data.forEach((arr, c) => {
      const mid = rowH * (c + 0.5);
      const lab = trace.labels[c];
      ctx.strokeStyle = 'rgba(125,249,255,0.07)'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(left, mid); ctx.lineTo(w, mid); ctx.stroke();
      ctx.fillStyle = QCOL[q[lab]] || '#9fb3c8';
      ctx.fillText(lab, 6, mid + 4);
      const col = c % 2 ? '#a855f7' : '#00e5ff';
      ctx.strokeStyle = col; ctx.lineWidth = 1.3; ctx.shadowColor = col; ctx.shadowBlur = 6;
      ctx.beginPath();
      const step = (w - left) / Math.max(1, arr.length - 1);
      arr.forEach((v, i) => {
        const y = mid - Math.max(-1, Math.min(1, v / this.uv)) * rowH * 0.48;
        if (i) ctx.lineTo(left + i * step, y); else ctx.moveTo(left, y);
      });
      ctx.stroke(); ctx.shadowBlur = 0;
    });
    ctx.fillStyle = 'rgba(160,180,210,.55)'; ctx.font = '10px ui-monospace, monospace'; ctx.textAlign = 'right';
    ctx.fillText(`±${this.uv} µV · ${(trace.data[0].length / trace.hz).toFixed(0)} s · 0.5–20 Hz`, w - 6, hgt - 4);
    ctx.textAlign = 'left';
  }
}

export class ErpPlot {
  constructor(canvas) { this.c = canvas; }
  draw(erp, channel = 'Pz') {
    const [ctx, w, hgt] = fitCanvas(this.c);
    ctx.clearRect(0, 0, w, hgt);
    const pad = { l: 40, r: 10, t: 14, b: 24 };
    ctx.font = '11px "JetBrains Mono", ui-monospace, monospace';
    if (!erp || !erp.target || !erp.nontarget) {
      ctx.fillStyle = 'rgba(160,180,210,.6)';
      ctx.fillText('Builds up after the first finished trials of a session…', pad.l, hgt / 2);
      return;
    }
    let ci = erp.labels.indexOf(channel);
    if (ci < 0) ci = 0;
    const T = erp.target[ci], N = erp.nontarget[ci], ts = erp.times;
    const all = [...T, ...N];
    const lim = Math.max(3, ...all.map(Math.abs)) * 1.15;
    const X = (t) => pad.l + ((t - ts[0]) / (ts[ts.length - 1] - ts[0])) * (w - pad.l - pad.r);
    const Y = (v) => pad.t + (1 - (v + lim) / (2 * lim)) * (hgt - pad.t - pad.b);
    ctx.fillStyle = 'rgba(255,43,214,0.08)';
    ctx.fillRect(X(0.25), pad.t, X(0.45) - X(0.25), hgt - pad.t - pad.b);
    ctx.strokeStyle = 'rgba(125,249,255,0.15)';
    ctx.beginPath(); ctx.moveTo(pad.l, Y(0)); ctx.lineTo(w - pad.r, Y(0)); ctx.stroke();
    ctx.fillStyle = 'rgba(160,180,210,.7)';
    for (const t of [0, 0.2, 0.4, 0.6, 0.8]) if (t >= ts[0] - 0.03 && t <= ts[ts.length - 1] + 0.03) ctx.fillText(`${t * 1000}`, X(t) - 8, hgt - 8);
    ctx.fillText(`+${lim.toFixed(0)}µV`, 2, pad.t + 8); ctx.fillText(`-${lim.toFixed(0)}`, 8, hgt - pad.b);
    const line = (arr, col, width) => {
      ctx.strokeStyle = col; ctx.lineWidth = width; ctx.shadowColor = col; ctx.shadowBlur = 10;
      ctx.beginPath(); arr.forEach((v, i) => (i ? ctx.lineTo(X(ts[i]), Y(v)) : ctx.moveTo(X(ts[i]), Y(v)))); ctx.stroke();
      ctx.shadowBlur = 0;
    };
    line(N, '#00e5ff', 1.6);
    line(T, '#ff2bd6', 2.4);
    ctx.fillStyle = '#ff2bd6'; ctx.fillText(`answer flashes (n=${erp.n_target})`, pad.l + 6, pad.t + 10);
    ctx.fillStyle = '#00e5ff'; ctx.fillText(`other flashes (n=${erp.n_nontarget})`, pad.l + 6, pad.t + 24);
    ctx.fillStyle = 'rgba(160,180,210,.7)'; ctx.textAlign = 'right'; ctx.fillText(`${erp.labels[ci]} · ms after flash`, w - pad.r, pad.t + 10); ctx.textAlign = 'left';
  }
}

/** Per-option evidence bars (DOM). */
export function evidenceBars(container, ev) {
  if (!ev || !ev.probabilities || !Object.keys(ev.probabilities).length) {
    container.replaceChildren(h('div', { class: 'muted' }, 'Waiting for a trial…'));
    return;
  }
  const best = ev.best_option_id;
  const entries = Object.entries(ev.probabilities).sort((a, b) => b[1] - a[1]);
  container.replaceChildren(
    ...entries.map(([id, p]) => h('div', { class: 'evrow' + (id === best && ev.final ? ' best' : '') },
      h('span', { class: 'evlab' }, id), h('span', { class: 'evbar' }, h('i', { style: { width: `${(p * 100).toFixed(1)}%` } })),
      h('span', { class: 'evpct' }, `${Math.round(p * 100)}%`))),
    h('div', { class: 'muted small' }, `trial #${ev.trial_id} · ${ev.final ? 'final' : 'live'} · scorer: ${ev.scorer} · ${ev.n_epochs_used}/${ev.n_epochs} epochs used` +
      (ev.n_epochs_rejected ? ` (${ev.n_epochs_rejected} rejected as artifacts)` : '') + ' · uncalibrated'));
}

/** Values to paint on the head map for a mode. */
export function mapValues(mode, trace, erp) {
  const labels = (trace && trace.labels) || (erp && erp.labels) || [];
  if (mode === 'p300') {
    if (!erp || !erp.p300_map) return null;
    return Object.fromEntries(erp.labels.map((l, i) => [l, erp.p300_map[i]]));
  }
  if (mode === 'signal') return Object.fromEntries((trace?.quality || []).map((q) => [q.label, q.std_uv]));
  const b = trace?.bands?.[mode];
  if (!b) return null;
  return Object.fromEntries(labels.map((l, i) => [l, b[i] * 100]));
}

export function regionBands(trace) {
  if (!trace || !trace.bands || !trace.bands.alpha) return [];
  return REGIONS.map((r) => {
    const idx = trace.labels.map((l, i) => (r.ch.includes(l) ? i : -1)).filter((i) => i >= 0);
    if (!idx.length) return null;
    const avg = (band) => idx.reduce((a, i) => a + trace.bands[band][i], 0) / idx.length;
    return { ...r, channels: idx.map((i) => trace.labels[i]), delta: avg('delta'), theta: avg('theta'), alpha: avg('alpha'), beta: avg('beta') };
  }).filter(Boolean);
}

export class EEGFeed {
  constructor({ traceSeconds = 5, traceMs = 250, erpMs = 2000 } = {}) {
    this.opts = { traceSeconds, traceMs, erpMs };
    this.trace = null; this.erp = null; this.cbs = new Set(); this.timers = []; this.paused = false;
  }
  on(fn) { this.cbs.add(fn); return () => this.cbs.delete(fn); }
  _emit() { this.cbs.forEach((fn) => { try { fn(this); } catch (e) { console.error(e); } }); }
  async _poll(path, key) {
    if (this.paused) return;
    try {
      const r = await fetch(path, { cache: 'no-store' });
      if (r.ok) { this[key] = await r.json(); this._emit(); }
    } catch { /* bridge offline */ }
  }
  start() {
    const tp = () => this._poll(`/api/eeg/trace?seconds=${this.opts.traceSeconds}`, 'trace');
    const ep = () => this._poll('/api/eeg/erp', 'erp');
    tp(); ep();
    this.timers.push(setInterval(tp, this.opts.traceMs), setInterval(ep, this.opts.erpMs));
    return this;
  }
  stop() { this.timers.forEach(clearInterval); this.timers = []; this.cbs.clear(); }
}

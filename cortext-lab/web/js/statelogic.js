// "Vibe meter" maths: crude, for-fun estimates of stress / tiredness / excitement / focus from the band powers the
// bridge already sends with every EEG trace. Pure functions, no DOM, no network, no storage: it only READS the trace.
//
// Honest scope: these are textbook band-power ratios (engagement index, theta/beta, alpha suppression) on 8 dry
// electrodes. They are NOT validated against self-report on our data, beta is contaminated by jaw/forehead muscle, and
// every number is shown relative to the person's own recent average, never as an absolute "stress level".
import { REGIONS } from './brainviz.js';

export const STATES = [
  { key: 'focus', label: 'Focus', hint: 'frontal beta vs theta (engagement)', lo: 'drifting', hi: 'locked in' },
  { key: 'stress', label: 'Stress', hint: 'frontal beta vs alpha (also rises with jaw tension)', lo: 'calm', hi: 'tense' },
  { key: 'excitement', label: 'Excitement', hint: 'beta up and posterior alpha suppressed', lo: 'mellow', hi: 'buzzing' },
  { key: 'tiredness', label: 'Tiredness', hint: 'slow (theta + alpha) vs beta', lo: 'alert', hi: 'drowsy' },
];

const EPS = 1e-4;
const FRONTAL = new Set(REGIONS.find((r) => r.name === 'Frontal').ch);
const POSTERIOR = new Set([...REGIONS.find((r) => r.name === 'Parietal').ch, ...REGIONS.find((r) => r.name === 'Occipital').ch]);
const WARMUP_S = 15;          // seconds of good signal before showing numbers
const SMOOTH_TAU_S = 4;       // display smoothing
const BASE_TAU_S = 90;        // personal baseline memory
const MOVE_DELTA = 0.6;       // mean relative delta above this = movement / bad contact: hold the last values
const MIN_SD = 0.15;          // floor on the baseline spread so a quiet start does not blow up the scores

/** Mean relative power of `band` over the given channel indices. */
function avg(trace, idx, band) {
  const b = trace.bands[band];
  let s = 0;
  for (const i of idx) s += b[i];
  return s / idx.length;
}

/**
 * Raw log-ratio indices for one trace, or null when there is nothing usable.
 * Returns { idx: {focus, stress, excitement, tiredness}, nGood, moving } .
 */
export function rawIndices(trace) {
  if (!trace || !trace.bands || !trace.bands.alpha || !Array.isArray(trace.labels)) return null;
  const q = Object.fromEntries((trace.quality || []).map((x) => [x.label, x.status]));
  const good = trace.labels.map((l, i) => (q[l] === undefined || q[l] === 'good' ? i : -1)).filter((i) => i >= 0);
  if (good.length < 2) return { idx: null, nGood: good.length, moving: false };
  const pick = (set) => {
    const sel = good.filter((i) => set.has(trace.labels[i]));
    return sel.length ? sel : good;
  };
  const front = pick(FRONTAL), post = pick(POSTERIOR);
  const moving = avg(trace, good, 'delta') > MOVE_DELTA;
  const ln = Math.log;
  const idx = {
    focus: ln((avg(trace, front, 'beta') + EPS) / (avg(trace, front, 'theta') + EPS)),
    stress: ln((avg(trace, front, 'beta') + EPS) / (avg(trace, front, 'alpha') + EPS)),
    excitement: ln((avg(trace, good, 'beta') + EPS) / (avg(trace, post, 'alpha') + EPS)),
    tiredness: ln((avg(trace, good, 'theta') + avg(trace, good, 'alpha') + EPS) / (avg(trace, good, 'beta') + EPS)),
  };
  return { idx, nGood: good.length, moving };
}

/** Keeps a slow personal baseline per index and turns each new trace into 0-100 scores (50 = your usual). */
export class StateTracker {
  constructor() { this.reset(); }

  reset() {
    this.base = {};     // key -> { m, v }
    this.smooth = {};   // key -> smoothed z
    this.t = null;      // last update time (ms)
    this.goodS = 0;     // seconds of usable signal seen
    this.scores = null; // { key: 0..100 }
    this.status = 'waiting';
  }

  /** @param trace bridge trace payload; @param nowMs wall clock in ms. Returns {status, scores, goodS}. */
  update(trace, nowMs) {
    const raw = rawIndices(trace);
    const dt = this.t == null ? 0 : Math.min(2, Math.max(0, (nowMs - this.t) / 1000));
    this.t = nowMs;
    if (!raw || !raw.idx) { this.status = raw ? 'contact' : 'waiting'; return this._out(); }
    if (raw.moving) { this.status = 'moving'; return this._out(); }   // hold values; do not learn from artifacts
    this.goodS += dt;
    for (const { key } of STATES) {
      const x = raw.idx[key];
      if (!Number.isFinite(x)) continue;
      const b = this.base[key] || (this.base[key] = { m: x, v: MIN_SD * MIN_SD, n: 0 });
      const z = (x - b.m) / Math.max(Math.sqrt(b.v), MIN_SD);   // scored against the baseline BEFORE this sample
      b.n += 1;
      const a = Math.max(dt / BASE_TAU_S, 1 / b.n);        // fast at the start, slow once warmed up
      const d = x - b.m;
      b.m += a * d;
      b.v += a * (d * d - b.v);
      const s = dt ? Math.min(1, dt / SMOOTH_TAU_S) : 1;
      this.smooth[key] = this.smooth[key] == null ? z : this.smooth[key] + s * (z - this.smooth[key]);
    }
    if (this.goodS < WARMUP_S) { this.status = 'calibrating'; return this._out(); }
    this.status = 'ok';
    this.scores = Object.fromEntries(STATES.map(({ key }) => [key, Math.round(50 + 50 * Math.tanh(this.smooth[key] / 2))]));
    return this._out();
  }

  _out() { return { status: this.status, scores: this.scores, goodS: this.goodS, warmupS: WARMUP_S }; }
}

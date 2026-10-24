// Prediction hook. Modes:
//   off   - nothing shown ("EEG prediction: not connected")
//   mock  - SIMULATED scores so the demo works without a model. It is given the true answer, so it is NOT evidence
//           about the brain; every readout and every stored prediction is labelled source:"mock".
//   ws    - a real model server (see tools/mock_predictor_server.py for the protocol). The app never sends
//           is_target / true_answer to it, so a model cannot see the labels.
import { randn } from './util.js';

export class Predictor {
  constructor({ mode = 'mock', url = 'ws://localhost:8765', mockSnr = 0.8 } = {}) {
    this.mode = mode;
    this.url = url;
    this.mockSnr = mockSnr;
    this.connected = false;
    this.liveCbs = new Set();
    this._trial = null;
    this._resolve = null;
    this.ws = null;
    if (mode === 'ws') this._connect();
  }

  get label() { return this.mode === 'mock' ? 'MOCK (simulated, uses ground truth)' : this.mode === 'ws' ? 'ws model' : 'off'; }
  status() { return { mode: this.mode, connected: this.mode === 'ws' ? this.connected : this.mode === 'mock', url: this.url }; }
  onLive(fn) { this.liveCbs.add(fn); return () => this.liveCbs.delete(fn); }
  _live(p300, cat) { this.liveCbs.forEach((fn) => fn({ p300, category_scores: cat || {}, source: this.mode })); }

  _connect() {
    try { this.ws = new WebSocket(this.url); } catch { return; }
    this.ws.onopen = () => { this.connected = true; };
    this.ws.onclose = () => { this.connected = false; if (this.mode === 'ws') setTimeout(() => this._connect(), 2000); };
    this.ws.onerror = () => { try { this.ws.close(); } catch { /* ignore */ } };
    this.ws.onmessage = (ev) => {
      let m; try { m = JSON.parse(ev.data); } catch { return; }
      if (m.type === 'live') this._live(m.p300 || {}, m.category_scores);
      else if (m.type === 'prediction' && this._resolve && (this._trial == null || m.trial_id === this._trial.trial_id)) {
        this._resolve({
          source: 'ws', best_option_id: m.best_option_id, confidence: m.confidence, p300: m.p300 || {},
          category_scores: m.category_scores || {},
        });
        this._resolve = null;
      }
    };
  }

  _send(o) { if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify(o)); }

  /** info: { session_id, trial_id, options:[{id,category,label,position}], truth_for_mock } */
  startTrial(info) {
    this._trial = info;
    this._sum = Object.fromEntries(info.options.map((o) => [o.id, 0]));
    this._n = Object.fromEntries(info.options.map((o) => [o.id, 0]));
    // Mock "participant" attends the true answer (the mock cannot do better than ground truth by design).
    this._attended = info.truth_for_mock;
    if (this.mode === 'ws') {
      this._send({ type: 'trial_start', session_id: info.session_id, trial_id: info.trial_id,
        options: info.options.map(({ id, category, label, position }) => ({ id, category, label, position })) });
    }
    this._live(Object.fromEntries(info.options.map((o) => [o.id, 0])));
  }

  noteFlash(m) {
    if (this.mode === 'ws') {
      this._send({ type: 'flash', trial_id: m.trial_id, option_id: m.option_id, flash_index: m.flash_index,
        repeat_index: m.repeat_index, t_ms: m.t_ms, wall_ms: m.wall_ms });
    } else if (this.mode === 'mock' && this._sum) {
      const id = m.option_id;
      this._sum[id] += (id === this._attended ? this.mockSnr : 0) + randn();
      this._n[id] += 1;
      const avg = Object.fromEntries(Object.keys(this._sum).map((k) => [k, this._sum[k] / Math.max(1, this._n[k])]));
      this._live(avg);
    }
  }

  async finishTrial() {
    const t = this._trial;
    if (!t) return null;
    if (this.mode === 'mock') {
      const avg = Object.fromEntries(Object.keys(this._sum).map((k) => [k, this._sum[k] / Math.max(1, this._n[k])]));
      return { source: 'mock', ...argmaxSoftmax(avg), p300: avg, category_scores: {} };
    }
    if (this.mode === 'ws') {
      const p = new Promise((res) => { this._resolve = res; });
      this._send({ type: 'trial_end', trial_id: t.trial_id });
      return Promise.race([p, new Promise((res) => setTimeout(() => res({ source: 'ws', timeout: true }), 3000))]);
    }
    return null;
  }
}

function argmaxSoftmax(scores, k = 3) {
  const ids = Object.keys(scores);
  if (!ids.length) return { best_option_id: null, confidence: null };
  const exps = ids.map((i) => Math.exp(k * scores[i]));
  const z = exps.reduce((a, b) => a + b, 0);
  let bi = 0;
  ids.forEach((_, i) => { if (exps[i] > exps[bi]) bi = i; });
  return { best_option_id: ids[bi], confidence: exps[bi] / z };
}

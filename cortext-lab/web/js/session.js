// SessionRunner: owns the trial queue, the per-trial lifecycle, operator actions (next / discard / redo / pause /
// stop) and persistence. Trial types (trialtypes/*.js) only drive the phases of one trial.
import { db, trialKey } from './db.js';
import { mulberry32, median, mean, percentile } from './util.js';
import { makeTrialRecord, applyValidity } from './trial.js';
import { getTrialType } from './trialtypes/index.js';
import { SCHEMA_VERSION } from './schema.js';

export class InterruptError extends Error {
  constructor(kind) { super(kind); this.kind = kind; }
}

const INTERRUPTIBLE = new Set(['prompt', 'flash', 'response']);
const deferred = () => { let resolve, reject; const promise = new Promise((a, b) => { resolve = a; reject = b; }); promise.catch(() => {}); return { promise, resolve, reject }; };

export function measureRefresh(frames = 40) {
  return new Promise((resolve) => {
    const ts = [];
    const f = (t) => { ts.push(t); if (ts.length < frames) requestAnimationFrame(f); else {
      const d = ts.slice(1).map((x, i) => x - ts[i]);
      const m = median(d);
      resolve({ frame_ms: m, hz: 1000 / m });
    } };
    requestAnimationFrame(f);
  });
}

export class SessionRunner {
  constructor({ session, settings, stimuli, specs, markers, bridge, predictor, ui }) {
    Object.assign(this, { session, settings, stimuli, markers, bridge, predictor, ui });
    this.queue = specs.slice();
    this.trials = [];
    this.current = null;
    this.phase = 'idle';
    this.paused = false;
    this.stopped = false;
    this.token = { aborted: false };
    this.timing = { onset_err: [], dur_err: [], frame_p95: [], dropped: 0, flashes: 0 };
    this.listeners = new Set();
    this._int = null; this._start = null; this._post = null; this._resp = null; this._pause = null;
    this._needStart = true;
    this._autoTimer = null;
  }

  on(fn) { this.listeners.add(fn); return () => this.listeners.delete(fn); }
  notify() { this.listeners.forEach((fn) => { try { fn(this); } catch (e) { console.error(e); } }); }
  setPhase(p) { this.phase = p; this.notify(); }
  get byId() { return this.stimuli.byId; }
  get plannedTotal() { return this.session.planned_trials; }

  async saveSession() { await db.put('sessions', this.session); }
  async saveTrial(t) { await db.put('trials', t); }

  // ---------------- main loop ----------------
  async start() {
    const s = this.session;
    this.markers.setSession(s.session_id);
    this.bridge.hello({ session_id: s.session_id, participant_id: s.participant_id, device: s.device });
    const refresh = await measureRefresh();
    s.refresh_ms_est = refresh.frame_ms; s.refresh_hz_est = refresh.hz;
    s.status = 'running'; s.started_iso = new Date().toISOString();
    await this.saveSession();
    this.markers.emit('session_start', { extra: { participant_id: s.participant_id, device: s.device, refresh_hz_est: refresh.hz, settings: this.settings } });
    this.notify();
    try {
      while (this.queue.length && !this.stopped) {
        await this._waitIfPaused();
        if (this.stopped) break;
        const spec = this.queue.shift();
        const outcome = await this._runTrial(spec);
        if (outcome === 'redo' || outcome === 'paused' || outcome === 'tab_hidden') this.queue.unshift({ ...spec, _attempt: (spec._attempt || 0) + 1 });
        if (outcome === 'stop') break;
      }
    } finally {
      s.status = this.stopped || this.queue.length ? 'aborted' : 'finished';
      s.ended_iso = new Date().toISOString();
      this.markers.emit('session_end', { extra: { status: s.status } });
      await this.markers.flush();
      await this.saveSession();
      this.setPhase('finished');
      this.ui.onFinished?.(this);
    }
  }

  async _waitIfPaused() {
    while (this.paused && !this.stopped) {
      this.setPhase('paused');
      this._pause = deferred();
      await this._pause.promise;
    }
  }

  async _runTrial(spec) {
    const s = this.session;
    const type = getTrialType(spec.type);
    const trialId = s.next_trial_id++;
    const attempt = spec._attempt || 0;
    const seed = (s.seed * 2654435761 + trialId * 97) >>> 0;
    const rng = mulberry32(seed);
    const trial = makeTrialRecord({ sessionId: s.session_id, spec, trialId, attempt, rng, settings: this.settings, byId: this.byId });
    trial.rng_seed = seed;
    this.trials.push(trial);
    this.current = trial;
    await this.saveSession();
    await this.saveTrial(trial);
    this.ui.renderTrial(trial);
    this.notify();

    if (this._needStart) {
      this.setPhase('ready');
      this._start = deferred();
      const go = await this._start.promise;
      this._needStart = false;
      if (go === 'stop') { applyValidity(trial, { valid: false, reason: 'aborted', by: 'auto' }); trial.status = 'aborted'; await this.saveTrial(trial); return 'stop'; }
    }

    this._int = deferred();
    this.token = { aborted: false };
    let outcome = 'complete';
    try {
      await type.run(this._makeCtx(trial, rng), trial);
    } catch (e) {
      if (e instanceof InterruptError) outcome = e.kind; else { console.error(e); outcome = 'error'; trial.notes += ` error: ${e.message}`; }
    }
    this.bridge.quiet = false;
    this.ui.clearFlash?.();
    this._resp = null;
    if (outcome === 'tab_hidden') this._needStart = true;   // wait for the operator after the tab comes back

    if (outcome === 'complete') {
      await this.saveTrial(trial);
      outcome = await this._postPhase(trial);
    } else {
      const why = { redo: 'redo', discard: 'unspecified', paused: 'paused', pause: 'paused', stop: 'aborted', tab_hidden: 'tab_hidden', error: 'equipment_glitch' }[outcome] || 'aborted';
      trial.status = outcome === 'discard' ? 'discarded' : 'aborted';
      trial.ended_wall_ms = Date.now();
      applyValidity(trial, { valid: false, reason: why, by: outcome === 'discard' || outcome === 'redo' ? 'operator' : 'auto' });
      this.markers.emit('trial_aborted', { ...this._tf(trial), extra: { reason: outcome } });
      await this.saveTrial(trial);
      if (outcome === 'pause') outcome = 'paused';
      if (outcome === 'discard' || outcome === 'error') outcome = 'next';
    }
    if (outcome === 'redo') this.markers.emit('trial_redo', this._tf(trial));
    this.current = null;
    this.notify();
    return outcome;
  }

  // ---------------- post-trial (feedback, prediction, operator decision) ----------------
  async _postPhase(trial) {
    this.setPhase('post');
    this._post = deferred();
    this.ui.showPost(trial, this);
    if (this.settings.post_mode === 'auto') {
      this._autoTimer = setTimeout(() => this._post.resolve('next'), this.settings.auto_advance_ms);
    }
    const outcome = await this._post.promise;
    clearTimeout(this._autoTimer);
    this.ui.hidePost();
    if (outcome === 'redo') {
      applyValidity(trial, { valid: false, reason: 'redo', by: 'operator' });
      this.markers.emit('trial_invalid', { ...this._tf(trial), extra: { reason: 'redo' } });
      await this.saveTrial(trial);
    }
    return outcome;
  }

  _tf(trial) {
    return { trial_id: trial.trial_id, trial_index: trial.trial_index, true_answer_id: trial.true_answer_id };
  }

  _makeCtx(trial, rng) {
    return {
      runner: this, settings: this.settings, ui: this.ui, markers: this.markers, bridge: this.bridge,
      predictor: this.predictor, rng, token: this.token,
      mark: (event, fields = {}, frameMs = null) => this.markers.emit(event, { ...this._tf(trial), ...fields }, frameMs),
      frame: (fn) => new Promise((res) => requestAnimationFrame((ts) => { fn(ts); res(ts); })),
      guard: (p) => Promise.race([p, this._int.promise]),
      interrupt: (kind) => this._interrupt(kind),
      setPhase: (p) => this.setPhase(p),
      waitResponse: (timeoutMs) => {
        this._resp = deferred();
        if (timeoutMs > 0) setTimeout(() => this._resp && this._resp.resolve(null), timeoutMs);
        return this._resp.promise;
      },
      save: () => this.saveTrial(trial),
      reportTiming: (r) => this._reportTiming(r),
    };
  }

  _reportTiming(r) {
    const T = this.timing;
    for (const x of r.results) {
      if (x.onset_err_ms != null) T.onset_err.push(x.onset_err_ms);
      if (x.dur_err_ms != null) T.dur_err.push(x.dur_err_ms);
    }
    T.flashes += r.results.length;
    if (r.frame.p95_ms) T.frame_p95.push(r.frame.p95_ms);
    T.dropped += r.frame.dropped || 0;
    T.last_frame = r.frame;
    this.notify();
  }

  timingSummary() {
    const T = this.timing;
    const abs = T.onset_err.map(Math.abs);
    return {
      flashes: T.flashes, frame: T.last_frame, dropped: T.dropped,
      onset_err_mean: mean(T.onset_err), onset_err_p95: percentile(abs, 95), onset_err_max: abs.length ? Math.max(...abs) : NaN,
      dur_err_mean: mean(T.dur_err), dur_err_p95: percentile(T.dur_err.map(Math.abs), 95),
    };
  }

  _interrupt(kind) {
    if (!INTERRUPTIBLE.has(this.phase) || !this._int) return false;
    this.token.aborted = true; this.token.reason = kind;
    this._int.reject(new InterruptError(kind));
    return true;
  }

  // ---------------- operator actions (buttons + hotkeys) ----------------
  next() {
    if (this.phase === 'ready' && this._start) { if (!this.paused) this._start.resolve('go'); }
    else if (this.phase === 'post' && this._post) this._post.resolve('next');
  }

  discard() {
    const t = this.current;
    if (this.phase === 'post' && t) {
      clearTimeout(this._autoTimer);
      if (t.valid) {
        applyValidity(t, { valid: false, reason: 'unspecified', by: 'operator' });
        this.markers.emit('trial_invalid', { ...this._tf(t), extra: { reason: 'unspecified' } });
      }
      this.saveTrial(t);
      this.ui.showPost(t, this);
      this.notify();
    } else this._interrupt('discard');
  }

  setReason(reason, note = '') {
    const t = this.current;
    if (!t || this.phase !== 'post') return;
    applyValidity(t, { valid: false, reason, note, by: 'operator' });
    this.saveTrial(t);
    this.ui.showPost(t, this);
    this.notify();
  }

  redo() {
    if (this.phase === 'post' && this._post) this._post.resolve('redo');
    else this._interrupt('redo');
  }

  togglePause() {
    if (this.paused) {
      this.paused = false;
      this._needStart = true;
      this.markers.emit('resume', {});
      this._pause?.resolve();
    } else {
      this.paused = true;
      this.markers.emit('pause', {});
      this._interrupt('paused');
    }
    this.notify();
  }

  stop() {
    this.stopped = true;
    this._interrupt('stop');
    this._start?.resolve('stop');
    this._post?.resolve('stop');
    this._pause?.resolve();
    this._resp?.resolve(null);
  }

  tileClick(optionId, ts) {
    const t = this.current;
    if (!t) return;
    if (this.phase === 'response' && this._resp) this._resp.resolve({ optionId, ts });
    else if (this.phase === 'prompt' || this.phase === 'flash') {
      const o = t.options.find((x) => x.id === optionId);
      t.early_clicks.push({ t_ms: ts, option_id: optionId, phase: this.phase });
      this.markers.emit('early_click', { ...this._tf(t), option_id: optionId, category: o?.category, extra: { phase: this.phase } });
      this.notify();
    }
  }
}

export function newSessionRecord({ participant_id, device, notes, settings, stimuli, plannedTrials }) {
  const seed = settings.seed ?? (Math.random() * 2 ** 31) >>> 0;
  const stamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 14);
  const safe = (participant_id || 'anon').replace(/[^A-Za-z0-9_-]/g, '_');
  return {
    schema_version: SCHEMA_VERSION,
    session_id: `${safe}_${stamp}_${Math.random().toString(36).slice(2, 5)}`,
    participant_id, device, notes: notes || '', seed, settings: { ...settings, seed },
    stimuli_url: settings.stimuli_url, stimuli_item_count: stimuli.items.length,
    planned_trials: plannedTrials, next_trial_id: 0, status: 'created', created_iso: new Date().toISOString(),
    user_agent: navigator.userAgent, screen: { w: screen.width, h: screen.height, dpr: window.devicePixelRatio },
    time_origin_wall_ms: performance.timeOrigin,
  };
}

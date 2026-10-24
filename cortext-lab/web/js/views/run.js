// Run view: the participant stage (also the full-screen Demo view) + researcher dev panel.
import { h, isTyping, fmt, LETTERS } from '../util.js';
import { Predictor } from '../predictor.js';
import { SessionRunner } from '../session.js';
import { flashCounts, summarizeTrials, applyValidity } from '../trial.js';
import { INVALID_REASONS } from '../schema.js';

const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const noFocus = { tabindex: '-1', onmousedown: (e) => e.preventDefault() };

export function renderRun(root, app) {
  const pending = app.pending;
  if (!pending) { location.hash = '#/setup'; return; }
  app.pending = null;
  const { session, settings, stimuli, specs } = pending;
  const predictor = new Predictor({ mode: settings.predictor_mode, url: settings.predictor_url });

  // ---------- DOM ----------
  const promptEl = h('h1', { class: 'prompt' });
  const subEl = h('div', { class: 'subprompt' });
  const tilesEl = h('div', { class: 'tiles' });
  const progEl = h('span');
  const phaseEl = h('span');
  const postEl = h('div', { class: 'post' });
  const overlayEl = h('div', { class: 'overlay' });
  const pdEl = h('div', { class: `pd ${settings.photodiode_corner}${settings.photodiode ? ' enabled' : ''}` });
  const stage = h('section', { class: `stage${settings.show_bars ? ' show-bars' : ''}`, dataset: { phase: 'idle' } },
    h('div', { class: 'stage-top' }, progEl, phaseEl), promptEl, subEl, tilesEl, postEl, overlayEl);

  const devSections = {};
  const mkSec = (key, title) => { const body = h('div'); devSections[key] = body; return h('div', { class: 'card' }, h('h3', {}, title), body); };
  const ctrl = h('div', { class: 'ctrls' },
    h('button', { class: 'btn small', id: 'pauseBtn', ...noFocus, onclick: () => runner.togglePause() }, 'Pause ', h('kbd', {}, 'P')),
    h('button', { class: 'btn small', ...noFocus, onclick: () => app.toggleMode() }, 'Demo view ', h('kbd', {}, 'M')),
    h('button', { class: 'btn small', ...noFocus, onclick: () => runner.stop() }, 'End session'),
    h('a', { class: 'btn small', href: `#/review/${session.session_id}`, target: '_blank', rel: 'noopener' }, 'Review table ↗'));
  const dev = h('aside', { class: 'dev' },
    h('div', { class: 'card' }, h('h3', {}, 'Controls'), ctrl,
      h('div', { class: 'kbdhelp', style: { marginTop: '6px' } }, 'Space/Enter next · Backspace discard · R redo · 1–4 / A–D choose · P pause · M mode · Shift+F fullscreen · Shift+B bars')),
    mkSec('conn', 'Connections'), mkSec('sess', 'Session'), mkSec('count', 'Valid trials & repeats per class'),
    mkSec('timing', 'Timing'), mkSec('list', 'Trials'), mkSec('log', 'Marker log'));
  root.append(h('div', { class: 'run' }, stage, dev), pdEl);

  // ---------- stage API used by trial types ----------
  const tileEls = {};
  const barEls = {};
  const optLabel = (o) => `${o.letter} (${o.label})`;

  const ui = {
    renderTrial(trial) {
      tilesEl.replaceChildren();
      for (const k in tileEls) delete tileEls[k];
      tilesEl.style.setProperty('--cols', trial.options.length <= 4 ? 2 : 3);
      const showImg = trial.tile_content !== 'text';
      const showTxt = trial.tile_content !== 'image';
      for (const o of trial.options) {
        const tile = h('div', { class: 'tile', role: 'button', dataset: { id: o.id }, 'aria-label': `${o.letter}: ${o.label}` },
          showImg && o.image ? h('img', { src: o.image, alt: o.label, draggable: 'false' }) : null,
          h('div', { class: 'label' }, h('span', { class: 'letter' }, o.letter), showTxt ? h('span', {}, o.text) : null),
          h('div', { class: 'bar' }, h('i')));
        tile.addEventListener('pointerdown', (e) => { e.preventDefault(); runner.tileClick(o.id, e.timeStamp); });
        tileEls[o.id] = tile; barEls[o.id] = tile.querySelector('.bar > i');
        tilesEl.append(tile);
      }
      promptEl.textContent = trial.prompt;
      promptEl.classList.remove('show'); subEl.classList.remove('show');
      subEl.textContent = '';
      postEl.classList.remove('show');
      stage.dataset.phase = 'idle';
      progEl.textContent = `Question ${trial.trial_index + 1} / ${session.planned_trials}` + (trial.attempt ? ` · retry ${trial.attempt}` : '');
    },
    showPrompt() {
      promptEl.classList.add('show');
      subEl.textContent = 'Silently focus on the answer you think is correct. Do not click until the flashing stops.';
      subEl.classList.add('show');
      stage.dataset.phase = 'prompt';
    },
    setFlash(id, on) {
      const t = tileEls[id]; if (t) t.classList.toggle('flash', on);
      pdEl.classList.toggle('on', on);
    },
    clearFlash() { Object.values(tileEls).forEach((t) => t.classList.remove('flash')); pdEl.classList.remove('on'); },
    showResponseCue() {
      stage.dataset.phase = 'response';
      subEl.textContent = 'Now click the answer you chose (or press its key).';
      for (const id in barEls) barEls[id].style.width = '0%';
    },
    showPost(trial, r) {
      stage.dataset.phase = 'post';
      const byId = Object.fromEntries(trial.options.map((o) => [o.id, o]));
      Object.entries(tileEls).forEach(([id, el]) => {
        el.classList.remove('chosen', 'correct', 'wrong');
        if (id === trial.clicked_answer_id) el.classList.add('chosen');
        if (settings.feedback) {
          if (id === trial.true_answer_id) el.classList.add('correct');
          else if (id === trial.clicked_answer_id) el.classList.add('wrong');
        }
      });
      const chosen = byId[trial.clicked_answer_id];
      let result;
      if (!chosen) result = 'No answer was clicked.';
      else if (!settings.feedback) result = `Answer recorded: ${optLabel(chosen)}`;
      else result = trial.correct ? `✓ Correct — you chose ${optLabel(chosen)}` : `✗ You chose ${optLabel(chosen)}; the answer was ${optLabel(byId[trial.true_answer_id])}`;
      const chips = !trial.valid ? h('div', { class: 'chips' },
        h('span', { class: 'bad' }, `Trial marked INVALID (${trial.invalid_reason}). Reason:`),
        ...INVALID_REASONS.map(([v, lab]) => h('button', { class: 'btn small' + (trial.invalid_reason === v ? ' primary' : ''), ...noFocus, onclick: () => r.setReason(v) }, lab))) : null;
      postEl.replaceChildren(...[
        h('div', { class: 'result' }, result),
        h('div', { class: 'pred', id: 'predLine' }, predText(trial)),
        chips,
        h('div', { class: 'btns' },
          h('button', { class: 'btn danger big-discard', ...noFocus, onclick: () => r.discard() }, 'Discard trial', h('kbd', {}, '⌫')),
          h('button', { class: 'btn', ...noFocus, onclick: () => r.redo() }, 'Redo this question', h('kbd', {}, 'R')),
          h('button', { class: 'btn primary', ...noFocus, onclick: () => r.next() }, 'Next', h('kbd', {}, 'Space')))].filter(Boolean));
      postEl.classList.add('show');
    },
    hidePost() { postEl.classList.remove('show'); },
    updatePrediction(trial) {
      const el = postEl.querySelector('#predLine');
      if (el && runner.current === trial) el.textContent = predText(trial);
    },
    onFinished(r) { syncOverlay(); renderDev(true); },
  };

  function predText(trial) {
    const p = trial.prediction;
    if (settings.predictor_mode === 'off') return 'EEG indicates: — (prediction hook not connected)';
    if (!p) return 'EEG indicates: … (waiting for the predictor)';
    if (p.timeout || p.best_option_id == null) return 'EEG indicates: — (predictor sent no answer)';
    const o = trial.options.find((x) => x.id === p.best_option_id);
    const tag = p.source === 'mock' ? '  [MOCK — simulated, not evidence]' : '';
    return `EEG indicates: ${o ? optLabel(o) : p.best_option_id}, confidence ${fmt(p.confidence, 2)}${tag}`;
  }

  // ---------- runner ----------
  const runner = new SessionRunner({ session, settings, stimuli, specs, markers: app.markers, bridge: app.bridge, predictor, ui });
  app.runner = runner;

  predictor.onLive((live) => {
    if (!settings.show_bars) return;
    const now = performance.now();
    if (now - (predictor._lastBar || 0) < 250) return;
    predictor._lastBar = now;
    const vals = Object.values(live.p300);
    const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
    for (const [id, v] of Object.entries(live.p300)) if (barEls[id]) barEls[id].style.width = `${Math.round(((v - lo) / span) * 100)}%`;
  });

  function syncOverlay() {
    const ph = runner.phase;
    phaseEl.textContent = ph === 'idle' ? '' : ph;
    const show = (title, ...kids) => { overlayEl.replaceChildren(h('h1', {}, title), ...kids); overlayEl.classList.add('show'); };
    if (ph === 'ready') {
      const t = runner.current;
      show(`Ready for question ${t ? t.trial_index + 1 : ''} of ${session.planned_trials}`,
        h('div', { class: 'muted' }, 'Check the EEG signal and recorder, then start.'),
        h('button', { class: 'btn primary', ...noFocus, onclick: () => runner.next() }, 'Start ', h('kbd', {}, 'Space')));
    } else if (ph === 'paused') {
      show('Paused', h('button', { class: 'btn primary', ...noFocus, onclick: () => runner.togglePause() }, 'Resume ', h('kbd', {}, 'P')));
    } else if (ph === 'finished') {
      show(session.status === 'finished' ? 'Session complete' : 'Session ended',
        h('div', {}, h('a', { class: 'btn primary', href: `#/summary/${session.session_id}` }, 'Session summary'), ' ',
          h('a', { class: 'btn', href: `#/review/${session.session_id}` }, 'Review / clean trials')));
    } else overlayEl.classList.remove('show');
  }

  // ---------- dev panel (throttled; log frozen while flashing so DOM work cannot disturb frame timing) ----------
  let lastDev = 0, devTimer = null;
  const row = (k, v) => `<div class="row"><span class="muted">${k}</span><span>${v}</span></div>`;
  const dot = (c) => `<span class="dot ${c}"></span>`;
  function renderDev(force = false) {
    const now = performance.now();
    const flashing = runner.phase === 'flash';
    const minGap = flashing ? 500 : 250;
    if (!force && now - lastDev < minGap) {
      if (!devTimer) devTimer = setTimeout(() => { devTimer = null; renderDev(true); }, minGap);
      return;
    }
    lastDev = now;
    const b = app.bridge.status();
    const eeg = b.eeg || {};
    const eegOk = eeg.found && eeg.measured_srate > 0 && (eeg.age_ms == null || eeg.age_ms < 2000);
    const ps = predictor.status();
    devSections.conn.innerHTML =
      row('Bridge (WebSocket)', dot(b.connected ? 'green' : 'red') + (b.connected ? 'connected' : 'not connected')) +
      row('LSL marker outlet', dot(b.lsl_available ? 'green' : 'red') + (b.lsl_available ? esc(b.lsl_name) : 'off (CSV only)')) +
      row('LSL recorder attached', dot(b.lsl_consumers ? 'green' : 'red') + (b.lsl_consumers ? 'yes' : 'no consumer')) +
      row('EEG stream (LSL)', dot(eegOk ? 'green' : 'red') + (eeg.found ? `${esc(eeg.name)} · ${fmt(eeg.measured_srate, 0)} Hz` : 'not found')) +
      row('Predictor', dot(ps.mode === 'off' ? 'amber' : ps.connected ? 'green' : 'red') + `${esc(predictor.label)}`) +
      row('Clock offset / RTT', `${fmt(b.offset_ms, 1)} ms / ${fmt(b.rtt_ms, 2)} ms`) +
      row('Markers sent / bridge got', `${b.sent} / ${b.markers_received ?? '–'}`);
    const trials = runner.trials;
    const sum = summarizeTrials(trials);
    devSections.sess.innerHTML =
      row('Participant', esc(session.participant_id)) + row('Device', esc(session.device)) +
      row('Session', `<span class="mono">${esc(session.session_id)}</span>`) +
      row('Phase', esc(runner.phase) + (runner.paused ? ' (paused)' : '')) +
      row('Display refresh', `${fmt(session.refresh_hz_est, 1)} Hz`) +
      row('Early clicks (during flash)', sum.early_clicks);
    const fc = flashCounts(trials);
    const catRows = Object.entries(fc).map(([c, v]) => `<tr><td>${esc(c)}</td><td>${v.target}</td><td>${v.nontarget}</td></tr>`).join('');
    devSections.count.innerHTML =
      row('Valid / invalid / attempts', `<b class="good">${sum.valid}</b> / <b class="${sum.invalid ? 'bad' : ''}">${sum.invalid}</b> / ${sum.attempts}`) +
      `<table><tr><th>class</th><th>target flashes</th><th>non-target</th></tr>${catRows || '<tr><td colspan=3 class="muted">no flashes yet</td></tr>'}</table>`;
    const T = runner.timingSummary();
    devSections.timing.innerHTML = T.flashes
      ? row('Flashes measured', T.flashes) + row('Frame interval median / p95 / max', `${fmt(T.frame?.median_ms)} / ${fmt(T.frame?.p95_ms)} / ${fmt(T.frame?.max_ms)} ms`) +
        row('Dropped frames (total)', `<span class="${T.dropped ? 'warn' : ''}">${T.dropped}</span>`) +
        row('Onset error mean / p95 / max', `${fmt(T.onset_err_mean, 2)} / ${fmt(T.onset_err_p95, 2)} / ${fmt(T.onset_err_max, 2)} ms`) +
        row('Duration error mean / p95', `${fmt(T.dur_err_mean, 2)} / ${fmt(T.dur_err_p95, 2)} ms`)
      : '<span class="muted">Fills in after the first flash phase.</span>';
    const pb = document.getElementById('pauseBtn');
    if (pb) pb.firstChild.textContent = runner.paused ? 'Resume ' : 'Pause ';
    // trial list: skip re-render while a <select> in it is open/focused
    if (!devSections.list.contains(document.activeElement) || document.activeElement.tagName !== 'SELECT') renderTrialList();
    if (!flashing) {
      devSections.log.innerHTML = '<pre class="mono">' + esc(app.markers.recent.slice(-40).map(fmtMarker).join('\n')) + '</pre>';
      const pre = devSections.log.firstChild; pre.scrollTop = pre.scrollHeight;
    } else devSections.log.innerHTML = `<span class="muted">…flashing — log paused (${app.markers.seq} markers so far)</span>`;
  }
  const fmtMarker = (m) => `${String(m.seq).padStart(4)} ${m.event.padEnd(17)} t=${m.t_ms.toFixed(1).padStart(10)} tr${m.trial_id ?? '-'} ${m.option_id ?? ''} ${m.category ?? ''} ${m.is_target ? 'TARGET' : ''} ${m.repeat_index != null ? 'rep' + m.repeat_index : ''}${m.bridge_sent ? '' : ' (not sent)'}`;

  function renderTrialList() {
    const rows = runner.trials.slice().reverse().map((t) => {
      const sel = h('select', { onchange: (e) => setValidity(t, e.target.value) },
        h('option', { value: 'valid', selected: t.valid }, 'valid'),
        ...INVALID_REASONS.map(([v, lab]) => h('option', { value: v, selected: !t.valid && t.invalid_reason === v }, lab)),
        !t.valid && !INVALID_REASONS.some(([v]) => v === t.invalid_reason) ? h('option', { value: t.invalid_reason, selected: true }, t.invalid_reason) : null);
      return h('tr', { class: t.valid ? '' : 'invalid' },
        h('td', {}, `#${t.trial_id}`), h('td', {}, `q${t.trial_index + 1}${t.attempt ? '.' + (t.attempt + 1) : ''}`),
        h('td', {}, t.status + (t.correct == null ? '' : t.correct ? ' ✓' : ' ✗')), h('td', {}, sel));
    });
    devSections.list.replaceChildren(h('div', { class: 'triallist' }, h('table', {}, h('tr', {}, h('th', {}, 'id'), h('th', {}, 'q'), h('th', {}, 'status'), h('th', {}, 'validity')), ...rows)));
  }
  function setValidity(t, v) {
    if (v === 'valid') applyValidity(t, { valid: true, by: 'operator' });
    else applyValidity(t, { valid: false, reason: v, by: 'operator' });
    app.markers.emit('trial_invalid', { trial_id: t.trial_id, trial_index: t.trial_index, extra: { reason: v === 'valid' ? 'restored' : v } });
    runner.saveTrial(t);
    if (runner.current === t) ui.showPost(t, runner);
    renderDev(true);
  }

  const offRunner = runner.on(() => { syncOverlay(); renderDev(); });
  const offBridge = app.bridge.onChange(() => renderDev());
  const devInterval = setInterval(() => renderDev(), 1000);

  // ---------- keyboard ----------
  function onKey(e) {
    if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey || e.repeat) return;
    const k = e.key;
    const trial = runner.current;
    if (k === ' ' || k === 'Enter') { e.preventDefault(); runner.next(); }
    else if (k === 'Backspace') { e.preventDefault(); runner.discard(); }
    else if (k === 'r' || k === 'R') runner.redo();
    else if (k === 'p' || k === 'P') runner.togglePause();
    else if (k === 'm' || k === 'M') app.toggleMode();
    else if (k === 'F') { document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen?.(); }
    else if (k === 'B') { settings.show_bars = !settings.show_bars; stage.classList.toggle('show-bars', settings.show_bars); }
    else if (trial) {
      const idx = /^[1-8]$/.test(k) ? Number(k) - 1 : /^[a-hA-H]$/.test(k) ? LETTERS.indexOf(k.toUpperCase()) : -1;
      const opt = idx >= 0 ? trial.options.find((o) => o.position === idx) : null;
      if (opt) runner.tileClick(opt.id, e.timeStamp);
    }
  }
  document.addEventListener('keydown', onKey);
  const onUnload = (e) => { if (runner.phase !== 'finished') { e.preventDefault(); e.returnValue = ''; } };
  window.addEventListener('beforeunload', onUnload);

  syncOverlay();
  renderDev(true);
  runner.start().catch((e) => { console.error(e); overlayEl.replaceChildren(h('h1', {}, 'Error'), h('pre', {}, String(e.stack || e))); overlayEl.classList.add('show'); });

  return () => {
    document.removeEventListener('keydown', onKey);
    window.removeEventListener('beforeunload', onUnload);
    offRunner(); offBridge(); clearInterval(devInterval); clearTimeout(devTimer);
    if (runner.phase !== 'finished') runner.stop();
  };
}

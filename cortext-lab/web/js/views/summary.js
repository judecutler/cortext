import { h, fmt } from '../util.js';
import { loadSession, exportSessionZip, exportSingleFile } from '../exporter.js';
import { summarizeTrials, flashCounts } from '../trial.js';
import { buildDataset, pushDataset } from '../dataset.js';
import { deleteSessionsFlow } from '../sessionActions.js';
import { toast, confirmDialog } from '../ui.js';
import { launchFrom, markSuperseded } from '../launch.js';
import { record } from '../undo.js';
import { db } from '../db.js';

const pc = (x) => (x == null ? '–' : (100 * x).toFixed(0) + '%');

export async function renderSummary(root, app, sessionId) {
  const { session, trials, markers } = await loadSession(sessionId);
  if (!session) { root.append(h('p', { class: 'bad' }, `Unknown session ${sessionId}`)); return; }
  const S = summarizeTrials(trials);
  const fc = flashCounts(trials);
  const flashOn = markers.filter((m) => m.event === 'flash_on');
  const unsent = flashOn.filter((m) => !m.bridge_sent).length;
  const stat = (label, value, cls = '') => h('div', { class: 'card' }, h('div', { class: 'muted' }, label), h('div', { style: { fontSize: '28px', fontWeight: 700 }, class: cls }, value));

  const dsBody = h('div', {}, h('span', { class: 'muted' }, 'Cleaning the EEG and building the dataset entry…'));
  const dsLog = h('pre', { class: 'mono gitlog', style: { display: 'none' } });
  const pushBtn = h('button', { class: 'btn primary', disabled: true, onclick: doPush }, 'Upload to Hugging Face');
  const rebuildBtn = h('button', { class: 'btn', onclick: doBuild }, 'Rebuild');
  const dsCard = h('div', { class: 'card dataset-card' },
    h('h2', {}, 'Training dataset'),
    h('p', { class: 'muted small' }, 'The EEG of this session is cleaned automatically (drift + mains removal, 0.5–40 Hz, bad-electrode repair, artifact flags), cut into epochs on every flash and saved to data/cortext-p300/ on this computer. "Upload" sends that folder to the private Hugging Face dataset so teammates can download it and train (set up under Experiment → Lab & data).'),
    dsBody, h('div', { style: { marginTop: '10px', display: 'flex', gap: '8px' } }, pushBtn, rebuildBtn), dsLog);

  function showDs(r) {
    if (!r || r.ok === false) {
      dsBody.replaceChildren(h('div', { class: 'bad' }, 'Dataset not built: ' + (r?.error || 'bridge offline')),
        h('div', { class: 'muted small' }, 'Needs the bridge running with the EEG stream recorded during the session (and numpy + scipy).'));
      pushBtn.disabled = true;
      return;
    }
    const q = r.quality;
    dsBody.replaceChildren(
      h('div', { class: 'mono' }, r.relative_path + (r.simulated ? '   (SIMULATED: kept apart from real data)' : '')),
      h('div', { class: 'dsstats' },
        ...[['usable epochs', `${q.epochs_usable} / ${q.epochs_total}`], ['valid trials', q.trials_valid], ['rejected (artifacts)', q.epochs_rejected_artifact],
          ['bad electrodes', Object.keys(q.bad_channels).join(', ') || 'none'], ['mains removed', `${q.mains_hz} Hz`], ['duration', `${q.duration_s} s`]]
          .map(([k, v]) => h('div', {}, h('div', { class: 'muted small' }, k), h('b', {}, String(v))))),
      h('ul', { class: q.verdict.ok ? 'good' : 'warn' }, ...q.verdict.messages.map((m) => h('li', {}, m))));
    pushBtn.disabled = false;
  }
  let autoDone = false;
  async function doBuild() {
    dsBody.replaceChildren(h('span', { class: 'muted' }, 'Cleaning the EEG and building the dataset entry…'));
    let r;
    try { r = await buildDataset(sessionId); showDs(r); } catch (e) { showDs({ ok: false, error: e.message }); return; }
    // a redo marks the session it repeats as superseded; refresh that one's dataset copy too (now that no recording runs)
    if (session.redo_of) buildDataset(session.redo_of).catch(() => {});
    if (r?.ok && !r.simulated && !autoDone && localStorage.getItem('cortext-lab.autoupload') === '1') { autoDone = true; doPush(); }
  }
  async function doPush() {
    pushBtn.disabled = true; pushBtn.textContent = 'Uploading…';
    try {
      const r = await pushDataset(sessionId);
      dsLog.style.display = ''; dsLog.textContent = r.ok ? `✓ Uploaded to Hugging Face.\n${r.url}\nTeammates: python datasets/cortext-p300/cortext_store.py pull` : `✗ Not uploaded\n\n${r.error || ''}`;
    } catch (e) { dsLog.style.display = ''; dsLog.textContent = '✗ ' + e.message; }
    pushBtn.disabled = false; pushBtn.textContent = 'Upload to Hugging Face';
  }
  doBuild();

  // ---------- what next: next session / redo, and how this one relates to others ----------
  const running = app.runner && app.runner.phase !== 'finished';
  const pid = h('input', { type: 'text', value: session.participant_id, size: 8, 'aria-label': 'Participant for the next session',
    onkeydown: (e) => { if (e.key === 'Enter') { e.preventDefault(); next(); } } });
  async function next() {
    if (!pid.value.trim()) { toast('Enter a participant ID for the next session.', 'warn'); pid.focus(); return; }
    try { await launchFrom(app, sessionId, { mode: 'next', participant: pid.value }); } catch (e) { toast(e.message, 'bad'); }
  }
  async function redo() {
    const ok = await confirmDialog({ title: 'Redo this session?', confirmLabel: 'Redo session',
      body: h('div', {}, h('p', {}, `The same ${session.planned_trials} question(s) run again in the same order for participant ${session.participant_id}.`),
        h('p', { class: 'muted small' }, 'This recording is kept and marked “superseded”, so training leaves it out. You can unmark it here afterwards.')) });
    if (!ok) return;
    try { await launchFrom(app, sessionId, { mode: 'redo' }); } catch (e) { toast(e.message, 'bad'); }
  }
  const nextBar = h('div', { class: 'card nextbar' },
    h('div', { class: 'nextbar-main' },
      h('label', {}, 'Participant ', pid),
      h('button', { class: 'btn primary', disabled: running, onclick: next, title: 'Same settings and question list, new random order' }, 'Next session ▸'),
      h('button', { class: 'btn', disabled: running, onclick: redo, title: 'Same questions in the same order; this one is kept and marked superseded' }, '↺ Redo this session'),
      h('a', { class: 'btn', href: '#/experiment' }, 'Change settings')),
    running ? h('div', { class: 'warn small runwarn' }, 'A session is running in this window; finish it first.') : null);
  // a session that was just left is still saving: enable the buttons as soon as it has finished
  if (running) app.runner.finished.then(() => { nextBar.querySelectorAll('button').forEach((b) => { b.disabled = false; }); nextBar.querySelector('.runwarn')?.remove(); });
  const relBox = h('div');
  async function drawRelations() {
    const s = await db.get('sessions', sessionId);
    const link = (id) => h('a', { class: 'mono', href: `#/summary/${id}` }, id);
    relBox.replaceChildren(...[
      s?.superseded_by ? h('div', { class: 'note-banner warn-banner' }, 'Superseded: this session was redone as ', link(s.superseded_by), '. It is kept, but training leaves it out. ',
        h('button', { class: 'btn small', onclick: async () => {
          const by = s.superseded_by;
          await markSuperseded(sessionId, null, { rebuild: true }); drawRelations();
          record('Unmarked “superseded”', async () => { await markSuperseded(sessionId, by, { rebuild: true }); drawRelations(); }, async () => { await markSuperseded(sessionId, null, { rebuild: true }); drawRelations(); }, { kind: 'good' });
        } }, 'Use it after all')) : null,
      s?.redo_of ? h('p', { class: 'muted' }, 'This is a redo of ', link(s.redo_of), '.') : null,
      s?.previous_session_id ? h('p', { class: 'muted' }, 'Previous session: ', link(s.previous_session_id), '.') : null].filter(Boolean));
  }
  drawRelations();

  root.append(
    h('h1', {}, 'Session summary'),
    h('p', { class: 'muted' }, `${session.session_id} · participant ${session.participant_id} · ${session.device} · status ${session.status} · seed ${session.seed} · ${fmt(session.refresh_hz_est, 1)} Hz display`),
    session.notes ? h('p', {}, 'Notes: ' + session.notes) : null,
    relBox, nextBar,
    dsCard,
    h('div', { class: 'grid2' },
      stat('Attempts (incl. redos)', S.attempts), stat('Valid trials', S.valid, 'good'),
      stat('Discard rate', pc(S.discard_rate), S.discard_rate > 0.2 ? 'warn' : ''),
      stat('Click accuracy (valid)', pc(S.click_accuracy)), stat('Mean RT (ms)', fmt(S.mean_rt_ms, 0)),
      stat('Early clicks during flashing', S.early_clicks, S.early_clicks ? 'warn' : '')),
    h('div', { class: 'card' }, h('h2', {}, 'Details'),
      h('p', {}, 'Invalid reasons: ' + (Object.entries(S.invalid_reasons).map(([k, v]) => `${k} ×${v}`).join(', ') || 'none')),
      h('p', {}, 'Click accuracy by true category: ' + (Object.entries(S.by_category).map(([c, v]) => `${c} ${v.correct}/${v.n}`).join(' · ') || '–')),
      h('p', {}, 'Valid flashes per class (target / non-target): ' + (Object.entries(fc).map(([c, v]) => `${c} ${v.target}/${v.nontarget}`).join(' · ') || '–')),
      h('p', {}, `Markers stored: ${markers.length} (${flashOn.length} flash_on${unsent ? `, ${unsent} NOT delivered to the bridge — check the CSV/LSL log` : ', all delivered to the bridge'}).`),
      S.prediction_real ? h('p', {}, `EEG predictor vs ground truth (n=${S.prediction_real.n}): ${pc(S.prediction_real.accuracy_vs_truth)} — descriptive only; evaluate offline with held-out controls.`) : null,
      trials.some((t) => t.model_display) ? h('p', { class: 'warn' }, `The model's probabilities were shown before the click on ${trials.filter((t) => t.model_display).length} trial(s): those clicks may be anchored on the model, so use them as feedback (model_feedback.csv), not as independent ground truth.`) : null,
      S.prediction_mock ? h('p', { class: 'warn' }, `Mock / simulated-EEG predictions (n=${S.prediction_mock.n}) are simulated from the ground truth and are not a result.`) : null),
    h('div', {},
      h('button', { class: 'btn primary', onclick: () => exportSessionZip(sessionId) }, 'Export session (zip)'), ' ',
      ['session.json', 'trials.csv', 'markers.csv', 'model_feedback.csv'].map((f) => [h('button', { class: 'btn', onclick: () => exportSingleFile(sessionId, f) }, f), ' ']),
      h('a', { class: 'btn', href: `#/review/${sessionId}` }, 'Review / clean trials'), ' ',
      h('button', { class: 'btn danger', onclick: async () => { if ((await deleteSessionsFlow(app, [sessionId])).length) location.hash = '#/sessions'; } }, 'Delete session…')),
    h('div', { class: 'card', style: { marginTop: '14px' } }, h('h2', {}, 'Trials'),
      h('table', {}, h('thead', {}, h('tr', {}, ...['id', 'q', 'prompt', 'true', 'clicked', 'ok', 'RT ms', 'valid', 'reason'].map((c) => h('th', {}, c)))),
        h('tbody', {}, ...trials.map((t) => h('tr', { class: t.valid ? '' : 'invalid' },
          h('td', {}, `#${t.trial_id}`), h('td', {}, `q${t.trial_index + 1}${t.attempt ? '.' + (t.attempt + 1) : ''}`), h('td', {}, t.prompt),
          h('td', {}, t.true_answer_id), h('td', {}, t.clicked_answer_id ?? '–'),
          h('td', {}, t.correct == null ? '–' : t.correct ? '✓' : '✗'), h('td', {}, fmt(t.rt_ms, 0)),
          h('td', {}, t.valid ? 'yes' : 'no'), h('td', {}, t.invalid_reason || '')))))));
}

// Post-hoc review: mark any trial invalid (with a reason) or restore it. Trials are never deleted (a whole session can be, from here or the Sessions page).
import { h, fmt } from '../util.js';
import { db } from '../db.js';
import { applyValidity } from '../trial.js';
import { loadSession, exportSessionZip } from '../exporter.js';
import { INVALID_REASONS } from '../schema.js';
import { buildDataset } from '../dataset.js';
import { deleteSessionsFlow } from '../sessionActions.js';
import { record } from '../undo.js';

let rebuildTimer = null;

export async function renderReview(root, app, sessionId) {
  const { session, trials } = await loadSession(sessionId);
  if (!session) { root.append(h('p', { class: 'bad' }, `Unknown session ${sessionId}`)); return; }
  const live = app.runner && app.runner.session.session_id === sessionId && app.runner.phase !== 'finished';

  root.append(
    h('h1', {}, 'Review trials'),
    h('p', { class: 'muted' }, `Session ${session.session_id} · participant ${session.participant_id} · ${session.device} · ${trials.length} attempts. `,
      'Marking a trial invalid keeps all of its data (flash order, markers, click); only the flag changes.'),
    live ? h('p', { class: 'warn' }, 'This session is still running in this window. Use the dev panel there to change the CURRENT trial; changes made here to the trial in progress could be overwritten.') : null,
    h('div', {}, h('a', { class: 'btn', href: `#/summary/${sessionId}` }, 'Summary'), ' ',
      h('button', { class: 'btn', onclick: () => exportSessionZip(sessionId) }, 'Export session (zip)'), ' ',
      h('button', { class: 'btn danger', onclick: async () => { if ((await deleteSessionsFlow(app, [sessionId])).length) location.hash = '#/sessions'; } }, 'Delete session…')));

  const tbody = h('tbody');
  const note = (t) => h('input', { type: 'text', value: t.invalid_note || '', placeholder: 'note', style: { width: '150px' }, onchange: (e) => update(t, t.valid ? 'valid' : t.invalid_reason, e.target.value) });
  async function update(t, v, noteText, { undoable = true } = {}) {
    const before = { v: t.valid ? 'valid' : t.invalid_reason, note: t.invalid_note };
    if (v === 'valid') applyValidity(t, { valid: true, note: noteText ?? t.invalid_note, by: 'operator(review)' });
    else applyValidity(t, { valid: false, reason: v, note: noteText ?? t.invalid_note, by: 'operator(review)' });
    await db.put('trials', t);
    draw();
    // keep the shared dataset in sync with the reviewed validity flags
    clearTimeout(rebuildTimer);
    rebuildTimer = setTimeout(() => buildDataset(sessionId).catch(() => {}), 1500);
    if (undoable) {
      const after = { v, note: t.invalid_note };
      const label = noteText !== undefined && v === before.v ? `Edited the note on #${t.trial_id}` : `Marked #${t.trial_id} ${v === 'valid' ? 'valid' : `invalid (${v})`}`;
      record(label, () => update(t, before.v, before.note ?? '', { undoable: false }), () => update(t, after.v, after.note ?? '', { undoable: false }), { toast: noteText === undefined });
    }
  }
  function draw() {
    tbody.replaceChildren(...trials.map((t) => {
      const known = INVALID_REASONS.some(([v]) => v === t.invalid_reason);
      const sel = h('select', { onchange: (e) => update(t, e.target.value) },
        h('option', { value: 'valid', selected: t.valid }, 'valid'),
        ...INVALID_REASONS.map(([v, lab]) => h('option', { value: v, selected: !t.valid && t.invalid_reason === v }, lab)),
        !t.valid && !known ? h('option', { value: t.invalid_reason, selected: true }, t.invalid_reason) : null);
      const p = t.prediction;
      return h('tr', { class: t.valid ? '' : 'invalid' },
        h('td', {}, `#${t.trial_id}`), h('td', {}, `q${t.trial_index + 1}${t.attempt ? '.' + (t.attempt + 1) : ''}`),
        h('td', {}, t.prompt), h('td', {}, t.true_answer_id),
        h('td', {}, t.clicked_answer_id ?? '–'),
        h('td', {}, t.correct == null ? '–' : h('span', { class: t.correct ? 'good' : 'bad' }, t.correct ? '✓' : '✗')),
        h('td', {}, fmt(t.rt_ms, 0)), h('td', {}, (t.early_clicks || []).length || ''),
        h('td', {}, t.status), h('td', {}, p && p.best_option_id ? `${p.best_option_id}${p.source === 'mock' ? ' (mock)' : ''}` : ''),
        h('td', {}, sel), h('td', {}, note(t)));
    }));
  }
  draw();
  root.append(h('div', { class: 'card' }, h('table', {}, h('thead', {}, h('tr', {},
    ...['id', 'q', 'prompt', 'true', 'clicked', 'ok', 'RT ms', 'early clicks', 'status', 'EEG pred.', 'validity / reason', 'note'].map((c) => h('th', {}, c)))), tbody)));
}

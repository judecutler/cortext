import { h, fmt } from '../util.js';
import { loadSession, exportSessionZip, exportSingleFile } from '../exporter.js';
import { summarizeTrials, flashCounts } from '../trial.js';

const pc = (x) => (x == null ? '–' : (100 * x).toFixed(0) + '%');

export async function renderSummary(root, app, sessionId) {
  const { session, trials, markers } = await loadSession(sessionId);
  if (!session) { root.append(h('p', { class: 'bad' }, `Unknown session ${sessionId}`)); return; }
  const S = summarizeTrials(trials);
  const fc = flashCounts(trials);
  const flashOn = markers.filter((m) => m.event === 'flash_on');
  const unsent = flashOn.filter((m) => !m.bridge_sent).length;
  const stat = (label, value, cls = '') => h('div', { class: 'card' }, h('div', { class: 'muted' }, label), h('div', { style: { fontSize: '28px', fontWeight: 700 }, class: cls }, value));

  root.append(
    h('h1', {}, 'Session summary'),
    h('p', { class: 'muted' }, `${session.session_id} · participant ${session.participant_id} · ${session.device} · status ${session.status} · seed ${session.seed} · ${fmt(session.refresh_hz_est, 1)} Hz display`),
    session.notes ? h('p', {}, 'Notes: ' + session.notes) : null,
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
      S.prediction_mock ? h('p', { class: 'warn' }, `Mock predictor readouts (n=${S.prediction_mock.n}) are simulated from the ground truth and are not a result.`) : null),
    h('div', {},
      h('button', { class: 'btn primary', onclick: () => exportSessionZip(sessionId) }, 'Export session (zip)'), ' ',
      ['session.json', 'trials.csv', 'markers.csv', 'model_feedback.csv'].map((f) => [h('button', { class: 'btn', onclick: () => exportSingleFile(sessionId, f) }, f), ' ']),
      h('a', { class: 'btn', href: `#/review/${sessionId}` }, 'Review / clean trials')),
    h('div', { class: 'card', style: { marginTop: '14px' } }, h('h2', {}, 'Trials'),
      h('table', {}, h('thead', {}, h('tr', {}, ...['id', 'q', 'prompt', 'true', 'clicked', 'ok', 'RT ms', 'valid', 'reason'].map((c) => h('th', {}, c)))),
        h('tbody', {}, ...trials.map((t) => h('tr', { class: t.valid ? '' : 'invalid' },
          h('td', {}, `#${t.trial_id}`), h('td', {}, `q${t.trial_index + 1}${t.attempt ? '.' + (t.attempt + 1) : ''}`), h('td', {}, t.prompt),
          h('td', {}, t.true_answer_id), h('td', {}, t.clicked_answer_id ?? '–'),
          h('td', {}, t.correct == null ? '–' : t.correct ? '✓' : '✗'), h('td', {}, fmt(t.rt_ms, 0)),
          h('td', {}, t.valid ? 'yes' : 'no'), h('td', {}, t.invalid_reason || '')))))));
}

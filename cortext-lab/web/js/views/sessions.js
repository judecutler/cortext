import { h } from '../util.js';
import { db } from '../db.js';
import { exportSessionZip, exportAllZip } from '../exporter.js';
import { summarizeTrials } from '../trial.js';

export async function renderSessions(root) {
  const sessions = (await db.getAll('sessions')).sort((a, b) => (b.created_iso || '').localeCompare(a.created_iso || ''));
  const trials = await db.getAll('trials');
  const by = {};
  trials.forEach((t) => (by[t.session_id] ||= []).push(t));
  const persisted = navigator.storage && navigator.storage.persisted ? await navigator.storage.persisted() : null;
  root.append(
    h('h1', {}, 'Sessions'),
    h('p', { class: 'muted' }, 'Stored in this browser (IndexedDB)' + (persisted === false ? ' — not marked persistent; export regularly.' : persisted ? ' (persistent storage granted).' : '.')),
    h('div', { style: { marginBottom: '10px' } }, h('button', { class: 'btn primary', disabled: !sessions.length, onclick: exportAllZip }, 'Export ALL sessions (zip)')),
    h('div', { class: 'card' }, sessions.length
      ? h('table', {}, h('thead', {}, h('tr', {}, ...['session', 'participant', 'device', 'created', 'status', 'valid / attempts', ''].map((c) => h('th', {}, c)))),
        h('tbody', {}, ...sessions.map((s) => {
          const S = summarizeTrials(by[s.session_id] || []);
          return h('tr', {}, h('td', { class: 'mono' }, s.session_id), h('td', {}, s.participant_id), h('td', {}, s.device),
            h('td', {}, (s.created_iso || '').replace('T', ' ').slice(0, 16)), h('td', {}, s.status),
            h('td', {}, `${S.valid} / ${S.attempts}`),
            h('td', {}, h('a', { class: 'btn small', href: `#/summary/${s.session_id}` }, 'Summary'), ' ',
              h('a', { class: 'btn small', href: `#/review/${s.session_id}` }, 'Review'), ' ',
              h('button', { class: 'btn small', onclick: () => exportSessionZip(s.session_id) }, 'Export')));
        })))
      : h('p', { class: 'muted' }, 'No sessions yet.')));
}

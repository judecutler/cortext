// Deleting sessions for real: the browser copy (IndexedDB), the bridge machine's files (dataset folder, raw EEG and
// marker logs) and, only if the operator ticks the box, the copy on Hugging Face (including its history).
import { h } from './util.js';
import { db } from './db.js';
import { confirmDialog, toast, fmtBytes } from './ui.js';
import { bridgeSessions, deleteOnBridge, storageInfo } from './dataset.js';

const isRunning = (app, id) => !!(app.runner && app.runner.session.session_id === id && app.runner.phase !== 'finished');

/** Ask, then delete. Resolves to the list of session ids that were removed (empty if cancelled). */
export async function deleteSessionsFlow(app, ids) {
  ids = [...new Set(ids)];
  const running = ids.filter((id) => isRunning(app, id));
  if (running.length) { toast(`${running[0]} is still running in this window. End it first, then delete it.`, 'warn'); ids = ids.filter((id) => !running.includes(id)); }
  if (!ids.length) return [];

  let inv = [], bridgeUp = true, hf = null;
  try { inv = (await bridgeSessions()).sessions; } catch { bridgeUp = false; }
  try { hf = (await storageInfo(false)).hf; } catch { /* bridge offline */ }
  const known = new Map(inv.map((s) => [s.session_id, s]));
  const rows = [];
  let anyUploaded = false;
  for (const id of ids) {
    const [trials, markers] = await Promise.all([db.getAllBySession('trials', id), db.getAllBySession('markers', id)]);
    const b = known.get(id);
    anyUploaded ||= !!b?.uploaded;
    rows.push(h('li', {}, h('b', { class: 'mono' }, id), h('div', { class: 'muted small' },
      [`browser: ${trials.length} trial records, ${markers.length} markers`,
        b?.has_logs ? `raw logs: ${b.log_files} files, ${b.log_mb} MB` : null,
        b?.has_dataset ? `dataset folder: ${b.size_mb} MB` : null,
        b?.uploaded ? 'also on Hugging Face' : null].filter(Boolean).join(' · '))));
  }
  const body = h('div', {},
    h('p', {}, ids.length === 1 ? 'This permanently deletes the session from this browser and from this computer:' : `This permanently deletes ${ids.length} sessions from this browser and from this computer:`),
    h('ul', { class: 'dlg-list' }, ...rows),
    h('p', { class: 'warn small' }, 'There is no undo. Export first if you might want it back.'),
    bridgeUp ? null : h('p', { class: 'bad small' }, 'The bridge is not running, so only the browser copy can be deleted now. Files on the bridge computer will stay until you delete them again with the bridge running.'));
  const remoteOk = !!(hf && hf.configured && hf.has_token) && bridgeUp;
  const res = await confirmDialog({
    title: ids.length === 1 ? 'Delete this session?' : `Delete ${ids.length} sessions?`, body, danger: true,
    confirmLabel: 'Delete permanently',
    options: [{ key: 'remote', label: 'Also delete from Hugging Face', danger: true, disabled: !remoteOk,
      hint: remoteOk ? (anyUploaded ? 'Removes the folder and permanently erases its files from the repo history. Other people’s downloaded copies stay with them.' : 'These do not look uploaded from this computer; this checks the server anyway.')
        : 'Not available: Hugging Face is not set up on this computer (see Experiment → Lab & data).' }],
    typeToConfirm: (o) => (o.remote || ids.length >= 3 ? 'delete' : null),
  });
  if (!res) return [];

  const done = [], problems = [];
  for (const id of ids) {
    let report = null;
    if (bridgeUp) {
      try { report = await deleteOnBridge(id, { remote: res.options.remote }); } catch (e) { problems.push(`${id}: bridge unreachable (${e.message})`); }
      if (report && report.ok === false && !report.session_id) { problems.push(`${id}: ${report.error}`); continue; }   // refused (e.g. still running): keep everything
      if (report && report.errors?.length) problems.push(`${id}: ${report.errors.join('; ')}`);
    }
    await db.deleteSession(id);
    if (app.runner && app.runner.session.session_id === id) app.runner = null;
    done.push(id);
  }
  if (done.length) toast(`Deleted ${done.length === 1 ? done[0] : done.length + ' sessions'}${res.options.remote ? ' (including Hugging Face)' : ''}.`, 'good');
  if (problems.length) toast('Some parts could not be deleted: ' + problems.join(' | '), 'bad');
  return done;
}

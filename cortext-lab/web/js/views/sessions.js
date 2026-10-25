// Experiment → Recorded sessions: everything recorded, where each copy lives (this browser, the bridge's raw logs,
// the dataset folder, Hugging Face), bucket labels for organising test/train sets, export and real deletion.
import { h } from '../util.js';
import { db } from '../db.js';
import { exportSessionZip, exportAllZip } from '../exporter.js';
import { summarizeTrials } from '../trial.js';
import { bridgeSessions, setBucket } from '../dataset.js';
import { deleteSessionsFlow } from '../sessionActions.js';
import { subTabs, toast, fmtBytes } from '../ui.js';

const SUGGESTED_BUCKETS = ['train', 'test', 'pilot'];

export async function renderSessions(root, app) {
  const selected = new Set();
  const view = { text: '', bucket: '*', sim: true };
  let rows = [], bridgeUp = true;

  const list = h('div', { class: 'card' });
  const bulk = h('div', { class: 'bulkbar' });
  const buckets = h('datalist', { id: 'bucket-options' });
  const filterText = h('input', { type: 'search', placeholder: 'filter by id, participant, device…', oninput: (e) => { view.text = e.target.value.toLowerCase(); draw(); } });
  const filterBucket = h('select', { onchange: (e) => { view.bucket = e.target.value; draw(); } });
  const filterSim = h('input', { type: 'checkbox', checked: true, onchange: (e) => { view.sim = e.target.checked; draw(); } });
  const persisted = navigator.storage && navigator.storage.persisted ? await navigator.storage.persisted() : null;
  const note = h('p', { class: 'muted' });

  async function load() {
    const [sessions, trials] = await Promise.all([db.getAll('sessions'), db.getAll('trials')]);
    let inv = [];
    try { inv = (await bridgeSessions()).sessions; bridgeUp = true; } catch { bridgeUp = false; }
    const by = {};
    trials.forEach((t) => (by[t.session_id] ||= []).push(t));
    const map = new Map();
    for (const b of inv) map.set(b.session_id, { id: b.session_id, bridge: b, browser: null });
    for (const s of sessions) {
      const r = map.get(s.session_id) || { id: s.session_id, bridge: null, browser: null };
      r.browser = s; r.stats = summarizeTrials(by[s.session_id] || []);
      map.set(s.session_id, r);
    }
    rows = [...map.values()].map((r) => ({
      ...r,
      participant: r.browser?.participant_id ?? r.bridge?.participant_id ?? '–',
      device: r.browser?.device ?? r.bridge?.device ?? '–',
      when: r.browser?.created_iso || r.bridge?.recorded || '',
      simulated: r.bridge?.simulated ?? /simulated/i.test(r.browser?.device || ''),
      bucket: r.bridge?.bucket || '',
    })).sort((a, b) => String(b.when).localeCompare(String(a.when)));
    for (const id of [...selected]) if (!map.has(id)) selected.delete(id);
    const names = [...new Set([...SUGGESTED_BUCKETS, ...rows.map((r) => r.bucket).filter(Boolean)])];
    buckets.replaceChildren(...names.map((n) => h('option', { value: n })));
    filterBucket.replaceChildren(h('option', { value: '*' }, 'All buckets'), h('option', { value: '' }, 'No bucket'), ...names.map((n) => h('option', { value: n }, n)));
    filterBucket.value = view.bucket;
    note.textContent = 'Browser copies live in this browser (IndexedDB)' + (persisted === false ? ' — not marked persistent; export regularly' : persisted ? ' (persistent storage granted)' : '')
      + (bridgeUp ? '.' : '. The bridge is offline, so files on disk and Hugging Face status are not shown.');
    draw();
  }

  const visible = () => rows.filter((r) => (view.sim || !r.simulated) && (view.bucket === '*' || r.bucket === view.bucket)
    && (!view.text || [r.id, r.participant, r.device, r.bucket].join(' ').toLowerCase().includes(view.text)));
  const badge = (on, label, title) => h('span', { class: `loc ${on ? 'on' : ''}`, title }, label);

  function draw() {
    const vis = visible();
    const allBox = h('input', { type: 'checkbox', checked: vis.length > 0 && vis.every((r) => selected.has(r.id)), 'aria-label': 'Select all shown',
      onchange: (e) => { vis.forEach((r) => (e.target.checked ? selected.add(r.id) : selected.delete(r.id))); draw(); } });
    list.replaceChildren(vis.length ? h('table', {},
      h('thead', {}, h('tr', {}, h('th', {}, allBox), ...['session', 'participant', 'device', 'recorded', 'status', 'valid / attempts', 'bucket', 'copies', ''].map((c) => h('th', {}, c)))),
      h('tbody', {}, ...vis.map((r) => {
        const b = r.bridge;
        const sel = h('input', { type: 'checkbox', checked: selected.has(r.id), onchange: (e) => { e.target.checked ? selected.add(r.id) : selected.delete(r.id); drawBulk(); } });
        const bucket = h('input', { type: 'text', class: 'bucket-in', list: 'bucket-options', value: r.bucket, placeholder: b?.has_dataset ? '—' : 'n/a', disabled: !b?.has_dataset,
          title: b?.has_dataset ? 'Label this session, e.g. train or test' : 'Buckets apply once the session has been built into the dataset',
          onchange: async (e) => {
            const out = await setBucket(r.id, e.target.value).catch((err) => ({ ok: false, error: err.message }));
            if (out.ok) { toast(`${r.id} → ${out.bucket || 'no bucket'}${b.uploaded ? ' (upload again to share the change)' : ''}`, 'good'); load(); } else toast(out.error || 'could not set bucket', 'bad');
          } });
        return h('tr', { class: r.simulated ? 'sim' : '' },
          h('td', {}, sel),
          h('td', { class: 'mono' }, r.id, r.simulated ? h('span', { class: 'pill warn', style: { marginLeft: '6px' } }, 'SIM') : null,
            r.browser?.superseded_by ? h('span', { class: 'pill warn', style: { marginLeft: '6px' }, title: `Redone as ${r.browser.superseded_by}; kept, but training leaves it out` }, 'superseded') : null,
            r.browser?.redo_of ? h('span', { class: 'pill', style: { marginLeft: '6px' }, title: `Redo of ${r.browser.redo_of}` }, 'redo') : null, b?.running ? h('span', { class: 'pill good', style: { marginLeft: '6px' } }, 'running') : null),
          h('td', {}, r.participant), h('td', {}, r.device), h('td', {}, String(r.when).replace('T', ' ').slice(0, 16)),
          h('td', {}, r.browser?.status ?? '–'), h('td', {}, r.stats ? `${r.stats.valid} / ${r.stats.attempts}` : '–'),
          h('td', {}, bucket),
          h('td', { class: 'locs' },
            badge(!!r.browser, 'browser', 'Copy in this browser'),
            badge(!!b?.has_logs, 'logs', b?.has_logs ? `Raw EEG + marker logs on the bridge computer (${b.log_mb} MB)` : 'No raw logs on the bridge computer'),
            badge(!!b?.has_dataset, 'dataset', b?.has_dataset ? `Cleaned dataset folder (${b.size_mb} MB)` : 'Not built into the dataset'),
            badge(!!b?.uploaded, 'hub', b?.uploaded ? 'Uploaded to Hugging Face' : 'Not on Hugging Face from this computer')),
          h('td', { class: 'acts' },
            r.browser ? [h('a', { class: 'btn small', href: `#/summary/${r.id}` }, 'Summary'), ' ', h('a', { class: 'btn small', href: `#/review/${r.id}` }, 'Review'), ' ',
              h('button', { class: 'btn small', onclick: () => exportSessionZip(r.id) }, 'Export'), ' '] : null,
            h('button', { class: 'btn small danger', title: 'Permanently delete this session', onclick: () => remove([r.id]) }, 'Delete')));
      }))) : h('p', { class: 'muted' }, rows.length ? 'No sessions match the filter.' : 'No sessions yet. Start one from Configure & run.'));
    drawBulk();
  }

  function drawBulk() {
    const n = selected.size;
    const sims = rows.filter((r) => r.simulated).map((r) => r.id);
    const bucketIn = h('input', { type: 'text', class: 'bucket-in', list: 'bucket-options', placeholder: 'bucket name', disabled: !n });
    bulk.replaceChildren(
      h('span', { class: 'muted' }, n ? `${n} selected` : 'Select rows to act on several at once'),
      h('span', { class: 'bulk-bucket' }, bucketIn, h('button', { class: 'btn small', disabled: !n, onclick: async () => {
        let ok = 0, skipped = 0;
        for (const id of selected) { const r = rows.find((x) => x.id === id); if (!r?.bridge?.has_dataset) { skipped++; continue; } const o = await setBucket(id, bucketIn.value).catch(() => ({})); if (o.ok) ok++; }
        toast(`Bucket “${bucketIn.value || 'none'}” set on ${ok} session(s)${skipped ? `, ${skipped} skipped (not in the dataset yet)` : ''}.`, ok ? 'good' : 'warn'); load();
      } }, 'Put in bucket')),
      h('button', { class: 'btn small danger', disabled: !n, onclick: () => remove([...selected]) }, `Delete selected${n ? ` (${n})` : ''}`),
      h('button', { class: 'btn small danger', disabled: !sims.length, title: 'Sessions recorded with the simulated headset', onclick: () => remove(sims) }, `Delete all simulated${sims.length ? ` (${sims.length})` : ''}`));
  }

  async function remove(ids) {
    const done = await deleteSessionsFlow(app, ids);
    done.forEach((id) => selected.delete(id));
    if (done.length) load();
  }

  root.append(subTabs('sessions'), h('h1', {}, 'Recorded sessions'), note, buckets,
    h('div', { class: 'toolbar' }, filterText, filterBucket, h('label', { class: 'inline' }, filterSim, ' show simulated'),
      h('span', { class: 'spacer' }),
      h('button', { class: 'btn', onclick: load }, 'Refresh'),
      h('button', { class: 'btn primary', disabled: !rows.length, onclick: exportAllZip }, 'Export ALL sessions (zip)')),
    bulk, list);
  await load();
  root.querySelector('.toolbar .btn.primary').disabled = !rows.some((r) => r.browser);
}

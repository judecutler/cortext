// "Vibe meter" panel for the EEG tab. Strictly read-only and isolated:
//   - listens to the EEGFeed that the tab already polls (no extra requests, no bridge messages, no markers)
//   - writes nothing: no localStorage, no files, no dataset columns, never reaches the scorer or the recorder
//   - every update is wrapped so a bug here can never break the rest of the monitor
import { h } from './util.js';
import { STATES, StateTracker } from './statelogic.js';

const NOTE = {
  waiting: 'Waiting for signal…',
  contact: 'Need at least 2 electrodes with good contact.',
  moving: 'Movement or poor contact detected: holding the last values.',
  calibrating: 'Learning your usual levels…',
  ok: '',
};

/** Returns { el, detach }. `feed` is the tab's EEGFeed. */
export function createStatePanel(feed) {
  const tracker = new StateTracker();
  const note = h('div', { class: 'muted small vibe-note' }, NOTE.waiting);
  const rows = Object.fromEntries(STATES.map((s) => {
    const fill = h('i', {});
    const num = h('span', { class: 'vibe-num' }, '–');
    const el = h('div', { class: 'vibe-row', dataset: { state: s.key }, title: s.hint },
      h('span', { class: 'vibe-lab' }, s.label),
      h('span', { class: 'vibe-bar' }, fill, h('b', { class: 'vibe-mid' })),
      num,
      h('span', { class: 'vibe-word muted small' }, ''));
    return [s.key, { el, fill, num, word: el.lastChild, s }];
  }));
  const el = h('section', { class: 'panel vibe-panel', dataset: { panel: 'vibes' } },
    h('div', { class: 'panel-title' }, 'Vibe meter', h('span', { class: 'muted small' }, 'just for fun · not validated')),
    h('div', { class: 'vibes' }, ...STATES.map((s) => rows[s.key].el)),
    note,
    h('div', { class: 'muted small' }, '50 = your own recent average. Crude band-power ratios from the headset's electrodes; jaw tension and blinks can move them.'));

  let lastTrace = null;
  const off = feed.on(() => {
    try {
      const tr = feed.trace;
      if (!tr || tr === lastTrace) return;      // the feed also fires for ERP polls: only react to a new trace
      lastTrace = tr;
      const out = tracker.update(tr, Date.now());
      note.textContent = (tr.simulated && out.status === 'ok' ? 'Simulated EEG: these numbers are not a person. ' : '') + NOTE[out.status];
      el.classList.toggle('dim', out.status !== 'ok' && !out.scores);
      for (const { key } of STATES) {
        const r = rows[key], v = out.scores ? out.scores[key] : null;
        r.fill.style.width = v == null ? '0%' : `${v}%`;
        r.num.textContent = v == null ? '–' : String(v);
        r.word.textContent = v == null ? '' : v >= 65 ? r.s.hi : v <= 35 ? r.s.lo : 'about usual';
      }
    } catch (e) { console.error('vibe meter', e); }
  });
  return { el, detach: () => { try { off(); } catch { /* ignore */ } } };
}

// App shell: hash router, shared singletons (bridge, markers), Researcher/Demo mode toggle.
import { BridgeClient } from './bridgeClient.js';
import { MarkerBus } from './markers.js';
import { renderAgent } from './views/agent.js';
import { renderExperiment } from './views/experiment.js';
import { renderTiming } from './views/timing.js';
import { renderRun } from './views/run.js';
import { renderReview } from './views/review.js';
import { renderSummary } from './views/summary.js';
import { renderSessions } from './views/sessions.js';
import { renderEeg } from './views/eeg.js';
import { renderLibrary } from './views/library.js';
import { h, isTyping } from './util.js';
import { undo, redo, clearUndo, onUndoChange, canUndo, nextUndoLabel, UNDO_KEY } from './undo.js';

const bridge = new BridgeClient();
const markers = new MarkerBus(bridge);
export const app = {
  bridge, markers, runner: null, pending: null,
  mode: localStorage.getItem('cortext-lab.mode') || 'researcher',
  /** Re-render the current page (used when a new session starts while #/run is already showing). */
  rerender: () => route(true),
  toggleMode() { app.setMode(app.mode === 'demo' ? 'researcher' : 'demo'); },
  setMode(m) {
    app.mode = m;
    document.body.dataset.mode = m;
    localStorage.setItem('cortext-lab.mode', m);
    const b = document.getElementById('modeToggle');
    if (b) b.textContent = m === 'demo' ? 'Mode: Demo' : 'Mode: Researcher';
  },
};
window.cortext = app; // handy in the console

const view = document.getElementById('view');
let cleanup = null;
let currentHash = null;

function topStatus() {
  const s = bridge.status();
  const el = document.getElementById('topstatus');
  if (!el) return;
  const e = s.eeg || {};
  const eegTxt = !s.connected ? '' : e.found ? ` · <span class="dot ${e.simulated ? 'amber' : 'green'}"></span>EEG ${e.simulated ? 'SIMULATED' : (e.device || e.name)} ${e.good_channels != null ? `${e.good_channels}/${e.channels} good` : ''}` : ' · <span class="dot red"></span>no EEG';
  el.innerHTML = `<span class="dot ${s.connected ? 'green' : 'red'}"></span>bridge ${s.connected ? 'online' : 'offline'}` + eegTxt;
}

async function route(force = false) {
  // #/setup was merged into the Experiment tab (participant, headset, storage) and the Agent tab (model, voice)
  if (!location.hash || location.hash === '#/' || location.hash.startsWith('#/setup')) {
    history.replaceState(null, '', '#/experiment');
  }
  const hash = location.hash;
  if (!force && hash === currentHash && view.childElementCount) return;
  // Leaving a running session: confirm first.
  if (app.runner && app.runner.phase !== 'finished' && !hash.startsWith('#/run') && currentHash && currentHash.startsWith('#/run')) {
    if (!confirm('A session is running. Leave and end it? (Data recorded so far is kept.)')) {
      history.replaceState(null, '', currentHash);
      return;
    }
  }
  if (cleanup) { try { cleanup(); } catch (e) { console.error(e); } cleanup = null; }
  clearUndo();   // undo entries refer to the page that made them
  // nothing may pop up over the participant's screen: drop leftover notices when a session view opens
  if (hash.startsWith('#/run')) document.querySelectorAll('#toasts .toast:not(.bad)').forEach((t) => t.remove());
  currentHash = hash;
  view.replaceChildren();
  const [, name, arg] = hash.split('/');
  document.body.dataset.route = name || 'experiment';
  // sessions, run, review and summary are all part of the Experiment tab
  const section = { brain: 'eeg', sessions: 'experiment', library: 'experiment', run: 'experiment', review: 'experiment', summary: 'experiment' }[name] || name || 'experiment';
  document.querySelectorAll('#topbar nav a').forEach((a) => a.classList.toggle('active', a.dataset.nav === section));
  try {
    if (name === 'run') cleanup = renderRun(view, app);
    else if (name === 'review') await renderReview(view, app, decodeURIComponent(arg));
    else if (name === 'summary') await renderSummary(view, app, decodeURIComponent(arg));
    else if (name === 'sessions') await renderSessions(view, app);
    else if (name === 'library') cleanup = await renderLibrary(view, app);
    else if (name === 'eeg' || name === 'brain') cleanup = renderEeg(view, app);
    else if (name === 'timing') cleanup = renderTiming(view, app);
    else if (name === 'agent') cleanup = await renderAgent(view, app, arg ? decodeURIComponent(arg) : null);
    else cleanup = await renderExperiment(view, app);
  } catch (e) {
    console.error(e);
    view.replaceChildren(h('div', { class: 'card' }, h('h2', { class: 'bad' }, 'Something went wrong'), h('pre', { class: 'mono' }, String(e.stack || e))));
  }
}

document.getElementById('modeToggle').addEventListener('click', () => app.toggleMode());
const undoBtn = document.getElementById('undoBtn');
undoBtn.addEventListener('click', () => undo());
onUndoChange(() => {
  undoBtn.disabled = !canUndo();
  undoBtn.title = canUndo() ? `Undo: ${nextUndoLabel()} (${UNDO_KEY})` : 'Nothing to undo';
});
document.addEventListener('keydown', (e) => {
  // Undo / redo everywhere. Inside a text field the browser's own text undo wins.
  const mod = e.ctrlKey || e.metaKey;
  if (mod && !e.altKey && !isTyping(e)) {
    const k = e.key.toLowerCase();
    if (k === 'z' && !e.shiftKey) { e.preventDefault(); undo(); return; }
    if ((k === 'z' && e.shiftKey) || k === 'y') { e.preventDefault(); redo(); return; }
  }
});
document.addEventListener('keydown', (e) => {
  // Mode toggle outside the run view (the run view handles it itself).
  if (!location.hash.startsWith('#/run') && (e.key === 'm' || e.key === 'M') && !/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) app.toggleMode();
});
window.addEventListener('hashchange', () => route());
app.setMode(app.mode);
bridge.onChange(topStatus);
topStatus();
bridge.start();
route(true);

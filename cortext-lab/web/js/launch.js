// Starting sessions: the first one from the Experiment page, and "Next session" / "Redo this session" afterwards,
// from the end-of-session screen, the summary page or the run panel. All of them go through prepareLaunch(), so a
// session started any of these ways is built exactly the same way.
//
//   Next session  same settings and question list as the session just recorded, a NEW random seed (and a re-shuffle
//                 of the list if "Re-shuffle every time a session starts" is on). For the next run or the next person.
//   Redo session  the SAME questions in the SAME order (same seed / the exact list that was used). The earlier session
//                 is kept, not deleted, and marked superseded_by so training can leave it out.
import { loadStimuli, buildTrialSpecs, specsFromPlan, shuffled } from './stimuli.js';
import { getSet, materializeSet, stimuliSnapshot } from './stimulusSets.js';
import { newSessionRecord } from './session.js';
import { loadParticipant, saveParticipant } from './settings.js';
import { db } from './db.js';
import { buildDataset } from './dataset.js';

export const newSeed = () => (Math.random() * 2 ** 31) >>> 0;
const copyPlan = (plan) => (plan || []).map((q) => ({ prompt: q.prompt, target: q.target, options: q.options.slice() }));

/** Load the stimulus set a settings object points at (built-in/path or an uploaded set). */
export async function stimuliFor(settings) {
  const src = settings.stimuli_source || 'path';
  if (src.startsWith('set:')) {
    const set = await getSet(src.slice(4));
    if (!set) throw new Error(`The stimulus set “${src.slice(4)}” is no longer in this browser. Pick another one in Images & categories.`);
    return materializeSet(set);
  }
  const url = settings.stimuli_url || 'config/stimuli.json';
  const st = await loadStimuli(url);
  st.source = { type: 'path', url };
  return st;
}

/**
 * Turn settings (+ an optional question list) into everything the run view needs.
 *   settings   the experiment settings (copied, not changed)
 *   plan       the question list, used when settings.use_plan is on
 *   reshuffle  shuffle the list first (Next session / Start with "re-shuffle every time")
 *   seed       fixed seed (Redo) or null for a new one
 *   participant, device, notes, links   who is recorded and how this session relates to earlier ones
 * Returns { pending, plan } where plan is the list actually used (so the Experiment page can show it).
 */
export function prepareLaunch({ settings, stimuli, plan = [], reshuffle = false, seed = null, participant, device, notes = '', links = {} }) {
  const cur = { ...settings };
  cur.seed = seed ?? newSeed();
  let specs, used = null;
  if (cur.use_plan) {
    if (!plan.length) throw new Error('“Use this list” is on but the question list is empty.');
    used = reshuffle ? shuffled(plan).map((q) => ({ ...q, options: shuffled(q.options) })) : copyPlan(plan);
    specs = specsFromPlan(stimuli, used, { trialType: cur.trial_type, tileContent: cur.tile_content, fixedOrder: cur.order_mode === 'fixed' });
    cur.n_trials = specs.length;
    cur.question_plan = copyPlan(used);
  } else {
    delete cur.question_plan;
    specs = buildTrialSpecs(stimuli, { n: cur.n_trials, seed: cur.seed, optionsPerTrial: cur.options_per_trial, tileContent: cur.tile_content, trialType: cur.trial_type });
  }
  const session = newSessionRecord({ participant_id: participant, device, notes, settings: cur, stimuli, plannedTrials: specs.length, snapshot: stimuliSnapshot(stimuli), links });
  return { pending: { session, settings: cur, stimuli, specs }, plan: used };
}

/** Hand a prepared session to the run view (works even when the run view is already on screen). */
export function goRun(app, pending) {
  app.pending = pending;
  if (location.hash === '#/run') app.rerender(); else location.hash = '#/run';
}

/**
 * Start the next session or a redo, based on a recorded session.
 *   mode 'next' | 'redo'
 *   participant  for 'next' only (defaults to the Setup tab's participant); a redo keeps the original participant
 */
export async function launchFrom(app, sessionId, { mode = 'next', participant = null } = {}) {
  const prev = await db.get('sessions', sessionId);
  if (!prev) throw new Error(`Session ${sessionId} is not in this browser.`);
  const who = loadParticipant();
  const st = prev.settings || {};
  const stimuli = await stimuliFor(st);
  const redo = mode === 'redo';
  const pid = redo ? prev.participant_id : String(participant ?? who.participant ?? prev.participant_id ?? '').trim();
  if (!pid) throw new Error('Enter a participant ID first.');
  const device = redo ? prev.device : (who.device || prev.device);
  const { pending } = prepareLaunch({
    settings: { ...st, use_plan: !!st.question_plan },
    stimuli, plan: st.question_plan || [],
    reshuffle: !redo && !!st.shuffle_on_start,
    seed: redo ? st.seed : null,
    participant: pid, device,
    links: redo ? { redo_of: prev.session_id } : { previous_session_id: prev.session_id },
  });
  if (!redo && pid !== who.participant) saveParticipant({ ...who, participant: pid, notes: '' });
  if (redo) await markSuperseded(prev.session_id, pending.session.session_id);
  goRun(app, pending);
  return pending.session;
}

/**
 * Mark a session as replaced by a redo (kept, not deleted). Pass null to clear the mark.
 * rebuild: refresh the dataset copy's meta.json now. Off when a recording is about to start, because cleaning runs on
 * the bridge and must not compete with marker relaying; the new session's summary page rebuilds it afterwards.
 */
export async function markSuperseded(sessionId, byId, { rebuild = false } = {}) {
  const s = await db.get('sessions', sessionId);
  if (!s) return;
  if (byId) { s.superseded_by = byId; s.superseded_iso = new Date().toISOString(); } else { delete s.superseded_by; delete s.superseded_iso; }
  await db.put('sessions', s);
  // keep the dataset copy's meta.json in step (silently skipped when the bridge or the EEG is not there)
  if (rebuild) buildDataset(sessionId).catch(() => {});
}

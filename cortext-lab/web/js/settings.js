// Session settings: defaults, persistence of the last-used values, and a derived-duration helper.

export const DEFAULT_SETTINGS = {
  trial_type: 'mcq_flash',         // mcq_flash (options flash in place) | mcq_sequential (one at a time, then choose)
  stimuli_source: 'path',          // 'path' (stimuli_url below) or 'set:<name>' (an uploaded set)
  stimuli_url: 'config/stimuli.json',
  n_trials: 12,
  options_per_trial: 4,
  tile_content: 'image_text',      // image | text | image_text (what each stimulus shows, in both modes)
  prompt_ms: 2000,                 // question shown before anything else happens
  // --- flash mode ---
  repeats: 10,                     // flashes per option per trial
  on_ms: 500,
  off_min_ms: 100,
  off_max_ms: 200,
  // --- sequential mode ---
  seq_on_ms: 1000,                 // each option's media on screen alone
  seq_gap_ms: 300,                 // blank gap between options (question stays)
  seq_repeats: 1,                  // times each option is shown per trial
  show_model_probs: true,          // show the model's probability on every choice before the click
  send_feedback: false,            // after each trial send the answer back to the ws model server (opt-in)
  // --- common ---
  feedback: true,                  // show correct/incorrect after the click
  post_mode: 'manual',             // manual: operator presses Next; auto: advance after auto_advance_ms
  auto_advance_ms: 2500,
  response_timeout_ms: 0,          // 0 = wait indefinitely
  log_flash_off: true,             // also emit flash_off markers
  photodiode: false,
  photodiode_corner: 'bottom-left',
  predictor_mode: 'mock',          // off | mock | ws
  predictor_url: 'ws://localhost:8765',
  show_bars: false,                // live score bars under tiles (flash mode, demo only)
  use_plan: false,                 // true: run the question list from the Experiment tab instead of generating questions
  shuffle_on_start: false,         // reshuffle the question list (order and images) every time a session starts
  order_mode: 'random',            // random | fixed: in a custom list, shuffle option order per trial or keep the listed order
  // --- voice & audio (demo) ---
  speak_question: false,           // read the question aloud before the stimuli (the stimuli wait until it has finished)
  stim_modality: 'visual',         // visual | audio (spoken options, no visual change) | audiovisual (both together)
  narrate: 'off',                  // off | after_click (narrator announces the model's guess after the answer) | before_click
  narrator_llm: 'template',        // template (built in, offline) | ollama | openai | anthropic  (LLMs go through the bridge)
  narrator_mood: 'default',        // a NARRATOR_MOODS preset, or 'custom'
  narrator_context: '',            // the LLM's persona / mood / experiment context; '' = the chosen mood's text
  tts_engine: 'auto',              // auto: bridge voice (edge-tts / OS) if installed, else the browser's | browser
  tts_voice: '',                   // bridge voice name (edge-tts, e.g. en-US-AndrewMultilingualNeural); '' = default
  browser_voice: '',               // speechSynthesis voice name; '' = best English voice found
  tts_rate: 1,
  seed: null,
};

const LS_KEY = 'cortext-lab.settings.v1';

export function loadSettings() {
  try {
    const saved = JSON.parse(localStorage.getItem(LS_KEY) || '{}');
    return { ...DEFAULT_SETTINGS, ...saved, seed: null };
  } catch { return { ...DEFAULT_SETTINGS }; }
}
export function saveSettings(s) {
  try { localStorage.setItem(LS_KEY, JSON.stringify({ ...s, seed: null })); } catch { /* ignore */ }
}

/** The on/gap/repeat timing that applies to a given trial type (used for records and estimates). */
export function flashParamsFor(type, s) {
  return type === 'mcq_sequential'
    ? { on_ms: s.seq_on_ms, off_min_ms: s.seq_gap_ms, off_max_ms: s.seq_gap_ms, repeats: s.seq_repeats }
    : { on_ms: s.on_ms, off_min_ms: s.off_min_ms, off_max_ms: s.off_max_ms, repeats: s.repeats };
}

export function estimateDurations(s) {
  const p = flashParamsFor(s.trial_type, s);
  const flashes = s.options_per_trial * p.repeats;
  const avgOff = (p.off_min_ms + p.off_max_ms) / 2;
  const flashSec = (flashes * (p.on_ms + avgOff)) / 1000;
  const trialSec = flashSec + s.prompt_ms / 1000 + 6; // + ~6 s for response/review
  return { flashes, flashSec, trialSec, totalMin: (trialSec * s.n_trials) / 60 };
}

/** Save only some keys, keeping everything else as it is (Setup and Experiment each own part of the settings). */
export function patchSettings(partial) { saveSettings({ ...loadSettings(), ...partial }); }

const PARTICIPANT_KEY = 'cortext-lab.last-participant';
/** Who is being recorded and with what: filled in on the Setup tab, read when an experiment starts. */
// display_name is for the narrator only ("I'm 78% sure Jude saw the dog"). It stays in this browser and is never written
// to the session record, markers or dataset (the participant ID is what identifies a recording).
export function loadParticipant() {
  const empty = { participant: '', device: '', notes: '', display_name: '' };
  try { return { ...empty, ...JSON.parse(localStorage.getItem(PARTICIPANT_KEY) || '{}') }; } catch { return empty; }
}
export function saveParticipant(p) { try { localStorage.setItem(PARTICIPANT_KEY, JSON.stringify(p)); } catch { /* ignore */ } }

const PLAN_KEY = 'cortext-lab.plan.v1';
/** The custom question list being edited on the Experiment tab: [{ prompt, target, options: [item ids in display order] }]. */
export function loadPlan() { try { const p = JSON.parse(localStorage.getItem(PLAN_KEY) || '[]'); return Array.isArray(p) ? p : []; } catch { return []; } }
export function savePlan(plan) { try { localStorage.setItem(PLAN_KEY, JSON.stringify(plan)); } catch { /* ignore */ } }

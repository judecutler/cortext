// Session settings: defaults, persistence of the last-used values, and a derived-duration helper.

export const DEFAULT_SETTINGS = {
  trial_type: 'mcq_flash',
  stimuli_url: 'config/stimuli.json',
  n_trials: 12,
  options_per_trial: 4,
  tile_content: 'image_text',      // image | text | image_text
  repeats: 10,                     // flashes per option per trial
  on_ms: 500,
  off_min_ms: 100,
  off_max_ms: 200,
  prompt_ms: 2000,                 // prompt visible before flashing starts
  feedback: true,                  // show correct/incorrect after the click
  post_mode: 'manual',             // manual: operator presses Next; auto: advance after auto_advance_ms
  auto_advance_ms: 2500,
  response_timeout_ms: 0,          // 0 = wait indefinitely
  log_flash_off: true,             // also emit flash_off markers
  photodiode: false,
  photodiode_corner: 'bottom-left',
  predictor_mode: 'mock',          // off | mock | ws
  predictor_url: 'ws://localhost:8765',
  show_bars: false,                // live score bars under tiles (demo only; changes the display during flashing)
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

export function estimateDurations(s) {
  const flashes = s.options_per_trial * s.repeats;
  const avgOff = (s.off_min_ms + s.off_max_ms) / 2;
  const flashSec = (flashes * (s.on_ms + avgOff)) / 1000;
  const trialSec = flashSec + s.prompt_ms / 1000 + 6; // + ~6 s for response/review
  return { flashes, flashSec, trialSec, totalMin: (trialSec * s.n_trials) / 60 };
}

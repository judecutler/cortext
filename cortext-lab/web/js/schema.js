// Single source of truth for exported column orders and enumerations.

export const SCHEMA_VERSION = 1;

// One row per marker. flash_on rows are the ones to epoch the EEG on.
export const MARKER_COLUMNS = [
  'seq', 'session_id', 'event', 't_ms', 'frame_ms', 'wall_ms',
  'trial_id', 'trial_index', 'option_id', 'option_position', 'category', 'is_target',
  'repeat_index', 'flash_index', 'true_answer_id', 'bridge_sent', 'extra',
];

export const TRIAL_COLUMNS = [
  'session_id', 'trial_id', 'trial_index', 'attempt', 'type', 'status', 'valid', 'invalid_reason', 'invalid_note',
  'prompt', 'tile_content', 'option_ids_by_position', 'true_answer_id', 'true_answer_category',
  'clicked_answer_id', 'clicked_category', 'correct', 'rt_ms', 'early_clicks',
  'n_flashes', 'flash_order', 'flash_on_ms', 'flash_off_min_ms', 'flash_off_max_ms', 'repeats',
  'prediction_source', 'prediction_best_option_id', 'prediction_confidence', 'model_probabilities', 'model_shown_before_click',
  'started_wall_ms', 'ended_wall_ms', 'notes',
];

// One row per (trial attempt, option) with the model's probability next to the label and the human's click.
// This is the file to feed back into model training / calibration. Rows where model_shown_before_click is true have a
// click that may have been influenced by the displayed probabilities.
export const FEEDBACK_COLUMNS = [
  'session_id', 'trial_id', 'trial_index', 'valid', 'invalid_reason', 'presentation_mode',
  'option_id', 'option_position', 'category', 'model_probability', 'model_score', 'model_source', 'model_probability_calibrated',
  'is_true_answer', 'is_clicked', 'is_model_top_choice', 'true_answer_id', 'clicked_answer_id', 'model_top_option_id',
  'human_agrees_with_model', 'model_shown_before_click',
];

export const INVALID_REASONS = [
  ['misclick', 'Misclick'],
  ['blink_or_movement', 'Blink / movement'],
  ['distracted', 'Distracted'],
  ['equipment_glitch', 'Equipment glitch'],
  ['other', 'Other'],
];
// Reasons set automatically by the app (not in the manual dropdown).
export const AUTO_REASONS = ['unspecified', 'redo', 'aborted', 'paused', 'tab_hidden', 'no_response', 'early_click'];

export const DEVICES = ['g.tec Unicorn / BCI Core-8', 'Simulated EEG (fake_eeg.py)', 'Muse', '8-node EEG', '32-node EEG', 'Other'];

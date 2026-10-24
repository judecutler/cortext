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
  'prediction_source', 'prediction_best_option_id', 'prediction_confidence',
  'started_wall_ms', 'ended_wall_ms', 'notes',
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

export const DEVICES = ['Muse', '8-node EEG', '32-node EEG', 'Other'];

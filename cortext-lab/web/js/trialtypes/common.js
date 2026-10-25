// Shared pieces of the MCQ trial types: the prompt phase (with optional read-aloud), stimulus onsets in the chosen
// modality (visual / audio / both), the response phase (cue -> wait for click -> record), and the narrator.
import { recordClick, applyValidity } from '../trial.js';
import { sleep } from '../util.js';
import { narrationFacts, composeNarration, NARRATOR_MOODS } from '../voice.js';

const PREP_TIMEOUT_MS = 15000;

/**
 * Prompt phase. Shows the question; if speak_question is on, reads it aloud and the stimuli wait until it has finished
 * (+400 ms), so the voice never overlaps the EEG window. For audio modalities it also decodes this trial's sounds.
 */
export async function promptPhase(ctx, trial) {
  const { settings: s, ui, voice } = ctx;
  await ctx.frame((ts) => { ui.showPrompt(); ctx.mark('prompt_on', {}, ts); });
  const audioStim = voice && (s.stim_modality === 'audio' || s.stim_modality === 'audiovisual');
  const prep = audioStim ? Promise.race([voice.prepareStimuli(trial.options), sleep(PREP_TIMEOUT_MS).then(() => ({ ready: 0, missing: trial.options.length, timeout: true }))]) : null;
  const speak = voice && s.speak_question ? (async () => {
    ctx.mark('question_audio_start', {});
    const r = await voice.speak(trial.prompt);
    ctx.mark('question_audio_end', { extra: { engine: r.engine } });
    await sleep(400);   // let the auditory response to the voice settle before the first stimulus
  })() : null;
  await ctx.guard(Promise.all([sleep(s.prompt_ms), speak, prep].filter(Boolean)));
  if (prep) {
    const r = await prep;
    trial.audio_stimuli = { modality: s.stim_modality, ready: r.ready, missing: r.missing, output_latency_ms: voice.outputLatencyMs,
      timing: r.missing ? 'approximate (browser speech for some options)' : 'buffer' };
  }
}

/** One stimulus onset, inside the onset frame. Returns the marker extra (modality and audio timing). */
export function stimulusOn(ctx, o, ts) {
  const { settings: s, ui, voice } = ctx;
  const mod = s.stim_modality || 'visual';
  ui.setFlash(o.id, true, { visual: mod !== 'audio' });
  if (mod === 'visual' || !voice) return { modality: 'visual' };
  return { modality: mod, ...voice.playStim(o, ts) };
}
export function stimulusOff(ctx, o) {
  ctx.ui.setFlash(o.id, false, { visual: (ctx.settings.stim_modality || 'visual') !== 'audio' });
  // audio is not cut at "off": a word plays to its end (the next onset stops it)
}

/** The LLM context for the narrator: the edited text, else the chosen mood's preset. */
export function narratorContext(s) {
  const own = (s.narrator_context || '').trim();
  if (own) return own;
  return (NARRATOR_MOODS.find(([id]) => id === s.narrator_mood) || NARRATOR_MOODS[0])[2];
}

/**
 * The narrator announces the model's guess (text from the LLM or the template, then spoken). Runs AFTER the EEG phase.
 * `when` is 'before_click' or 'after_click' and must match settings.narrate. The display name is replaced by {name} in
 * what is stored, so it never reaches the dataset.
 */
export async function narrateGuess(ctx, trial, pred, when) {
  const { settings: s, voice, ui } = ctx;
  if (!voice || s.narrate !== when) return null;
  const name = (ctx.displayName || '').trim();
  const facts = narrationFacts({ trial, pred, name, when, modality: s.stim_modality || 'visual' });
  ui.showNarration?.(null, { thinking: true });
  const n = await composeNarration(facts, { provider: s.narrator_llm, context: narratorContext(s), plain: s.narrator_mood === 'scientist' });
  if (ctx.runner?.current && ctx.runner.current !== trial) return null;   // operator moved on while the LLM was thinking
  const scrub = (t) => (name ? t.split(name).join('{name}') : t);
  trial.narration = { when, text: scrub(n.text), provider: n.provider, model: n.model || null, llm_ms: n.ms, fallback_error: n.fallback_error || null,
    mood: s.narrator_mood, custom_context: !!(s.narrator_context || '').trim(), simulated: !!facts?.simulated, at_wall_ms: Date.now() };
  ctx.save?.();
  ui.showNarration?.(n.text, { simulated: !!facts?.simulated, provider: n.provider });
  ctx.mark('narration_start', { extra: { when, provider: n.provider } });
  const r = await voice.speak(n.text);
  ctx.mark('narration_end', { extra: { engine: r.engine } });
  return trial.narration;
}

/** Marker fields describing one option in a flash/stimulus event. */
export const optFields = (o, ev) => ({
  option_id: o.id, option_position: o.position, category: o.category, is_target: o.is_target,
  repeat_index: ev.repeat_index, flash_index: ev.flash_index,
});

/**
 * Response phase. Frozen tiles + cue frame, then wait for the click.
 * onCue(ts) runs INSIDE the cue frame (reveal tiles, show model probabilities, ...); extraCue goes into the marker.
 * Returns when the trial is complete (or invalid by timeout). Throws InterruptError through ctx.guard.
 */
export async function responsePhase(ctx, trial, { onCue, extraCue, afterCue } = {}) {
  const { ui, settings: s } = ctx;
  const opt = Object.fromEntries(trial.options.map((o) => [o.id, o]));
  trial.status = 'awaiting_response';
  const cue = await ctx.frame((ts) => {
    ctx.setPhase('response');
    ui.showResponseCue();
    if (onCue) onCue(ts);
    ctx.mark('response_cue', { extra: extraCue || null }, ts);
  });
  if (afterCue) Promise.resolve().then(afterCue).catch((e) => console.error(e));   // e.g. the narrator; never blocks the click
  const resp = await ctx.guard(ctx.waitResponse(s.response_timeout_ms));
  if (!resp) {
    trial.status = 'complete';
    trial.ended_wall_ms = Date.now();
    applyValidity(trial, { valid: false, reason: 'no_response', by: 'auto' });
    ctx.mark('no_response', {});
    return;
  }
  const o = opt[resp.optionId];
  recordClick(trial, resp.optionId, resp.ts - cue);
  ctx.mark('click', { option_id: o.id, option_position: o.position, category: o.category, is_target: o.is_target,
    extra: { rt_ms: trial.rt_ms, model_shown_before_click: !!trial.model_display } });
  trial.status = 'complete';
  trial.ended_wall_ms = Date.now();
  ctx.mark('trial_end', { extra: { correct: trial.correct } });
}

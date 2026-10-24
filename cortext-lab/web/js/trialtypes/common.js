// Shared pieces of the MCQ trial types: the response phase (cue -> wait for click -> record) and flash-marker fields.
import { recordClick, applyValidity } from '../trial.js';

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
export async function responsePhase(ctx, trial, { onCue, extraCue } = {}) {
  const { ui, settings: s } = ctx;
  const opt = Object.fromEntries(trial.options.map((o) => [o.id, o]));
  trial.status = 'awaiting_response';
  const cue = await ctx.frame((ts) => {
    ctx.setPhase('response');
    ui.showResponseCue();
    if (onCue) onCue(ts);
    ctx.mark('response_cue', { extra: extraCue || null }, ts);
  });
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

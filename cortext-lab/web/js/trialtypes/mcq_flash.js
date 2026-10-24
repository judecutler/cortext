// mcq_flash: prompt -> randomized option flashes -> click to declare answer -> (post phase handled by SessionRunner).
import { sleep } from '../util.js';
import { makeFlashPlan, runFlashPlan } from '../flasher.js';
import { recordClick, applyValidity } from '../trial.js';

export const mcqFlash = {
  id: 'mcq_flash',
  label: 'Multiple choice, flashing options (P300)',
  implemented: true,

  async run(ctx, trial) {
    const { settings: s, ui, predictor } = ctx;
    const opt = Object.fromEntries(trial.options.map((o) => [o.id, o]));
    trial.started_wall_ms = Date.now();
    trial.status = 'prompt';
    predictor.startTrial({
      session_id: trial.session_id, trial_id: trial.trial_id, options: trial.options, truth_for_mock: trial.true_answer_id,
    });
    ctx.mark('trial_start', { extra: { attempt: trial.attempt } });

    // Tab hidden => rAF stops => flash timing would be garbage. Abort and redo automatically.
    const onVis = () => { if (document.hidden) ctx.interrupt('tab_hidden'); };
    document.addEventListener('visibilitychange', onVis);
    try {
      // ---- 1. prompt ----
      ctx.setPhase('prompt');
      await ctx.frame((ts) => { ui.showPrompt(); ctx.mark('prompt_on', {}, ts); });
      await ctx.guard(sleep(s.prompt_ms));

      // ---- 2. flash ----
      const plan = makeFlashPlan({
        optionIds: trial.options.map((o) => o.id), repeats: s.repeats, onMs: s.on_ms,
        offMinMs: s.off_min_ms, offMaxMs: s.off_max_ms, rng: ctx.rng,
      });
      trial.status = 'flashing';
      ctx.setPhase('flash');
      ctx.bridge.quiet = true;
      await ctx.save();
      ctx.mark('flash_phase_start', { extra: { n_flashes: plan.events.length, planned_ms: plan.total_ms } });

      const result = await ctx.guard(runFlashPlan(plan, {
        token: ctx.token,
        onOn: (ev, ts, res) => {
          const o = opt[ev.option_id];
          ui.setFlash(ev.option_id, true);
          const m = ctx.mark('flash_on', {
            option_id: o.id, option_position: o.position, category: o.category, is_target: o.is_target,
            repeat_index: ev.repeat_index, flash_index: ev.flash_index,
          }, ts);
          res.t_on_ms = m.t_ms; res.marker_seq = m.seq;
          trial.flash_order.push(res);
          predictor.noteFlash(m);
        },
        onOff: (ev, ts, aborted) => {
          ui.setFlash(ev.option_id, false);
          if (s.log_flash_off && !aborted) {
            const o = opt[ev.option_id];
            ctx.mark('flash_off', { option_id: o.id, option_position: o.position, category: o.category, is_target: o.is_target,
              repeat_index: ev.repeat_index, flash_index: ev.flash_index }, ts);
          }
        },
      }));
      ctx.bridge.quiet = false;
      ctx.mark('flash_phase_end', {});
      trial.flash_timing = result.frame;
      ctx.reportTiming(result);
      predictor.finishTrial().then((p) => {
        if (p) { trial.prediction = p; ctx.save(); ui.updatePrediction?.(trial); }
      });

      // ---- 3. response ----
      trial.status = 'awaiting_response';
      let cue;
      cue = await ctx.frame((ts) => { ctx.setPhase('response'); ui.showResponseCue(); ctx.mark('response_cue', {}, ts); });
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
      ctx.mark('click', { option_id: o.id, option_position: o.position, category: o.category, is_target: o.is_target, extra: { rt_ms: trial.rt_ms } });
      trial.status = 'complete';
      trial.ended_wall_ms = Date.now();
      ctx.mark('trial_end', { extra: { correct: trial.correct } });
    } finally {
      document.removeEventListener('visibilitychange', onVis);
    }
  },
};

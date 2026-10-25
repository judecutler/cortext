// mcq_flash: prompt -> [optional imagery window] -> randomized option flashes (in place) -> click to declare answer -> (post phase in SessionRunner).
import { sleep } from '../util.js';
import { makeFlashPlan, runFlashPlan } from '../flasher.js';
import { optFields, responsePhase, imageryPhase } from './common.js';

export const mcqFlash = {
  id: 'mcq_flash',
  label: 'Flash options in place, then click (original)',
  implemented: true,

  async run(ctx, trial) {
    const { settings: s, ui, predictor } = ctx;
    const opt = Object.fromEntries(trial.options.map((o) => [o.id, o]));
    trial.started_wall_ms = Date.now();
    trial.status = 'prompt';
    predictor.startTrial({
      session_id: trial.session_id, trial_id: trial.trial_id, options: trial.options, truth_for_mock: trial.true_answer_id,
    });
    ctx.mark('trial_start', { extra: { attempt: trial.attempt, mode: 'flash' } });

    // Tab hidden => rAF stops => flash timing would be garbage. Abort and redo automatically.
    const onVis = () => { if (document.hidden) ctx.interrupt('tab_hidden'); };
    document.addEventListener('visibilitychange', onVis);
    try {
      // ---- 1. prompt ----
      ctx.setPhase('prompt');
      await ctx.frame((ts) => { ui.showPrompt(); ctx.mark('prompt_on', {}, ts); });
      await ctx.guard(sleep(s.prompt_ms));
      await imageryPhase(ctx, trial);   // no-op unless imagery_ms > 0

      // ---- 2. flash ----
      const plan = makeFlashPlan({
        optionIds: trial.options.map((o) => o.id), repeats: s.repeats, onMs: s.on_ms,
        offMinMs: s.off_min_ms, offMaxMs: s.off_max_ms, rng: ctx.rng, fixedOrder: !!trial.fixed_order,
      });
      trial.status = 'flashing';
      ctx.setPhase('flash');
      ctx.bridge.quiet = true;
      await ctx.save();
      ctx.mark('flash_phase_start', { extra: { n_flashes: plan.events.length, planned_ms: plan.total_ms, mode: 'flash' } });

      const result = await ctx.guard(runFlashPlan(plan, {
        token: ctx.token,
        onOn: (ev, ts, res) => {
          const o = opt[ev.option_id];
          ui.setFlash(ev.option_id, true);
          const m = ctx.mark('flash_on', optFields(o, ev), ts);
          res.t_on_ms = m.t_ms; res.marker_seq = m.seq;
          trial.flash_order.push(res);
          predictor.noteFlash(m);
        },
        onOff: (ev, ts, aborted) => {
          ui.setFlash(ev.option_id, false);
          if (s.log_flash_off && !aborted) ctx.mark('flash_off', optFields(opt[ev.option_id], ev), ts);
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
      await responsePhase(ctx, trial);
    } finally {
      document.removeEventListener('visibilitychange', onVis);
    }
  },
};

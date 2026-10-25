// mcq_sequential: the question stays on screen; each option's media appears ALONE for ~1 s, one after another;
// then the full multiple-choice grid appears, every choice labelled with the model's probability; the participant clicks
// the answer they think is correct. The EEG is recorded during the one-at-a-time phase, BEFORE the model's numbers are
// shown, so the probabilities cannot influence the brain signal (they can influence the click: see model_display).
//
// Markers use the same schema as flash mode (event `flash_on` / `flash_off`, extra.mode = "sequential") so the same
// epoching and detector code works for both modes.
import { makeFlashPlan, runFlashPlan } from '../flasher.js';
import { optFields, responsePhase, promptPhase, stimulusOn, stimulusOff, narrateGuess } from './common.js';

export const mcqSequential = {
  id: 'mcq_sequential',
  label: 'One at a time, then choose (with model probabilities)',
  implemented: true,

  async run(ctx, trial) {
    const { settings: s, ui, predictor } = ctx;
    const opt = Object.fromEntries(trial.options.map((o) => [o.id, o]));
    trial.started_wall_ms = Date.now();
    trial.status = 'prompt';
    predictor.startTrial({
      session_id: trial.session_id, trial_id: trial.trial_id, options: trial.options, truth_for_mock: trial.true_answer_id,
    });
    ctx.mark('trial_start', { extra: { attempt: trial.attempt, mode: 'sequential' } });

    const onVis = () => { if (document.hidden) ctx.interrupt('tab_hidden'); };
    document.addEventListener('visibilitychange', onVis);
    try {
      // ---- 1. question alone ----
      ctx.setPhase('prompt');
      await promptPhase(ctx, trial);

      // ---- 2. options one at a time (question stays) ----
      const plan = makeFlashPlan({
        optionIds: trial.options.map((o) => o.id), repeats: s.seq_repeats, onMs: s.seq_on_ms,
        offMinMs: s.seq_gap_ms, offMaxMs: s.seq_gap_ms, rng: ctx.rng, fixedOrder: !!trial.fixed_order,
      });
      trial.status = 'flashing';
      ctx.setPhase('flash');
      ctx.bridge.quiet = true;
      await ctx.save();
      ctx.mark('flash_phase_start', { extra: { n_flashes: plan.events.length, planned_ms: plan.total_ms, mode: 'sequential' } });

      const result = await ctx.guard(runFlashPlan(plan, {
        token: ctx.token,
        onOn: (ev, ts, res) => {
          const o = opt[ev.option_id];
          const extra = stimulusOn(ctx, o, ts);         // sequential layout: shows this option's media alone (and/or plays it)
          const m = ctx.mark('flash_on', { ...optFields(o, ev), extra: { mode: 'sequential', on_ms: s.seq_on_ms, ...extra } }, ts);
          res.t_on_ms = m.t_ms; res.marker_seq = m.seq;
          trial.flash_order.push(res);
          predictor.noteFlash(m);
        },
        onOff: (ev, ts, aborted) => {
          stimulusOff(ctx, opt[ev.option_id]);
          if (s.log_flash_off && !aborted) ctx.mark('flash_off', { ...optFields(opt[ev.option_id], ev), extra: { mode: 'sequential' } }, ts);
        },
      }));
      ctx.bridge.quiet = false;
      ctx.mark('flash_phase_end', {});
      trial.flash_timing = result.frame;
      ctx.reportTiming(result);

      // ---- 3. model scores the epochs (EEG is finished; nothing is shown yet) ----
      ui.setSub('Scoring…');
      const predP = predictor.finishTrial();
      ctx.runner._predP = predP;
      const pred = await ctx.guard(predP);
      if (pred) { trial.prediction = pred; await ctx.save(); }
      const narrateFirst = s.narrate === 'before_click';

      // ---- 4. full MCQ with the model's probability on every choice, then the click ----
      const showProbs = s.show_model_probs && pred && pred.probabilities;
      await responsePhase(ctx, trial, {
        extraCue: { model_probs_shown: !!showProbs, narrated: narrateFirst },
        afterCue: narrateFirst ? () => narrateGuess(ctx, trial, pred, 'before_click') : null,
        onCue: () => {
          ui.revealTiles();
          if (showProbs) {
            trial.model_display = {
              shown_wall_ms: Date.now(), source: pred.source, calibrated: !!pred.probabilities_calibrated,
              probabilities: pred.probabilities, scores: pred.p300 || null,
            };
            ui.showProbabilities(pred.probabilities, pred.source === 'mock'
              ? 'Percentages are the model’s rating — MOCK, simulated from the true answer.'
              : pred.probabilities_calibrated ? 'Percentages are the model’s probability for each choice.'
                : 'Percentages are the model’s scores run through a softmax (uncalibrated).');
          }
          if (narrateFirst && !trial.model_display) {
            trial.model_display = { shown_wall_ms: Date.now(), source: pred?.source ?? null, narrated: true,
              calibrated: !!pred?.probabilities_calibrated, probabilities: pred?.probabilities || null, scores: pred?.p300 || null };
          } else if (narrateFirst) trial.model_display.narrated = true;
        },
      });
    } finally {
      document.removeEventListener('visibilitychange', onVis);
    }
  },
};

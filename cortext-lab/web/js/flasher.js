// Flash scheduling. Two parts:
//   makeFlashPlan  - pure, deterministic given an rng (unit-tested in tests/logic.test.mjs)
//   runFlashPlan   - requestAnimationFrame loop; the onOn callback runs INSIDE the frame callback in which the
//                    highlight class is applied, so the marker timestamp is taken on the frame being drawn.
import { shuffle, median, mean, stdev, percentile } from './util.js';

/**
 * Build the flash sequence. Each "repeat" is a random permutation of all options, with the extra rule that the
 * first flash of a repeat never equals the last flash of the previous repeat (no immediate re-flash).
 * With fixedOrder every repeat uses optionIds exactly as given (a hand-made sequence from a preset).
 * With firstRoundInOrder only the first repeat does (one-at-a-time mode: A is shown first, then B, C, D), later repeats are random.
 * Times are relative to flash-phase start, in ms.
 */
export function makeFlashPlan({ optionIds, repeats = 10, onMs = 500, offMinMs = 100, offMaxMs = 200, rng = Math.random, fixedOrder = false, firstRoundInOrder = false }) {
  if (optionIds.length < 2) throw new Error('need at least 2 options to avoid repeating the same flash');
  const events = [];
  let prev = null;
  let t = 0;
  for (let r = 0; r < repeats; r++) {
    let perm = fixedOrder || (firstRoundInOrder && r === 0) ? optionIds.slice() : shuffle(optionIds, rng);
    if (!fixedOrder && perm[0] === prev) {
      const j = 1 + Math.floor(rng() * (perm.length - 1));
      [perm[0], perm[j]] = [perm[j], perm[0]];
    }
    for (const id of perm) {
      const off = offMinMs + rng() * Math.max(0, offMaxMs - offMinMs);
      events.push({
        flash_index: events.length, option_id: id, repeat_index: r,
        on_at: t, off_at: t + onMs, off_gap_ms: off,
      });
      t += onMs + off;
      prev = id;
    }
  }
  return { events, total_ms: events.length ? events[events.length - 1].off_at : 0 };
}

export function frameStats(intervals) {
  if (!intervals.length) return { n: 0 };
  const med = median(intervals);
  return {
    n: intervals.length, median_ms: med, mean_ms: mean(intervals), sd_ms: stdev(intervals),
    p95_ms: percentile(intervals, 95), max_ms: Math.max(...intervals),
    dropped: intervals.filter((d) => d > med * 1.5).length,
  };
}

/**
 * Run a plan on the display clock.
 * @param plan        result of makeFlashPlan
 * @param onOn(ev, frameTs, plannedAbsMs)   called inside rAF right after the caller applies the highlight
 * @param onOff(ev, frameTs)
 * @param token       { aborted:boolean, reason:string } - set aborted to stop at the next frame
 * @param leadMs      delay before the first flash (lets the first frame be scheduled cleanly)
 */
export function runFlashPlan(plan, { onOn, onOff, token = {}, leadMs = 50 }) {
  return new Promise((resolve) => {
    const events = plan.events;
    const intervals = [];
    const results = [];
    let est = 1000 / 60;
    let last = null;
    let i = 0;
    let t0 = null;
    let on = false;

    const finish = (aborted) => resolve({ aborted, results, frame: frameStats(intervals), planned_ms: plan.total_ms });

    function frame(ts) {
      if (token.aborted) {
        if (on) onOff(events[i], ts, true);
        return finish(true);
      }
      if (t0 === null) t0 = ts + leadMs;
      if (last !== null) {
        const d = ts - last;
        intervals.push(d);
        if (intervals.length > 5) est = median(intervals.slice(-30));
      }
      last = ts;
      const rel = ts - t0;
      const half = est / 2;
      // At most ONE state change per frame so every highlight lasts at least one displayed frame.
      if (i < events.length) {
        const ev = events[i];
        if (!on && rel >= ev.on_at - half) {
          on = true;
          const res = { ...ev, planned_on_abs: t0 + ev.on_at, frame_on_ms: ts, onset_err_ms: ts - (t0 + ev.on_at) };
          results.push(res);
          onOn(ev, ts, res);
        } else if (on && rel >= ev.off_at - half) {
          on = false;
          const res = results[results.length - 1];
          res.frame_off_ms = ts;
          res.dur_ms = ts - res.frame_on_ms;
          res.dur_err_ms = res.dur_ms - (ev.off_at - ev.on_at);
          onOff(ev, ts, false);
          i++;
        }
      }
      if (i >= events.length) return finish(false);
      requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
  });
}

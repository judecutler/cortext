// Trial-type plug-in registry.
//
// A trial type is an object:
//   {
//     id:          'mcq_flash',
//     label:       'Multiple choice, flashing options',
//     implemented: true,
//     // async run(ctx, trial): drive the phases of ONE trial. `trial` is the persisted record (options, positions,
//     // flash_params ...). Throw nothing; use ctx.guard(promise) around every wait so Redo/Discard/Pause/Stop work.
//   }
//
// ctx (see session.js `_makeCtx`):
//   ctx.settings, ctx.ui (stage API; see views/run.js), ctx.markers, ctx.predictor, ctx.rng
//   ctx.mark(event, fields)              emit a trial-scoped marker (adds trial_id, trial_index, true_answer_id)
//   ctx.frame(fn) -> Promise<frameTs>    run fn inside the next requestAnimationFrame callback
//   ctx.guard(promise)                   rejects with InterruptError if the operator interrupts
//   ctx.token                            { aborted, reason } for rAF loops
//   ctx.waitResponse(timeoutMs)          resolves {optionId, ts} on tile click / key, or null on timeout
//   ctx.save()                           persist the trial record now
//
// To add a type: create a file, export an object like the above, and call registerTrialType() below.
import { mcqFlash } from './mcq_flash.js';
import { mcqSequential } from './mcq_sequential.js';
import { stubTypes } from './stubs.js';

const registry = new Map();
export function registerTrialType(t) { registry.set(t.id, t); }
export const getTrialType = (id) => registry.get(id);
export const listTrialTypes = () => [...registry.values()];

registerTrialType(mcqFlash);
registerTrialType(mcqSequential);
stubTypes.forEach(registerTrialType);

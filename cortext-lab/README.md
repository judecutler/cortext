# CorText Lab

Browser experiment + demo app for the CorText EEG project. Participants see a multiple-choice question
("Which one is the dog?") whose options flash one at a time while EEG is recorded. They silently attend to the
option they believe is correct, then **click it after the flashing stops**. The click is the ground truth for
training and for checking a P300 (attention response) detector.

* Plain HTML/JS (ES modules, **no build step, no npm, no accounts, nothing paid**).
* One local Python script (standard library only) serves the app **and** relays flash markers to
  [LSL](https://labstreaminglayer.org) (optional `pip install pylsl`) with an always-on CSV backup.

```
 browser (rAF flash + marker)  --WebSocket-->  bridge/bridge.py  --LSL "CorTextMarkers"-->  LabRecorder / your recorder
        |                                             |
   IndexedDB (trials, markers)                  bridge/logs/markers_*.csv (backup)
```

## Run it

```bash
python bridge/bridge.py                 # then open http://localhost:8000
pip install pylsl                       # optional, enables the LSL marker outlet
```

Python 3.8+ (tested on 3.13). Use a Chromium-based browser (Chrome/Edge) for the best frame timing. Without `pylsl` the bridge says
so at startup and still logs every marker to CSV. Options: `--port`, `--host 0.0.0.0` (serve other machines; there is
no authentication, so only on a trusted network), `--lsl-name`, `--format json|compact`, `--no-lsl`, `--no-csv`.

Typical recording setup: start the bridge, open the page, start your EEG stream, open LabRecorder, press *Update*,
select `CorTextMarkers` **and** the EEG stream, *Start*, then run the session. The dev panel shows red/green
indicators for: bridge connection, LSL outlet, LSL recorder attached, EEG stream present/delivering samples.

Hotkeys (run view): `Space`/`Enter` next/start · `Backspace` discard · `R` redo · `1–4` or `A–D` choose answer ·
`P` pause · `M` Researcher/Demo mode · `Shift+F` fullscreen · `Shift+B` live score bars.

## Live EEG, cleaning and the shared dataset

See [`../DATA_COLLECTION.md`](../DATA_COLLECTION.md). In short:
- `eeg/stream_eeg.py` streams the g.tec headset (via gpype) to LSL; `eeg/fake_eeg.py` streams SIMULATED EEG for testing.
- The bridge (`bridge/eeg_engine.py`) records it per session (`bridge/logs/eeg_<session>_*.csv`), reports per-electrode
  quality and band power, and runs a live P300 scorer on `ws://…/predict` (Predictor = "Live EEG").
  - The scorer starts with an untrained heuristic (mean 250–450 ms amplitude over parietal channels).
  - After 5 valid trials and 30 target epochs it retrains a shrinkage-LDA on this session after each trial.
  - It never uses a trial's own labels before that trial has been predicted.
  - `--p300-model file.npz` loads a model trained with `pipeline/train_p300.py` instead.
- At session end, `pipeline/build_session.py` cleans the EEG and writes `datasets/cortext-p300/sessions/<id>/`.
  The summary page's "Publish to GitHub" button commits and pushes only that folder.
- `#/brain` is the Neural monitor: scalp map, region band power, live traces, P300 waveform and answer evidence. It is
  for the audience on a second screen, never for the participant during flashing.
- Each browser tab keeps its own clock offset in the bridge. Opening the monitor in a second tab must not (and no longer
  does) shift the session's marker timestamps.

## Presentation modes

Pick one under **Setup → Presentation mode**. Both write the same marker schema and the same exports, so the same
epoching/detector code works for either. "What each stimulus shows" (image / text / image + text) applies to both.

| | **Flash options in place** (original) | **One at a time, then choose** (new) |
|---|---|---|
| `trial_type` | `mcq_flash` | `mcq_sequential` |
| During the EEG phase | all answer tiles stay visible; one lights up at a time (500 ms on, 100–200 ms gap, 10 repeats) | the **question stays on screen**; each option's picture/text appears **alone** for 1 s (default), 0.3 s gap, once per option (default; repeats configurable) |
| After it | tiles freeze, click your answer | the full multiple-choice grid appears with **the model's probability on every choice**, then you click the correct one |
| Marker event for each onset | `flash_on` (`extra.mode: "flash"`) | `flash_on` (`extra.mode: "sequential"`) so it epochs identically |

Sequential mode details: the model scores the trial only **after** the last stimulus has gone off, and the numbers are
shown **after** that, so they cannot touch the EEG window (the test suite checks this). The percentages come from the
predictor: the mock (clearly labelled, simulated from the true answer), or your model server's `probabilities` field, or,
if it only sends scores, a softmax of them labelled *uncalibrated*. Untick "Show model probability" to get the same flow
without the numbers.

**Anchoring caveat:** seeing the model's probabilities can bias the click. Each such trial stores what was shown
(`model_display`, `model_shown_before_click = true`), the summary page warns about it, and `model_feedback.csv` carries the
flag. Use those clicks as *feedback* ("the model said 60% for X, the person picked Y, the true answer was Z"), not as
independent ground truth, and never evaluate the model on them. For `mcq_sequential` the **true answer is known from the
question**; the click shows whether the person agreed with it, which is what makes it useful feedback.

## Trial flow (flash mode)

1. **Prompt** (default 2 s): question + tiles (image, text, or both).
2. **Flash phase**: options light up one at a time (default 500 ms on, 100–200 ms random gap, 10 repeats/option).
   Each repeat is a random permutation of all options; the same option is never flashed twice in a row, also across
   repeat boundaries. Clicks during flashing are ignored as answers and logged (`early_click`).
3. **Response**: tiles freeze, click (or key) the answer. Reaction time is measured from the response-cue frame.
4. **Post-trial**: optional correct/incorrect feedback, the predictor readout next to the click, then
   **Discard trial** (`Backspace`), **Redo this question** (`R`) or **Next** (`Space`).

Cleaning: nothing is ever deleted. Discard/redo/tab-hidden/pause-mid-trial only flip `valid` to `false` and record a
reason in `validity_log` (misclick, blink or movement, distracted, equipment glitch, other, plus automatic ones:
`redo`, `paused`, `tab_hidden`, `no_response`, `aborted`, `unspecified`). The review page (`#/review/<session>`)
lets you change any trial's validity afterwards; every change is appended to the log. *Redo* re-queues the same
question as a new attempt (`attempt` + 1) with a fresh option order. *Pause* mid-trial aborts that trial (flash timing
cannot be paused), re-queues it, and waits for resume.

## Marker schema

Every marker is one row in the page (IndexedDB → `markers.csv`), one JSON string on the LSL outlet and one row in
`bridge/logs/markers_<session>_<time>.csv`. **Epoch EEG on `flash_on`.**

| field | meaning |
|---|---|
| `seq` | per-session counter, no gaps |
| `session_id` | e.g. `P01_20261024133800_abc` |
| `event` | see list below |
| `t_ms` | page `performance.now()` when the marker was created, inside the rAF callback that applied the highlight |
| `frame_ms` | the rAF callback's own timestamp (start of that frame) |
| `wall_ms` | `performance.timeOrigin + t_ms` (Unix ms) |
| `trial_id` | unique per attempt (a redo gets a new id); `trial_index` = question number, same across retries |
| `option_id` | flashed/clicked stimulus id from the stimulus config |
| `option_position` | 0-based tile position (A=0…) |
| `category` | the option's category (`animal`, `object`, `shape`, …) |
| `is_target` | `true` if the option is the true answer |
| `repeat_index` | 0-based repeat of this option within the trial |
| `flash_index` | 0-based flash number within the trial |
| `true_answer_id` | the trial's true answer id |
| `bridge_sent` | whether the WebSocket was open when it was emitted (CSV/IndexedDB have it regardless) |
| `extra` | JSON for event-specific details |

Events: `session_start`, `trial_start`, `prompt_on`, `flash_phase_start`, **`flash_on`**, `flash_off`,
`flash_phase_end`, `response_cue`, `click`, `early_click`, `no_response`, `trial_end`, `trial_aborted`,
`trial_invalid`, `trial_redo`, `pause`, `resume`, `session_end` (and `timing_flash` from the timing page).

The bridge CSV adds `recv_perf_ms`, `bridge_t_ms`, `est_transit_ms` (receive time minus the page timestamp mapped into
bridge time) and `lsl_timestamp`. **LSL timestamps**: the page runs 16 ping-pong rounds at connect and every 30 s
(not during flashing), keeps the lowest-RTT sample (NTP-style) and reports `offset = bridge_clock − page_clock`.
The bridge stamps each marker with `lsl_local_clock() − transit`, so WebSocket and event-loop delay do not move
the marker in the EEG time base. The remaining error is bounded by about half the sync RTT (sub-millisecond on
localhost) plus the display latency described below.

LSL string formats: `json` (default; the fields above) or `compact`
(`event|session|trial|option|category|is_target|repeat|flash`).

### What the timestamp does and does not mean

The marker is taken **in the frame callback where the highlight class is applied**, i.e. just before the browser
paints that frame. The photons appear later: roughly one frame of compositing plus the monitor's own latency
(typically 10–50 ms, constant per setup). Flash onsets are also quantised to display frames (16.7 ms at 60 Hz). So:
expect a constant lag of tens of ms and a jitter of about one frame. Measure the constant with a photodiode
(`timing.html`, `TIMING_CHECKLIST.md`) and correct for it in analysis; do not assume it is zero.

## Data

* **IndexedDB** (`cortext-lab`: `sessions`, `trials`, `markers`), written as you go (markers flushed every 250 ms and
  on page hide), so a crash loses at most the last quarter second. The app asks the browser for persistent storage;
  still **export after every session** (browsers can evict site data).
* **Export** (Sessions page, Summary page): one session as a zip (`session.json`, `trials.csv`, `markers.csv`, `model_feedback.csv`), all
  sessions as one zip (a folder per session), or the single files.
* `trials.csv` columns: `session_id, trial_id, trial_index, attempt, type, status, valid, invalid_reason,
  invalid_note, prompt, tile_content, option_ids_by_position, true_answer_id, true_answer_category,
  clicked_answer_id, clicked_category, correct, rt_ms, early_clicks, n_flashes, flash_order, flash_on_ms,
  flash_off_min_ms, flash_off_max_ms, repeats, prediction_source, prediction_best_option_id, prediction_confidence,
  model_probabilities, model_shown_before_click, started_wall_ms, ended_wall_ms, notes`. `session.json` additionally holds, per trial, the full per-flash timing
  (`flash_order[]`: scheduled vs actual onset, duration, marker seq) and the validity history.
* **`model_feedback.csv`** (for improving the model): one row per (trial attempt, option) with `model_probability`,
  `model_score`, `is_true_answer`, `is_clicked`, `is_model_top_choice`, `human_agrees_with_model`,
  `model_shown_before_click`, `valid` and the trial/option ids. Join it to `markers.csv` on `(session_id, trial_id,
  option_id)` to get the EEG epochs. Only trials where the predictor returned probabilities appear.
* Session record: participant, device (Muse / 8-node / 32-node / Other), notes, seed (option positions and flash
  orders are reproducible from it), settings, measured display refresh, user agent.

## Stimuli

**Upload (Setup → Stimulus set).** Choose files, choose a folder, or drop files on the box:

* a stimuli `.json` (format below) plus the image files it names, matched **by file name** (folders and upper/lower case
  are ignored; an item with no `image` picks up a file named `<id>.png/.svg/...`), or
* **just images**: no `.json` needed. Each image becomes an item (file name = id and label), and the **sub-folder name is the
  category**: upload `pets/` containing `animals/dog.png`, `animals/cat.png`, `objects/cup.png` and you get categories
  `animals` and `objects`. Files dropped loose (not as a folder) are all `uncategorized`.

Warnings appear for missing/unused files. Uploaded sets are stored in the browser (IndexedDB; never overwritten, a repeated
name gets "(2)"), reappear in the "Use" menu after a reload, and each session records which set was used plus the item list
(file names, not temporary links) in `session.json` (`stimuli_source`, `stimuli_snapshot`). The image files themselves are not
exported, so keep your original folder. Audio files can be uploaded and are stored on the item but are **not played yet**.
Use "Download example config" in setup for a template.

The config format: `web/config/stimuli.json` lists `items` (`id, category, label, text, image`, optional `prompt`, `distractor_set`),
`prompts` per category, `distractor_sets` (`all`, `same_category`, or explicit `members`) and an optional
`trial_plan` (explicit `{target, options[], tile_content, prompt}` questions). Add categories/items by editing the JSON
and dropping files in `web/stimuli/`; no code changes. The sample set is 6 animals, 3 objects, 3 shapes, drawn from
scratch as SVG and released as CC0 (`web/stimuli/LICENSE.txt`).

## Prediction hook

Setup → Predictor: `off`, `mock`, or `ws` (default `ws://localhost:8765`).

* **mock**: simulated scores so the demo works with no model. It is *given the true answer*, so it is **not
  evidence**: it is labelled `[MOCK — simulated, not evidence]` in the UI and stored with `source: "mock"`; the
  summary keeps mock and real predictions separate.
* **ws**: protocol in `tools/mock_predictor_server.py` (a runnable template that returns random scores). The app sends
  `trial_start` (options, no labels), a `flash` message per flash (option id, repeat, timestamps) and `trial_end`; the
  server may push `live` scores any time and must answer `trial_end` with `prediction`; add `probabilities` (`{option_id: p}`)
  to that message to control the percentages shown in sequential mode. With "Send answer back to model server" ticked, the
  app also sends a `feedback` message *after* each trial (true answer, click, validity, `model_shown_before_click`). **`is_target` and the true
  answer are never sent to the predictor**, so a model cannot see the labels. Live bars under the tiles
  (`Shift+B`, demo use) change the screen during flashing, so leave them off for recordings.

## Extending

* **New trial type**: add `web/js/trialtypes/<id>.js` exporting `{id, label, implemented: true, async run(ctx, trial)}`
  and register it in `trialtypes/index.js`. The context API (markers, frame-synced callbacks, interruptible waits,
  predictor) is documented at the top of that file; `mcq_flash.js` is the reference. Stubs with plans exist for
  `passive_view` (RSA), `picture_word_match` (image/text/audio match–mismatch), `imagery_hidden` and `free_choice`.
* **Stimulus types**: items already accept `image`, `text` and an `audio` field (stored on the option, not played yet).
* **Other recorders**: markers are plain JSON over WebSocket; the bridge's `handle_marker` is the one place to add
  e.g. a serial/parallel-port or BrainFlow output.

## Known limits and design cautions

* Verified here: unit tests, a headless-Chromium end-to-end session, bridge→LSL→inlet round trip. **Not yet verified
  on real hardware** (Muse, the 8/32-node amplifiers, LabRecorder, a photodiode). Run `TIMING_CHECKLIST.md` first.
* Browser timing is good (frame-locked, sub-frame scheduling error in tests) but not hardware-grade; background
  tabs stop `requestAnimationFrame`, so a hidden tab aborts the trial automatically.
* **Gaze confound**: tiles are at different screen positions, so overt eye movements toward the target can produce
  signals that look like attention responses. Positions are randomised and stored (`option_position`); consider a
  fixation protocol or EOG/gaze checks, and analyse position as a covariate.
* **Low-level stimulus confound**: the placeholder images differ in colour, luminance and extent between categories.
  Any claim about category/semantic information needs a low-level-feature baseline (as in the project's control list).
* Ground truth is "what the participant clicked as correct". For `mcq_flash` the *true* answer is known from the
  question; the click tells you whether they knew it. Use `correct` and the validity flags to select trials.
* **If the bridge drops mid-session** the experiment keeps running and the page reconnects by itself, but markers
  emitted while it was down are *not* replayed to LSL or the bridge CSV. They are still in IndexedDB and the exported
  `markers.csv` with `bridge_sent=false` (and the summary page says how many were not delivered), so the data is
  recoverable from the page timestamps, but the EEG recording will lack those events. Treat such trials as invalid
  for LSL-based analysis, or re-align from `markers.csv`.
* No authentication on the bridge; keep it on localhost or a trusted network.

## Layout

```
bridge/    bridge.py (HTTP + WebSocket + LSL + CSV), wsproto.py (RFC 6455 helpers), logs/ (git-ignored)
web/       index.html, timing.html, css/, js/ (app, session, trialtypes/, views/, flasher, markers, db, stimulusSets, …),
           config/stimuli.json, stimuli/*.svg
tools/     analyze_timing.py, mock_predictor_server.py
tests/     logic.test.mjs, test_bridge.py, test_bridge_lsl.py, e2e*.py       (see TEST_PLAN.md)
```

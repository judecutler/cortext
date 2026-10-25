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

Hotkeys (run view): `Space`/`Enter` next/start · `Backspace` discard · `R` redo question · `1–4` or `A–D` choose answer ·
`P` pause · `M` Researcher/Demo mode · `V` voice on/off (if "Voice on (demo)" was ticked at Start) · `I` info panel · `Shift+F` fullscreen · `Shift+B` live score bars · `N` next session
(on the end screen). Everywhere: `Ctrl/⌘+Z` undo, `Ctrl/⌘+Shift+Z` (or `Ctrl+Y`) redo.

## Defaults = the demo (2026-10-25)

A fresh browser (and, once, every browser that saved older settings) starts with the hackathon demo flow: **one at a
time, then choose** · question shown 5 s and **read aloud** · each picture 1 s, 0.3 s gap, **3 rounds** · Predictor
**Live EEG** · **narrator before the click** ("Based on Jude's brain waves, I'm thinking answer B is the dog… about 45%").
`models/active/` ships with the **p3000 function**, so the live decoder works from trial 1 with no calibration; pick a
calibrated model in Agent → Model when you have one. Start is refused (with a message) until a headset stream reaches
the bridge; for testing without a headset run `eeg/fake_eeg.py`, or choose Mock on the Agent tab. "Reset design" on the
Experiment tab returns the design to these values (predictor and narrator are on the Agent tab). `tests/e2e_showtime.py` runs exactly this flow from a fresh browser.

## Demo with a Muse headband instead of the 8-channel headset

```bash
pip install muselsl pylsl numpy scipy        # once
python eeg/stream_muse.py                    # turn the Muse on first; leave this running
python bridge/bridge.py                      # then open http://localhost:8000 as usual
```
Nothing else changes: same app, same demo flow, same keys. What is Muse-specific:
- **Streamer.** `eeg/stream_muse.py` (muselsl, Bluetooth LE, no dongle) streams TP9 AF7 AF8 TP10 at 256 Hz with
  **LSL timestamps** (plain `muselsl stream` uses wall-clock time unless you add `--lsltime`; the bridge detects that
  and corrects it, and says so in its log). BlueMuse on Windows works too. The Mind Monitor app does not (OSC, not LSL).
  Close the Muse phone app first: only one program can hold the headband.
- **Channels.** The bridge drops the unconnected "Right AUX" input, so the Experiment tab shows 4 chips.
- **Decoder.** `models/library/p3000_function_muse.npz`: the same no-training spike detector, reading TP9/TP10 (where a
  Muse sees the P300; its reference is Fpz) and ignoring the blink-heavy forehead channels. If the running model is
  the 8-channel p3000 function (the default in `models/active/`), the bridge **switches to the Muse one by itself** when
  a Muse connects, together with its filter chain. You can also pick it on Agent → Model.
- **Filters.** "Muse P300" chain: 60 Hz notch + the usual 0.5–20 Hz band-pass, artifact threshold 150 µV (dry
  sensors). Blinks on AF7/AF8 never reject an epoch (live, in training and in the offline dataset build).
- **Fit.** Push the band down so TP9/TP10 sit on skin behind the ears (not hair) and the forehead sensors touch skin;
  a drop of water on the sensors helps. Wait for 4 green chips (dry sensors can take 1–2 minutes).
- **Expect less than the 8-channel headset.** A Muse has 2 usable P300 channels, so use more rounds per question
  (4–5) if you can, and quote the rehearsal hit rate, not the simulation.
- Test without a Muse: `python eeg/fake_eeg.py --muse` (add `--wall-clock` to mimic plain muselsl);
  `tests/test_muse.py` and `tests/e2e_showtime.py --muse` run the whole Muse path.

## Live EEG, cleaning and the shared dataset

See [`../docs/DATA_COLLECTION.md`](../docs/DATA_COLLECTION.md) and [`../docs/DATA_STORAGE.md`](../docs/DATA_STORAGE.md). In short:
- `eeg/stream_eeg.py` streams the g.tec headset (via gpype) to LSL, `eeg/stream_muse.py` a Muse (see above);
  `eeg/fake_eeg.py` streams SIMULATED EEG for testing (`--muse` for a simulated Muse).
- The bridge (`bridge/eeg_engine.py`) records it per session (`bridge/logs/eeg_<session>_*.csv`), reports per-electrode
  quality and band power, and runs a live P300 scorer on `ws://…/predict` (Predictor = "Live EEG").
  - The scorer starts with an untrained heuristic (mean 250–450 ms amplitude over parietal channels).
  - After 5 valid trials and 30 target epochs it retrains a shrinkage-LDA on this session after each trial.
  - It never uses a trial's own labels before that trial has been predicted.
  - A trained model replaces both: the bridge runs the file in `models/active/` at startup. **Agent → Model** lists
    the team library (`models/library/`, shared through git; `pipeline/train_p300.py` saves there) and the models
    calibrated live on this laptop, and switches the running model without a restart. `--p300-model FILE` overrides
    `models/active/`; `--p300-model none` ignores it. See [`models/README.md`](models/README.md).
  - Before epoching, every sample goes through the running **filter chain** (`bridge/filters.py`): ordered, causal
    layers (band-pass, high/low-pass, mains notch, re-reference). The default is the 0.5–20 Hz band-pass every model so
    far was trained with. **Agent → System map** edits a draft chain, shows the real signal after every layer, and
    applies it live; chains are shared through git in `filters/library/`, and the running one is kept in `filters/active/`
    (`--filters FILE` overrides it). A model stores the chain it was trained with and only runs with that chain.
    See [`filters/README.md`](filters/README.md).
- At session end, `pipeline/build_session.py` cleans the EEG and writes `data/cortext-p300/sessions/<id>/`
  (git-ignored). Upload sends only that folder to the private Hugging Face dataset (button, or automatic: Experiment → Lab & data).
- **Raw EEG → Hugging Face** (Experiment → Lab & data): *Record raw EEG* captures the headset stream with no stimuli
  (every sample: `unix_time`, `lsl_time`, one µV column per electrode) and *Stop & upload* puts it in the dataset's
  `raw_eeg/` folder; *Upload a raw EEG file…* sends any file from another recorder there untouched (CSV, XDF, EDF, FIF…).
  Command line: `python datasets/cortext-p300/cortext_store.py raw FILE --participant P01 --note "…"`. Simulated
  recordings stay on this computer.
- The **EEG** tab (`#/eeg`) is the neural monitor: scalp map, region band power, live traces, P300 waveform and answer evidence. It is
  for the audience on a second screen, never for the participant during flashing.
- Each browser tab keeps its own clock offset in the bridge. Opening the monitor in a second tab must not (and no longer
  does) shift the session's marker timestamps.

## Tabs

| Tab | What it is for |
|---|---|
| **Experiment** | Three sub-pages, in the order you use them: **Images & categories** (see and rearrange the stimulus set), **Configure & run**, **Recorded sessions** (buckets, filters, bulk actions and real deletion). *Configure & run* is one page in four sections: **1 Participant & headset** (headset check and electrode chips, participant ID, device, narrator name, and a summary of what the Agent will do), **2 Design** (presets, mode, stimulus set, timing, voice & audio, response), **3 Questions & image order**, **4 Lab & data** (photodiode, data storage / Hugging Face). Start sits at the bottom with readiness chips. (This replaces the old Setup tab; `#/setup` links land here.) |
| **Agent** | Everything between the headset and the answer, as a **system map**: Input → filter layers → Epoching → Model → Output → Result, plus the offline lane (recording → dataset cleaning → training → model library). Click a block to inspect or change it: filter layers (add with +, drag to reorder, ● bypass, ✕ remove, before/after trace and spectrum), the artifact limit, the P300 model picker, the predictor (live EEG / mock / model server) and the narrator's voice and LLM. Filter edits are a draft until **Apply to live**; the **Filter library** below shares chains like models. |
| **EEG** | Neural monitor: scalp map, band power, live traces, P300 waveform. Can open in its own window for a second screen. |
| **Timing** | The display/marker timing test, run inside the app (no new tab; it needs the same bridge connection). |

## Undo

Edits happen immediately, without “are you sure?” dialogs, and every one can be reversed: a notice says what changed
with an **Undo** button, `Ctrl/⌘+Z` undoes and `Ctrl/⌘+Shift+Z` redoes, and the **↶ Undo** button in the top bar names
the next thing it will undo. This covers the question list, presets (load, update, delete, reset), everything on
Images & categories, discarding a trial after the click, validity changes in the run panel and on the review page, and
the “superseded” mark. History is per page (it is cleared when you move to another page). What cannot be reversed keeps
its confirmation: deleting a recorded session (from disk / Hugging Face) and starting a recording. Nothing pops up over
the participant’s screen while stimuli are on it.

## Images & categories

`#/library` shows the chosen stimulus set as a board: one column per category with its images.
- **Drag** an image to another category or up/down to reorder; **drop image files** onto a column to add them there
  (or use *+ Add images*). Keyboard: Tab to an image, `Alt+↑/↓` reorder, `Alt+←/→` move to the neighbouring category,
  `Enter` edit, `Delete` remove.
- **Categories**: click the name to rename, ◀ ▶ to reorder, ✕ to delete (its images move to a category you choose,
  or go with it), *+ Category* or the dashed column to add one. Empty categories are kept.
- **Edit** (✎ or double-click): label (used as `{label}` in question wording), text shown on screen, category, image.
  Item **ids never change**, so question lists and recorded sessions keep matching.
- **Sets**: switch set, *Use this set in the experiment*, *New set…* (empty), *Duplicate…*, *Delete set*. Every change
  is saved in this browser at once. The built-in sample set is files on disk, so your first change makes an editable copy
  (“Sample set (my copy)”) and, if the experiment was using the built-in set, switches it to the copy.
- Removing an image that your question list uses says which questions (Start then reports the list as invalid).

## Next session, redo and restart

When a session ends, the end screen shows:
- **Next session** (`N`, or `Enter` in the participant field): same settings and question list as the session just
  recorded, a new random seed (and a re-shuffle if “Re-shuffle every time a session starts” is on). Change the
  participant ID there for the next person. The new session stores `previous_session_id`.
- **Redo this session**: exactly the same questions in the same order (same seed / same list). The first recording is
  **kept, not deleted**, and marked `superseded_by` (the redo stores `redo_of`). Its summary page shows a banner with
  *Use it after all* to remove the mark; the Recorded sessions list shows *superseded* / *redo* pills.
- **Session summary**, **Review / clean trials**, **Change settings**.

The same Next / Redo buttons are on every session’s summary page. **↺ Restart session** in the run panel stops a session
part-way and starts it again from question 1 (same as Redo). The mark travels into the dataset’s `meta.json`;
`cortext_data.load_epochs` / `load_imagery` and `train_p300.py` skip superseded sessions unless you ask for them
(`include_superseded=True`, `--include-superseded`, or by naming them in `sessions=` / `--sessions`).

## Presets, question wording and image order

- A **preset** is a named bundle of the experiment settings plus an optional question list. Built-in presets are provided;
  yours are saved in this browser and can be **exported/imported as a .json file** to share with teammates.
  The participant, headset, storage and Agent settings (filters, model, predictor) are deliberately not part of a preset.
- **Questions & image order** card: tick *Use this list for the session* and the app asks exactly those questions, with your
  wording and images. Edit each question's text, the correct answer and the images (◀ ▶ reorder, ✕ remove, add more).
  *Generate from stimulus set* fills the list; *Apply wording to all* uses a template such as `Which one is the {label}?`.
- **Randomize**: *Shuffle questions*, *Shuffle images* (order inside each question), *Shuffle both*, a 🔀 per question, and
  *Re-shuffle every time a session starts* so each participant gets a different pattern.
- **Image order in each question**: *Random every trial* (default; avoids position effects) or *Exactly as listed*
  (tile positions and the one-at-a-time sequence follow your order; position then correlates with the item, so use it
  deliberately). The list used is stored in the session's settings (`question_plan`).
- During a session, **Hide panel** (key `I`) hides the researcher info sidebar and keeps the top navigation.

## Presentation modes

Pick one under **Experiment → Design → Presentation mode**. Both write the same marker schema and the same exports, so the same
epoching/detector code works for either. "What each stimulus shows" (image / text / image + text) applies to both.

| | **Flash options in place** (original) | **One at a time, then choose** (the demo; default) |
|---|---|---|
| `trial_type` | `mcq_flash` | `mcq_sequential` (**default**) |
| During the EEG phase | all answer tiles stay visible; one lights up at a time (500 ms on, 100–200 ms gap, 10 repeats) | the **question stays on screen**; each option's picture/text appears **alone** for 1 s (default), 0.3 s gap, 3 rounds (default; configurable) |
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

## Voice, audio stimuli and the narrator (demo)

All free. Nothing needs an account unless you choose a cloud LLM.

| Feature | Where | What happens |
|---|---|---|
| **Read the question aloud** | Experiment → Voice & audio | Spoken during the prompt. The stimuli start only after the voice has finished (+400 ms), so it never overlaps the EEG window. |
| **Stimulus modality** | Experiment → Voice & audio | *Visual* (as before), *Audio* (each option is spoken; nothing changes on screen, the one-at-a-time view shows a static “Listen”), or *Audio + visual*. Uses the item's own audio file if the stimulus set has one (upload `dog.mp3` next to `dog.png`, or set `"audio"` in the config), otherwise speaks its text. |
| **Narrator** | Experiment → Voice & audio (when), Agent → Result → Voice & narrator (how) | After the EEG phase, announces the model's guess out loud with a caption: “Hmm... based on the brain patterns, I'm about 78% sure Jude saw the dog at option B.” *After the answer* (default, clean data) or *Before the answer* (show mode). |
| **Narrator context** | Agent → Result → Voice & narrator | Mood presets (warm, sports commentator, stage mind-reader, calm scientist, dry skeptic) and a free-text box for your own persona or experiment context. A fixed set of rules is always added after it: say the real percent and option, never claim certainty, and say “simulation” when the prediction is mock/simulated. Replies without the real percent are rejected. |

**Voices.** The bridge makes speech files (cached in `bridge/logs/tts_cache/`) with the first engine that works:

```bash
pip install edge-tts      # Microsoft neural voices (free, no key, needs internet): best quality
pip install pyttsx3       # the computer's own voices, offline (Windows/macOS built in; Linux needs espeak-ng)
```

With neither, the browser's built-in speech is used. It works, but its start time is not exact, so audio stimuli are then
flagged `audio_timing: "approximate"`. For EEG with spoken stimuli, install one of the two.

**Narrator text.** *Built-in lines* are instant and offline. For LLM-written lines (they follow the mood/context box),
the bridge calls one of these. Keys stay in the bridge and never reach the browser:

| Provider | Cost | Setup |
|---|---|---|
| Ollama | free, local, offline | install [ollama.com](https://ollama.com), `ollama pull llama3.2` |
| OpenAI-compatible | Groq has a free tier; OpenAI/OpenRouter paid per use | `OPENAI_API_KEY`, `OPENAI_BASE_URL` (e.g. `https://api.groq.com/openai/v1`), `OPENAI_MODEL` (e.g. `llama-3.1-8b-instant`) |
| Claude | paid per use (a fraction of a cent per line with Haiku) | `ANTHROPIC_API_KEY`, optional `ANTHROPIC_MODEL` |

Set them as environment variables before `python bridge/bridge.py`, or in `bridge/voice.local.json` (git-ignored):
`{"openai_base_url": "https://api.groq.com/openai/v1", "openai_api_key": "gsk_...", "openai_model": "llama-3.1-8b-instant"}`.
If the LLM fails or takes over 5 s, the built-in line is used, so the demo never stalls. *Test voice* / *Test narrator*
on Agent → Result try the whole chain.

**What is stored.** Each trial gets `narration` {when, text, provider, model, mood, custom_context, …}; the session's
settings keep the narrator context used. The *Name for the narrator* (Experiment → Participant) stays in the browser: it is replaced by
`{name}` in the stored text and never written to the session, markers or dataset. Note that a cloud LLM does receive it.
*Before the answer* sets `model_display.narrated` and `model_shown_before_click`, so keep those clicks out of model
evaluation, as with displayed probabilities. With the mock predictor the narrator always says it is a simulation.

## Trial flow (flash mode)

1. **Prompt** (default 2 s): question + tiles (image, text, or both).
2. **Flash phase**: options light up one at a time (default 500 ms on, 100–200 ms random gap, 10 repeats/option).
   Each repeat is a random permutation of all options; the same option is never flashed twice in a row, also across
   repeat boundaries. Clicks during flashing are ignored as answers and logged (`early_click`).
3. **Response**: tiles freeze, click (or key) the answer. Reaction time is measured from the response-cue frame.
4. **Post-trial**: optional correct/incorrect feedback, the predictor readout next to the click, then
   **Discard trial** (`Backspace`), **Redo this question** (`R`) or **Next** (`Space`).

Cleaning: trials are never deleted by accident. Discard/redo/tab-hidden/pause-mid-trial only flip `valid` to `false` and record a
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
Voice features add `question_audio_start` / `question_audio_end` and `narration_start` / `narration_end`, and every
`flash_on` carries `extra.modality` (`visual` / `audio` / `audiovisual`); for audio also `audio_engine`, `audio_timing`
(`buffer` = sample-accurate Web Audio start, or `approximate` = browser speech fallback), `audio_latency_ms` (device output
latency), `audio_offset_ms` (frame timestamp → sound reaching the speaker; shift audio epochs by this) and `audio_ms`.

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
(the **Timing** tab, `TIMING_CHECKLIST.md`) and correct for it in analysis; do not assume it is zero.

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

**Upload (Experiment → Design → Stimulus set).** Choose files, choose a folder, or drop files on the box:

* a stimuli `.json` (format below) plus the image files it names, matched **by file name** (folders and upper/lower case
  are ignored; an item with no `image` picks up a file named `<id>.png/.svg/...`), or
* **just images**: no `.json` needed. Each image becomes an item (file name = id and label), and the **sub-folder name is the
  category**: upload `pets/` containing `animals/dog.png`, `animals/cat.png`, `objects/cup.png` and you get categories
  `animals` and `objects`. Files dropped loose (not as a folder) are all `uncategorized`.

Warnings appear for missing/unused files. Uploaded sets are stored in the browser (IndexedDB; never overwritten, a repeated
name gets "(2)"), reappear in the "Use" menu after a reload, and each session records which set was used plus the item list
(file names, not temporary links) in `session.json` (`stimuli_source`, `stimuli_snapshot`). The image files themselves are not
exported, so keep your original folder. Audio files can be uploaded and are stored on the item but are **not played yet**.
Use "Download example config" on the Stimulus set card for a template.

The config format: `web/config/stimuli.json` lists `items` (`id, category, label, text, image`, optional `prompt`, `distractor_set`),
`prompts` per category, `distractor_sets` (`all`, `same_category`, or explicit `members`) and an optional
`trial_plan` (explicit `{target, options[], tile_content, prompt}` questions). Add categories/items by editing the JSON
and dropping files in `web/stimuli/`; no code changes. The sample set is 6 animals, 3 objects, 3 shapes, drawn from
scratch as SVG and released as CC0 (`web/stimuli/LICENSE.txt`).

## Prediction hook

Agent → Output → Predictor: `eeg` (live EEG, the default once a headset is seen), `off`, `mock`, or `ws` (default `ws://localhost:8765`).

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
web/       index.html (timing.html just redirects to the Timing tab), css/, js/ (app, session, trialtypes/, views/, flasher, markers, db, stimulusSets, …),
           config/stimuli.json, stimuli/*.svg
tools/     analyze_timing.py, mock_predictor_server.py
tests/     logic.test.mjs, test_bridge.py, test_bridge_lsl.py, e2e*.py       (see TEST_PLAN.md)
```

# CorText fusion layer

This package combines the **P300 detector** and the **category/imagery model** for the multiple-choice task ("Which one is the dog?", with four options flashing). Each trial gets one answer such as `EEG indicates B (dog), confidence 0.72`, or `unsure`.

Each model's output format can stay as it is, because **each model gets an adapter**. The adapter turns the model's output into one score per option, which is the fusion layer's standard internal format. Wiring in a model means editing YAML, not code.

The fusion layer reads [CorText Lab](../cortext-lab/) exports directly and speaks the app's live WebSocket protocol. You can test everything on mock data first, and the mock data is written in the Lab's exact export format.

```
CorText Lab export (trials.csv, markers.csv, session.json) ─┐
P300 output (any shape) ────────────────────────────────────┼─ adapters ─> one score per option ─> normalize ─> weighted sum ─> prediction
category output (any shape) ────────────────────────────────┘                                     (per trial)   (calibrated)   + confidence
```

## Install

```bash
cd fusion
pip install -e ".[dev]"       # Python 3.9+; numpy, pandas, scipy, pyyaml, matplotlib
pytest -q                     # about 70 tests, about 15 s
python tests/e2e_lab_app.py   # optional: the real Lab app in headless Chromium <-> fusion (needs playwright)
```

This folder is self-contained. It does not change `cortext-lab/`, the notebooks or the root `requirements.txt`.

## Two-minute demo (mock data)

```bash
cortext-fusion mock demo/                        # mock Lab exports + model outputs in 11 formats + configs
cortext-fusion evaluate demo/mock_config.yaml    # held-out evaluation with controls
open demo/report/report.md                       # report and plots
cortext-fusion serve --mock                      # live servers with fake scores, for the app's "ws" predictor mode
```

To check that everything, controls included, drops to chance, run the mock with `--p300-signal 0 --category-signal 0`.

## The CorText Lab contract (what the fusion layer assumes)

- **Stimulus log** = Lab exports. Set `stimulus_log: {lab_export: path}` to a session folder, an "Export all" folder or the `.zip`. Globs work.
  - Only `flash_on` markers count as flashes.
  - The option order is `option_ids_by_position`, so A–D means screen position.
  - Ground truth is `true_answer_id` and `clicked_answer_id`.
  - `valid = false` trials are skipped but counted. Trials aborted before any flash are kept as invalid.
  - The participant comes from `session.json`.
- **Trial keys.** Lab trial ids restart at 0 in every session, so the key is **`<session_id>:<trial_id>`**, the app's own key format. Model outputs may use any of these forms:
  - that key
  - separate `session_id` + `trial_id` columns or fields
  - `<session_id>__<trial_id>` file names
  - the bare `trial_id`, when only one session is loaded
- **Option ids** are the app's item ids, for example `dog_01`. A list *without* ids must be in screen-position order.

## Wire in your model in under 10 minutes

**1. Save your outputs with the trial key** (2 min). Use whichever of these matches your model:

| Your model produces | Save it as |
|---|---|
| a score per flash/epoch | CSV `session_id, trial_id, option_id, repeat_index, score` (one row per flash) |
| `predict_proba` per epoch | `.npy` `[trials, flashes, 2]` + `p300_trial_ids.txt` (one `session:trial` per line), flashes in `flash_index` order |
| a score per option | JSON `{"<session>:<trial>": {"dog_01": 1.2, "cat_02": 3.4, ...}}` |
| probabilities over categories | CSV `session_id, trial_id, p_dog, p_cat, ...` |
| an embedding per trial | `.npy` `[trials, dim]` + ids file + a `{category: vector}` JSON |
| one file per trial | a folder of `<session_id>__<trial_id>.json` / `.npy` / `.csv` |

**2. Ask the tool what it sees** (1 min):

```bash
cortext-fusion describe outputs/p300.csv --markers data/lab_exports/ --name p300
```

It prints the file's shape, dtype, range and NaN count, checks your trial ids against the export, ranks the adapters (testing each one on your trials), and prints a config entry ready to paste.

**3. Paste that entry** into a copy of [`configs/example.yaml`](configs/example.yaml) (2 min). Every option is in [`configs/adapter_cookbook.yaml`](configs/adapter_cookbook.yaml). If higher values mean *less* likely (distances, losses), add `lower_is_better: true`.

**4. Run `cortext-fusion evaluate configs/my_config.yaml`** (3 min). Read **Diagnostics** first. It flags:
- unknown or missing trial ids
- option-order mismatches
- NaNs
- scores that are constant across options
- a model that adds no information
- accuracy **below** chance, which usually means a flipped sign or the wrong option order

### From the starter notebook (`notebooks/p300_model.ipynb`)

`find_p300_peak(erp, times)` returns the peak amplitude of an averaged ERP. Average each option's `flash_on` epochs, take the amplitude, and save the result:

```python
import json
out = {f"{session_id}:{trial_id}": {opt: float(find_p300_peak(erp_by_option[opt], times)[1]) for opt in erp_by_option}
       for (session_id, trial_id), erp_by_option in trials.items()}
json.dump(out, open("outputs/p300_peak_amplitude.json", "w"))
```

Then use `adapter: per_option`. If you score every epoch instead, use the per-flash CSV and `per_flash`, which also gives you accuracy versus the number of repeats.

## Adapters

| Name | Input | Turned into a per-option score by |
|---|---|---|
| `per_option` | list in position order, dict by option id or category, or `option_id` + `score` | used as is (reordered by id) |
| `per_flash` | one score per flash: 1-D array in flash order, `[repeats, options]` grid, or long CSV | mean (or sum/median) per option |
| `probabilities` | one probability per option | logit (`transform: log` for a softmax); 0 and 1 are clipped safely |
| `epoch_target_prob` | P(target) per flash, or `predict_proba` `[flashes, 2]` | mean log-odds per option |
| `class_probabilities` | probabilities over many categories | looked up by each option's category (`category_map` if names differ); `check_target` flags when the top class is just the prompt's category |
| `embedding` | one vector per trial | cosine similarity to each option's category vector |
| `distance` | per-option distances or losses | negated (`lower_is_better: true` works on any adapter) |
| `auto` | any of the above | detected on the first trial, then locked in (the log says which) |

The API is `adapter.to_scores(raw_output, trial_meta) -> list[float]`. To write your own adapter, subclass `cortext_fusion.adapters.Adapter`, implement `_scores(raw, meta, max_repeat) -> (np.ndarray, extras)`, and add it to `ADAPTERS`.

## Fusion and calibration

- **Combining.** Each model's scores are normalized per trial across options (`zscore` by default; also `logsoftmax`, `softmax`, `center` or `none`), then combined as a weighted sum. Confidence is the softmax of the fused scores. A trial is "unsure" when confidence falls below `unsure_threshold`.
- **Default weights.** Only the P300 counts until shown otherwise (the category weight is 0).
- **Calibration.** `cortext-fusion calibrate cfg.yaml --exclude-sessions <test sessions>` learns the weights and temperature by conditional logistic regression on validation trials.
  - A model that starts at weight 0 is switched on only if it lowers the leave-one-session-out log-loss by at least `min_gain_logloss`. With a single session, it stays off.
  - The result is written to `calibration.yaml`, which `fuse`, `serve` and `watch` pick up.
- **Sequential fusion.** Per-flash evidence is accumulated as a running log-score. The trial stops early once confidence ≥ `stop_confidence` after at least `min_repeats` repeats.

## Evaluation (`cortext-fusion evaluate`)

Evaluation is held out by session or participant, never by random trials. Invalid trials are skipped and counted. Every number is reported against both the clicked answer and the true answer.

**Methods compared:**
- each model alone
- fused with the config weights
- fused with weights calibrated on the *other* sessions
- the prior-only baseline

**Controls:**
- shuffled labels: a permutation p-value for every method
- noise scores: the whole pipeline, calibration included, re-run on Gaussian noise; it must land at chance
- prior-only: always answer the training sessions' most common screen position

**What `report/` contains:**
- `report.md`, a filled one-page report (blank copy in [`docs/evaluation_report_template.md`](docs/evaluation_report_template.md))
- `metrics.json` and `predictions_heldout.csv`
- accuracy versus repeats, with an early-stopping simulation
- a reliability plot and the ECE
- a confusion matrix by screen position (a skew suggests a gaze or position confound)
- `issues.txt`

## Live: CorText Lab app ↔ fusion ↔ models

```bash
cortext-fusion serve configs/my_config.yaml      # ws://127.0.0.1:8765 for the app, http://127.0.0.1:8766 for models
cortext-fusion serve --mock                      # no models: random scores, every message has "mock": true
```

1. In the Lab app's setup screen, choose **Predictor: WebSocket model server**. The URL `ws://localhost:8765` is already the app's default.
2. The app sends `trial_start`, `flash` and `trial_end` to the fusion server. Its protocol is in `cortext-lab/tools/mock_predictor_server.py`. **The app never sends `is_target` or the true answer**, so no model can see the labels.
3. Your **P300 process** reads the EEG and the LSL `CorTextMarkers` stream, and POSTs one score per epoch:
   ```bash
   curl -X POST localhost:8766/sequential/flash -d '{"trial_id": 7, "session_id": "P01_...", "flash_index": 12, "value": 0.83}'
   ```
   `value` is P(target). For raw decision scores, set `sequential.input: score`. You can send `option_id` instead of `flash_index`. A score that arrives before the app's flash event is held until that event arrives.
4. Alternatively, after the trial ends, POST the whole trial's output to `/predict`:
   ```json
   {"trial_id": 7, "session_id": "P01_...", "outputs": {"p300": <raw>, "category": <raw>}}
   ```
   No options are needed for a live trial.
5. Every update is pushed to the app as `live` (bars per option). On `trial_end`, the server waits up to `final_wait_s` (default 1 s; the app gives up after 3 s) for late scores, then sends the `prediction`. The prediction includes `best_option_id`, `confidence`, `p300`, `category_scores`, `unsure`, `message` and `mock`. If no scores arrived, `best_option_id` is null, and the app shows "predictor sent no answer".

| HTTP endpoint (port 8766) | Body | Returns |
|---|---|---|
| `GET /health` | | mode, models, weights, live trials |
| `POST /sequential/flash` | `{trial_id, [session_id], flash_index or option_id, value}` | running prediction |
| `POST /predict` | `{trial_id, [session_id], outputs}` (live), or with `options` for any trial | prediction (also pushed to the app) |
| `POST /sequential/evidence` | `{trial_id, model, output}` or `{..., scores}` | running prediction (category model; uses its weight) |
| `POST /sequential/start` / `end` | trial / `{trial_id}` | for use without the app (tests, other front ends) |

If your models write one file per trial instead of using HTTP, run:

```bash
cortext-fusion watch cfg.yaml --folder p300=live/p300 --folder category=live/category --out live.jsonl
```

## Python API

```python
from cortext_fusion import load_config, load_dataset, Fuser, fuse_dataset, evaluate, load_lab_export, make_adapter
cfg = load_config("configs/my_config.yaml")
ds = load_dataset(cfg)                         # adapters applied; ds.issues has the diagnostics
preds = fuse_dataset(ds, Fuser.from_config(cfg))
res = evaluate(ds, cfg, out_dir="report")
metas = load_lab_export("data/lab_exports/")   # {"<session>:<trial>": TrialMeta}
```

## Layout

```
fusion/
  cortext_fusion/   adapters, io, stimulus_log (Lab exports), config, fusion, pipeline, evaluate, report,
                    mock, server (WebSocket for the app + HTTP for models), ws, describe, cli
  configs/          example.yaml (annotated), adapter_cookbook.yaml (one entry per output shape)
  docs/             evaluation_report_template.md
  tests/            pytest suite + e2e_lab_app.py (the real app in headless Chromium)
```

## Caveats

- **Mock data.** Mock results only show that the plumbing works.
- **Calibration and test sessions.** Calibrate and evaluate on different sessions. `evaluate` does this for you; `calibrate` does not, so pass `--exclude-sessions`.
- **Category model confound.** A category model that "works" may be decoding the prompt rather than the choice. `check_target` reports how often its top class equals the prompt's category. Run a hidden-prompt control before trusting it.
- **Live display.** The app shows `best_option_id` even when we mark the trial `unsure`. Treat a low confidence on screen as "unsure".

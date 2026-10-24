# CorText fusion layer

This package combines the **P300 detector** and the **category/imagery model** for the multiple-choice task ("Which one is the dog?" with options A–D flashing). Each trial gets one answer such as `EEG indicates B (dog), confidence 0.72` or `unsure`.

Neither model's output format had to be fixed before training. Instead, **each model gets an adapter**. The adapter converts whatever the model writes into one score per option, which is the fusion layer's standard internal format. You wire in a model by editing a YAML file. No code changes are needed.

Everything can be tested on **mock data** before the real models exist.

```
stimulus app marker log ─┐
P300 output (any shape) ─┼─ adapters ─> one score per option ─> normalize ─> weighted sum ─> prediction
category output (any)  ──┘                                     (per trial)   (calibrated)    + confidence
```

## Install

```bash
cd fusion
pip install -e ".[dev]"       # Python 3.9+; numpy, pandas, scipy, pyyaml, matplotlib
pytest -q                     # 63 tests, about 10 s
```

This folder is self-contained. It does not touch the repo's `requirements.txt` or notebooks.

## Two-minute demo (mock data)

```bash
cortext-fusion mock demo/                        # fake log + outputs in 11 formats + configs
cortext-fusion evaluate demo/mock_config.yaml    # held-out evaluation with controls
open demo/report/report.md                       # report + plots
cortext-fusion serve --mock                      # fake live predictions for the web app
```

Use `--p300-signal 0 --category-signal 0` to check that everything drops to chance, controls included.

## Wire in your model in under 10 minutes

**1. Save your outputs with trial IDs** (2 min). Use the **same `trial_id` strings as the stimulus app's marker log**. Any of these works:

| Your model produces | Save it as |
|---|---|
| a score per flash/epoch | CSV `trial_id, option_id, repeat_index, score` (one row per flash) |
| `predict_proba` per epoch | `.npy` `[trials, flashes, 2]` + `p300_trial_ids.txt` (one id per line), flashes in marker-log order |
| a score per option | JSON `{"P01_S1_T001": {"A": 1.2, "B": 3.4, ...}, ...}` |
| probabilities over categories | CSV `trial_id, p_dog, p_cat, ...` |
| an embedding per trial | `.npy` `[trials, dim]` + ids file + `{category: vector}` JSON |
| one file per trial | a folder of `<trial_id>.json` / `.npy` / `.csv` |

If you write a list without option IDs, its order **must** be the stimulus log's option order (`A, B, C, D` unless the trial table says otherwise). Including `option_id` is safer, because the adapter then reorders the values and flags any mismatch.

**2. Ask the tool what it sees** (1 min):

```bash
cortext-fusion describe outputs/p300.csv --markers data/markers.csv --trials data/trials.csv --name p300
```

This prints the shape, dtype, range and NaN count, checks your trial IDs against the log, ranks the adapters (and tests each one on your trials), and prints a config entry to paste.

**3. Paste that entry** into a copy of [`configs/example.yaml`](configs/example.yaml) (2 min). Every option is in [`configs/adapter_cookbook.yaml`](configs/adapter_cookbook.yaml). If higher values mean *less* likely (distances, losses), add `lower_is_better: true`.

**4. Run it** (3 min):

```bash
cortext-fusion evaluate configs/my_config.yaml
```

Read the **Diagnostics** list first. It flags unknown or missing trial IDs, option-order mismatches, NaNs, scores that are constant across options, a model that adds no information, and accuracy **below** chance (usually a flipped sign or a wrong option order).

### From the starter notebook (`p300_model.ipynb`)

`find_p300_peak(erp, times)` returns the peak amplitude of an averaged ERP. Average each option's epochs, take the amplitude, and save:

```python
import json
out = {trial_id: {opt: float(find_p300_peak(erp_by_option[opt], times)[1]) for opt in "ABCD"}
       for trial_id, erp_by_option in trials.items()}
json.dump(out, open("outputs/p300_peak_amplitude.json", "w"))
```

Then use `adapter: per_option`. If you score every single epoch instead, save the per-flash CSV and use `per_flash`, which also gives you accuracy against the number of repeats.

## Adapters

| Name | Input | Converted to a per-option score by |
|---|---|---|
| `per_option` | list `[sA..sD]`, dict by option ID or category, or `option_id` + `score` | used as is (reordered by ID) |
| `per_flash` | one score per flash: 1-D array in marker order, `[repeats, options]` grid, or long CSV | mean (or sum/median) per option |
| `probabilities` | one probability per option | logit (`transform: log` for a softmax); 0 and 1 are clipped safely |
| `epoch_target_prob` | P(target) per flash, or `predict_proba` `[flashes, 2]` | mean log-odds per option |
| `class_probabilities` | probabilities over many categories | looked up by each option's category label (`category_map` if names differ); `check_target` flags when the top class is just the prompt's category |
| `embedding` | vector per trial | cosine similarity to each option's category vector |
| `distance` | per-option distances or losses | negated (`lower_is_better: true` works on any adapter) |
| `auto` | anything above | detected on the first trial, then locked in (the log says which) |

The API is `adapter.to_scores(raw_output, trial_meta) -> list[float]`. To write your own, subclass `cortext_fusion.adapters.Adapter`, implement `_scores(raw, meta, max_repeat) -> (np.ndarray, extras_dict)`, and register it in `ADAPTERS`.

## Fusion and calibration

- Scores are normalized **per trial across options**: `zscore` by default, or `logsoftmax`, `softmax`, `center` or `none`. They are then combined as a weighted sum. Confidence is the softmax of the fused scores, and "unsure" means confidence is below `unsure_threshold`.
- **By default only the P300 counts** (category weight 0).
- `cortext-fusion calibrate cfg.yaml --exclude-sessions <test sessions>` learns the weights and temperature with conditional logistic regression on **validation** trials. A zero-weight model is switched on only if it lowers the **held-out** (leave-one-session-out) log-loss by `min_gain_logloss`. With a single session it stays off. The result is written to `calibration.yaml`, which `fuse`, `serve` and `watch` pick up.
- **Sequential fusion** (live demos): per-flash evidence is accumulated as a running log-score, and the trial stops early once confidence ≥ `stop_confidence` after at least `min_repeats` repeats.

## Evaluation (`cortext-fusion evaluate`)

Data is **held out by session or participant, never by random trials**. Trials marked invalid are skipped and counted. Every number is reported against both the **clicked** answer and the **true** answer.

Methods compared: each model alone, fused with the config weights, fused with weights calibrated on the *other* sessions, and the prior-only baseline.

Controls:
- **shuffled labels**: permutation p-value for every method
- **noise scores**: the whole pipeline, calibration included, re-run on Gaussian noise; it must land at chance
- **prior-only**: always answer the most common answer position from the training sessions

The `report/` folder contains:
- `report.md`, a filled one-page report (blank version in [`docs/evaluation_report_template.md`](docs/evaluation_report_template.md))
- `metrics.json`
- `predictions_heldout.csv`
- accuracy vs repeats, with an early-stopping simulation
- reliability plot and expected calibration error (ECE)
- confusion matrix
- `issues.txt`

## Live: server for the CorText Lab web app

```bash
cortext-fusion serve configs/my_config.yaml            # real models (via their adapters)
cortext-fusion serve --mock --mock-accuracy 0.8        # no models needed; responses carry "mock": true
```

The server listens on `http://127.0.0.1:8765` with CORS enabled.

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | | mode, models, weights |
| `POST /predict` | trial (below) + `"outputs": {"p300": <raw>, "category": <raw>}` | prediction |
| `POST /sequential/start` | trial | running prediction |
| `POST /sequential/flash` | `{trial_id, option_id, value, repeat_index}` (`value` = P(target); omit in mock mode) | running prediction + `stopped_early` |
| `POST /sequential/evidence` | `{trial_id, model, output}` or `{..., scores}` | running prediction |
| `POST /sequential/end` | `{trial_id}` | `{ok: true}` |

The trial body looks like this:

```json
{"trial_id": "P01_S1_T001",
 "options": [{"option_id": "A", "category": "cat"}, {"option_id": "B", "category": "dog"},
             {"option_id": "C", "category": "car"}, {"option_id": "D", "category": "tree"}],
 "target_category": "dog", "hint_option": "B"}
```

`hint_option` is used only by the mock to bias its fake answer.

A prediction looks like this:

```json
{"predicted_option": "B", "predicted_category": "dog", "confidence": 0.72, "margin": 0.51, "unsure": false,
 "probabilities": [...], "contributions": {"p300": 1.4, "category": 0.0}, "per_model_scores": {...},
 "warnings": [], "message": "EEG indicates B (dog), confidence 0.72", "mock": false}
```

To call it from the web app:

```js
const r = await fetch("http://127.0.0.1:8765/predict", {method: "POST",
  headers: {"Content-Type": "application/json"}, body: JSON.stringify({...trial, outputs})});
const pred = await r.json();   // show pred.message and draw bars from pred.probabilities
```

If the models write one file per trial instead, run `cortext-fusion watch cfg.yaml --folder p300=live/p300 --folder category=live/category --out live.jsonl`. It fuses each trial as soon as its files appear.

### Stimulus log contract

**Marker log**, one row per flash onset: `timestamp, trial_id, option_id, true_category, is_target, repeat_index, session_id[, participant_id]`.

**Trial table** (optional): `trial_id, option_order (A|B|C|D), target_category, target_option, clicked_option, valid, invalid_reason, prompt`.

Column names are matched loosely (`trial`, `subject`, `response`, ... also work). A JSON export `{"markers": [...], "trials": [...]}` loads too.

## Python API

```python
from cortext_fusion import load_config, load_dataset, Fuser, fuse_dataset, evaluate, Predictor, make_adapter
cfg = load_config("configs/my_config.yaml")
ds = load_dataset(cfg)                         # adapters applied; ds.issues has the diagnostics
preds = fuse_dataset(ds, Fuser.from_config(cfg))
res = evaluate(ds, cfg, out_dir="report")
make_adapter("per_flash").to_scores(raw_for_one_trial, ds.metas["P01_S1_T001"])
```

## Layout

```
fusion/
  cortext_fusion/   adapters, io, stimulus_log, config, fusion, pipeline, evaluate, report, mock, server, describe, cli
  configs/          example.yaml (annotated), adapter_cookbook.yaml (one entry per output shape)
  docs/             evaluation_report_template.md
  tests/            pytest suite (adapters, fusion, calibration, evaluation controls, server, CLI)
```

## Caveats

- Mock results show that the plumbing works, nothing more.
- Calibrate and evaluate on different sessions. `evaluate` does this for you; `calibrate` does not, so pass `--exclude-sessions`.
- A category model that "works" may be decoding the prompt rather than the choice. The `check_target` diagnostic reports how often its top class equals the prompt category. Test it with a hidden-prompt control before trusting it.

# CorText fusion: evaluation report (one page)

*`cortext-fusion evaluate <config>` fills most of this automatically in `report/report.md`. Use this blank copy when reporting by hand, or for a run that combines several reports.*

**Date:** ____  **Author:** ____  **Config / commit:** ____

## 1. Setup
- **Data source:** real recording / public dataset (name: ____) / **MOCK** (mock numbers say nothing about the brain)
- **Device and montage:** 32-node / 8-node / Muse; channels ____; sampling rate ____ Hz
- **Participants / sessions / trials:** __ / __ / __ (invalid trials skipped: __, with reasons ____)
- **Task:** __ options, __ repeats per option, flash __ ms on / __ ms off. Chance = ____
- **Held out by:** session / participant (leave-one-out). Never random trials.
- **Models:** P300 (adapter ____, weight ____), category (adapter ____, weight ____)
- **Ground truth:** calibration trained on clicked / true answer; both reported below.

## 2. Accuracy (held-out)
| Method | Acc. vs clicked (95% CI) | Acc. vs true answer | Shuffled-label p | Unsure rate |
|---|---|---|---|---|
| P300 only | | | | |
| Category only | | | | |
| Fused (config weights) | | | | |
| Fused (calibrated, held-out) | | | | |

## 3. Controls (each must be near chance)
| Control | Result | Pass? |
|---|---|---|
| Noise-score input (whole pipeline) | | |
| Shuffled labels (null mean) | | |
| Prior-only (most common answer position) | | |
| Low-level features only (luminance / position / timing) | | |

## 4. Repeats and early stopping
- Accuracy at 1 / 3 / 5 / max repeats: __ / __ / __ / __ (see `accuracy_vs_repeats.png`)
- Early stopping at confidence ≥ __: mean __ of __ repeats used, accuracy __

## 5. Calibration and errors
- Expected calibration error: ____ (see `reliability.png`)
- Confusion pattern (is any option position over-predicted?): ____
- Calibrated weights (P300 / category): ____. Category model kept in __ of __ folds.

## 6. Diagnostics raised
- [ ] trial-ID mismatches  [ ] option-order mismatches  [ ] NaNs  [ ] constant scores  [ ] model with no information  [ ] below chance (flipped sign?)
- Notes: ____

## 7. Verdict
- Fused beats P300 only on held-out data, beyond the CI? **yes / no**
- All controls at chance? **yes / no**
- **Go / no-go** for the next stage, and the reason: ____

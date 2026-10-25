# CorText P300 dataset

EEG recorded with CorText Lab (multiple-choice P300 task; mostly a g.tec Unicorn / BCI Core-8, 8 channels
Fz C3 Cz C4 Pz PO7 POz PO8, 250 Hz). The pipeline (`cortext-lab/pipeline/build_session.py`) adds sessions here
automatically. **The recordings are not in git**: they live in a private Hugging Face dataset and in `data/cortext-p300/`
on each machine. Get them with `python datasets/cortext-p300/cortext_store.py pull`; setup and storage decisions are in
`docs/DATA_STORAGE.md`, how to record in `docs/DATA_COLLECTION.md`.

## Load it

```python
import sys; sys.path.append("datasets/cortext-p300")
from cortext_data import load_epochs, list_sessions, load_continuous, to_mne
X, y, info = load_epochs()                       # real sessions, valid trials, artifact-free epochs
X, y, info = load_epochs(bucket="final")         # only sessions labelled with that bucket
# sessions the operator redid in the app ("superseded") are skipped; add include_superseded=True to keep them
X, y, info = load_epochs(channels=["Pz", "POz", "Cz"], tmin=0.0, tmax=0.8)
epochs = to_mne(X, info, y)                      # MNE EpochsArray (event 1 = target)

from cortext_data import load_imagery             # sessions recorded with the imagery window (others are skipped)
Xi, yi, info_i = load_imagery()                  # yi = concept the participant pictured ("dog", ...); split by info_i["groups"]
```

**Evaluate honestly.** Always split by `info["groups"]` (session:trial) with `GroupKFold`, so epochs from the same
trial never sit in both train and test. Report answer accuracy per trial (`trial_option_scores`) against chance
(1 / number of options). Also run a shuffled-label control. `cortext-lab/pipeline/train_p300.py` does all of this.

## Layout (under `data/cortext-p300/`, or `$CORTEXT_DATA_DIR`)

```
index.csv                      one row per local session (quality summary; rebuilt automatically)
sessions/<session_id>/         real recordings
simulated/<session_id>/        fake_eeg.py runs (never mix into results)
  meta.json      participant code, device, stream, settings, every cleaning parameter, quality report
  eeg_clean.npz  data (channels x samples, float32 µV, cleaned), times (LSL s), srate, labels, segment, artifact mask
  eeg_raw.npz    data, times, srate, labels: untouched recording
  epochs.npz     X (n, channels, times) cleaned + baseline-corrected, times (s), y (1 = attended answer's flash),
                 trial_id, trial_index, option_id, option_position, category, repeat_index, flash_index,
                 mode (flash | sequential), trial_valid, rejected_artifact, clicked_option_id, onset_lsl, session_id
  imagery.npz    only if the session used the imagery window (Experiment → Design → Imagery window > 0): one epoch per
                 trial, -200 ms .. imagery_ms after imagery_on, baseline -200..0 ms; X, times, labels, srate, trial_id,
                 option_id + category (the correct answer the participant was asked to picture), trial_valid,
                 rejected_artifact, clicked_option_id, onset_lsl, session_id
  events.csv     every marker (lsl_time, sample index into eeg_clean, event, trial/option fields)
  trials.csv     the app's trial table (validity after operator review, click, correct, model prediction)
voltages/<session_id>_raw.csv.gz   plain-text copy of eeg_raw.npz for people without numpy: time_s, lsl_time, one µV column
                                   per electrode (Fz_uV ... PO8_uV), event (markers on that sample). Uploaded with each session
                                   ("upload_voltages" in dataset.json); for older sessions: cortext_store.py voltages;
                                   download: cortext_store.py pull --voltages
```

## Cleaning (automatic)

1. Gap splitting and an exact 250 Hz grid.
2. Notch at the mains frequency (auto-detected 50/60 Hz) and its 2nd harmonic.
3. Zero-phase band-pass 0.5–40 Hz.
4. Flat or noisy electrodes detected and rebuilt from their neighbours.
5. Artifact spans (> 150 µV) flagged; epochs touching them get `rejected_artifact`.

The details and the reasons for each step are in `docs/DATA_COLLECTION.md` and in each session's `meta.json["processing"]`.

## Caveats

- **Labels.** `y` = flashes of the answer the participant was asked to find (the true answer). It is not proof that they attended it on every flash. Use `clicked_option_id` and `trial_valid` to filter.
- **Overlapping epochs.** With the default flash timing, consecutive epochs overlap (stimulus onset asynchrony < 800 ms).
- **Sequential-mode trials** have far fewer epochs per option.
- **Privacy.** Participant IDs must be codes (P01), never names. Session notes are included in `meta.json`.

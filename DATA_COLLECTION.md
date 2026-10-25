# Collecting EEG data with CorText (start here)

Headset → website → cleaned dataset on GitHub → teammates train. Three terminals, one browser tab.

## 0. One-time install (in the VS Code terminal, repo folder)

```bash
pip install "gpype[lsl]" pylsl numpy scipy
```

(`gpype` is what `bci.py` already uses. If `bci.py` runs, you only need `pip install pylsl` on top.)

## 1. Every recording

Close `bci.py` first: only one program can hold the Bluetooth headset at a time.

| Terminal | Command | What it does |
|---|---|---|
| 1 | `python cortext-lab/eeg/stream_eeg.py` | Connects to the Unicorn / BCI Core-8 and streams it to LSL. Opens the same scope window as `bci.py`. |
| 2 | `python cortext-lab/bridge/bridge.py` | Finds the headset stream, records it, scores the P300 live, and serves the website. |
| browser | http://localhost:8000 | The website. |

No headset? Run `python cortext-lab/eeg/fake_eeg.py` instead of terminal 1. It sends **simulated** EEG, labelled SIMULATED everywhere and stored apart from real data.

## 2. In the website

1. **Setup page → Headset card.** It should show the headset, 8 electrode chips and live traces. It auto-fills *Device* and sets *Predictor = Live EEG*.
   - Fix any electrode marked **noisy** or **flat** before you start: add gel, press it on, ask the participant to relax their jaw and sit still.
   - Healthy is about 5–40 µV.
   - Your first screenshot (traces clipping past ±50 µV) would show most chips as noisy.
2. Optional: click **Open neural monitor ↗** and put it on a second screen for the audience. Never show it to the participant during flashing.
3. Enter a **Participant ID**. Use a code like P01, never a real name, because this ends up on GitHub. Then click **Start session**.
4. When the session ends, open **Session summary**. The *Training dataset* card cleans the EEG and shows a quality report. **Publish to GitHub** commits and pushes only that session's folder.
5. If you mark trials invalid later on the Review page, the dataset is rebuilt automatically. Press Publish again to share the update.

## 3. For teammates training models

```bash
git pull
```

```python
import sys; sys.path.append("datasets/cortext-p300")
from cortext_data import load_epochs
X, y, info = load_epochs()     # X: (epochs, 8 channels, 250 samples) µV, y: 1 = attended answer
# use info["groups"] with GroupKFold so a trial is never split across train and test
```

Then:
- Train the live scorer and check it with honest controls: `python cortext-lab/pipeline/train_p300.py`
- Use the trained model live: `python cortext-lab/bridge/bridge.py --p300-model p300_model.npz`

## 4. How the automatic cleaning works

It runs in `cortext-lab/pipeline/build_session.py` when a session ends. Every parameter is written into each session's `meta.json`.

| Step | What | Tool / setting | Why |
|---|---|---|---|
| 1 | Split at Bluetooth gaps > 0.1 s; resample each piece onto an exact 250 Hz grid | linear interpolation (`numpy.interp`) | Filters assume evenly spaced samples. A gap must never be bridged by an epoch. |
| 2 | Detect mains hum (50 or 60 Hz) from the spectrum and notch it plus its 2nd harmonic | Welch spectrum; `scipy.signal.iirnotch` Q=30, zero-phase `filtfilt` | Removes power-line noise. It picks 60 Hz in the US automatically. |
| 3 | Band-pass 0.5–40 Hz | 4th-order Butterworth, zero-phase (`sosfiltfilt`) | Removes slow electrode drift (< 0.5 Hz) and muscle/high-frequency noise (> 40 Hz). Zero-phase means no time shift of the P300. |
| 4 | Find bad electrodes: flat (std < 0.5 µV) or noisy (std > 100 µV or > 4× the median channel) | robust thresholds | Loose or dry electrodes. |
| 5 | Rebuild bad electrodes from neighbours | inverse-distance weighting on 10-20 positions | Keeps the 8-channel shape. Listed in `meta.json` → `interpolated_channels`. |
| 6 | Flag artifacts (blinks, movement): any good channel > 150 µV, ±0.2 s padding | amplitude threshold | Epochs touching a flagged span get `rejected_artifact = True`. They are **kept, not deleted**, so you can choose. |
| 7 | Cut epochs −200…+800 ms around every flash; baseline-correct on −200…0 ms | | Ready-to-train P300 epochs |

The **raw recording is saved too** (`eeg_raw.npz`), so anyone can redo the cleaning differently.

What this cleaning does **not** do:
- **No ICA or blink regression.** With 8 channels and no EOG electrode it is unreliable, so blink epochs are flagged rather than "repaired".
- **No re-referencing.** The data stays in the Unicorn's own reference.

The **live** scorer uses a separate, causal 0.5–20 Hz filter. A live system can't look into the future, and `train_p300.py` reproduces those live features exactly.

Simulated sessions go to `datasets/cortext-p300/simulated/` and `load_epochs()` skips them unless `include_simulated=True`.

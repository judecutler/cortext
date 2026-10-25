# Trained P300 models

| Folder | What goes here | In git? |
|---|---|---|
| `library/` | Every trained model the team shares. `train_p300.py` saves here by default. | yes |
| `active/` | **The one model the bridge runs.** Exactly one `.npz`. | yes |
| `../bridge/logs/p300_adaptive_*.npz` | Models calibrated live on this laptop (the bridge saves one after each retrain). | no |

## Train and share a model
```
python cortext-lab/pipeline/train_p300.py              # -> models/library/P01_20261025-1530.npz
git add cortext-lab/models/library && git commit -m "P300 model for P01" && git push
```
Add `--activate` to also make it the running model (copies it into `active/`).

## No calibration yet? The "p3000 function"
`library/p3000_function.npz` is a fixed spike detector, not a trained model: for each picture it takes the mean voltage
240-480 ms after it appeared over Cz/Pz/POz/PO7/PO8, adds it up over repeats, and the picture with the biggest spike is
the answer. Same for every person, works from the first trial, needs the default filter chain and the 8-channel Unicorn
montage. Rebuild it with `python cortext-lab/pipeline/make_p3000_function.py` (`--activate` to make it the running
model). A calibrated model should beat it once you have one; percentages are deliberately soft and uncalibrated.
**One at a time only:** it needs each picture to stay alone for at least ~0.8 s (the demo's 1 s + 0.3 s gap). In flash
mode the next flash lands inside the 240-480 ms window and gets the credit: on simulated EEG it is then at chance (3/14).
`models/active/` ships with it, so a fresh checkout decodes from trial 1.

## Choose which model runs
- **On the website (easiest):** Agent → *Model* block lists the library and this laptop's calibrations, with who each
  was trained on, how many epochs, cross-validated accuracy and whether it fits the connected headset. **Use this**
  switches the running model immediately (no restart) and copies it into `active/`. It also switches the filters to
  the chain the model was trained with (see [`../filters/README.md`](../filters/README.md)); Undo switches both back.
- **By folder:** put one `.npz` in `active/`; the bridge loads it at startup.
- **By flag:** `bridge.py --p300-model FILE` overrides `active/`; `--p300-model none` ignores it (adapt live instead).

## Rules
- **A model only works for the person it was trained on, with the same electrodes.** Names use participant codes
  (`P01_…`), never real names. Check the Agent page shows no participant warning before a demo.
- Models trained on simulated EEG get `_SIM` in their name and a SIMULATED tag on the Agent page.
- **A model only runs with the filter chain it was trained with.** Each file stores its chain; files from before chains
  existed count as the default 0.5–20 Hz chain. If someone applies different filters, the model pauses (the Agent map
  shows a broken red wire) until its chain runs again.
- A model is a few KB of weights derived from one person's EEG. Only share models of people who agreed.
- Picking a model on the website changes `active/` on that computer. Before `git pull` there, either commit that
  change or run `git checkout -- cortext-lab/models/active` to drop it.

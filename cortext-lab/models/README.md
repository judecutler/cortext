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

## Choose which model runs
- **On the website (easiest):** Setup → *P300 model* lists the library and this laptop's calibrations, with who each
  was trained on, how many epochs, cross-validated accuracy and whether it fits the connected headset. **Use this**
  switches the running model immediately (no restart) and copies it into `active/`.
- **By folder:** put one `.npz` in `active/`; the bridge loads it at startup.
- **By flag:** `bridge.py --p300-model FILE` overrides `active/`; `--p300-model none` ignores it (adapt live instead).

## Rules
- **A model only works for the person it was trained on, with the same electrodes.** Names use participant codes
  (`P01_…`), never real names. Check the Setup page says the right participant before a demo.
- Models trained on simulated EEG get `_SIM` in their name and a SIMULATED tag on the Setup page.
- A model is a few KB of weights derived from one person's EEG. Only share models of people who agreed.
- Picking a model on the website changes `active/` on that computer. Before `git pull` there, either commit that
  change or run `git checkout -- cortext-lab/models/active` to drop it.

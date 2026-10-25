# Filter chains

A filter chain is the list of signal-cleaning layers between the headset and the P300 model. Every incoming EEG
sample goes through it, in order, before it is cut into epochs. Chains are small `.json` files, shared like models.

| Folder | What goes here | In git? |
|---|---|---|
| `library/` | Every chain the team shares. The Agent page's **Save draft to library** writes here. | yes |
| `active/` | **The one chain the bridge runs.** Empty = the built-in default (0.5–20 Hz band-pass, order 2). | yes |

## Layers

| Type | Settings | What it does |
|---|---|---|
| `bandpass` | `low`, `high` (Hz), `order` | Keeps one band: drift below `low` and muscle / mains noise above `high` are removed. |
| `highpass` | `cutoff`, `order` | Removes slow drift. |
| `lowpass` | `cutoff`, `order` | Removes fast activity. |
| `notch` | `freq` (50 / 60), `q`, `harmonics` | Cuts power-line hum (60 Hz in the US) and optionally its harmonics. |
| `reref` | `channels` (empty = all) | Subtracts the average of those channels from every channel (common average when empty). |

Each layer can be `"enabled": false` (bypassed: kept with its settings, does nothing). `reject_uv` is the epoch
artifact limit: an epoch whose filtered signal goes beyond ± this is skipped.

```json
{"name": "P300 + 60 Hz notch", "reject_uv": 100,
 "layers": [{"type": "notch", "freq": 60, "q": 30, "harmonics": 1},
            {"type": "bandpass", "low": 0.5, "high": 20, "order": 2}]}
```

## Rules
- **Every layer is causal** (it only uses past samples) and keeps its state between chunks, so the live bridge and
  `pipeline/train_p300.py` produce identical features. The offline dataset cleaning (`build_session.py`) is zero-phase
  and separate: it is for analysis and artifact flags, not for the live scorer.
- **A model only runs with the chain it was trained with.** A trained model stores its chain. The bridge compares the
  enabled layers and their settings (names and `reject_uv` don't count); on a mismatch the model is paused and the scorer
  falls back to calibrating on the session. Models saved before chains existed count as the default chain.
- Picking a model on the Agent page switches the filters to its chain. Applying a different chain rebuilds the
  session's live calibration from the last 90 s of EEG (re-filtered).
- **Artifact threshold and forehead channels:** `reject_uv` applies to every channel except Fp*/AF* when the headset
  has others (a Muse's AF7/AF8 see every blink; its P300 channels are TP9/TP10). The Unicorn has none, so nothing changes there.
- `library/muse-p300.json` is the chain for the Muse (60 Hz notch + 0.5–20 Hz, 150 µV); the Muse p3000 function is built for it.
- To compare chains honestly, train one model per chain on the same sessions
  (`train_p300.py --filters p300-60-hz-notch`) and compare held-out accuracy and the shuffled control.

## Choose which chain runs
- **On the website:** Agent → System map. Edit the draft (+ to add, drag to reorder, ● to bypass), watch the signal
  after every layer, then **Apply to live**. Or press **Use this** on a chain in the Filter library.
- **By folder:** put one `.json` in `active/`; the bridge loads it at startup.
- **By flag:** `bridge.py --filters FILE` overrides `active/`; `--filters default` ignores it.

Applying a chain on the website changes `active/` on that computer. Before `git pull` there, either commit that change
or run `git checkout -- cortext-lab/filters/active` to drop it.

---
license: other
pretty_name: CorText P300 EEG
tags: [eeg, bci, p300]
---
# CorText P300 EEG (private research dataset)

Multiple-choice P300 recordings from CorText Lab: 8 channels (Fz C3 Cz C4 Pz PO7 POz PO8), 250 Hz, g.tec Unicorn / BCI Core-8.

One folder per session under `sessions/<session_id>/`: `meta.json`, `trials.csv`, `events.csv`, `epochs.npz`, `eeg_clean.npz`, `eeg_raw.npz`.

## Just the voltages: `voltages/`

One file per session, `voltages/<session_id>_raw.csv.gz`, readable without any CorText code (pandas, Excel after unzipping, R...).
One row per sample (250 per second):

| column | meaning |
|---|---|
| `time_s` | seconds since the first sample of the session |
| `lsl_time` | the recording clock; the same clock as `lsl_time` in `events.csv` |
| `unix_time` | absolute time, seconds since 1970 UTC (sessions built from 2026-10-25 on; from the page's wall clock at each marker) |
| `Fz_uV` … `PO8_uV` | voltage at each electrode, microvolts, rounded to 0.01 µV |
| `event` | what happened on screen at that sample, e.g. `flash_on option=dog target=1` (`target=1` = the correct answer); empty otherwise |

```python
import pandas as pd
df = pd.read_csv("voltages/<session_id>_raw.csv.gz")
flashes = df[df.event.str.contains("flash_on", na=False)]
```

`raw` is the untouched recording: it includes each electrode's DC offset (thousands of µV) and mains hum, so subtract
each column's mean and filter (e.g. 0.5–40 Hz) before looking for brain responses. About 5 MB per 10 minutes.

## Raw EEG that is not a session: `raw_eeg/`

`raw_eeg/<YYYYMMDD-HHMMSS>_<name>` plus `raw_eeg/<same>.json` (participant code, note, size, sha256, device, sample
rate). Two kinds: free recordings made with the app's *Record raw EEG* button (`…_raw.csv.gz`: `unix_time`, `lsl_time`,
`Fz_uV` … `PO8_uV`, one row per sample, untouched: DC offset and mains hum included, no markers), and files from other
recorders uploaded as they were (CSV, XDF, EDF, FIF…; read them with that recorder's tools).

Participant IDs are codes (P01), never names. Do not share outside the team without consent.

Load it with `datasets/cortext-p300/cortext_data.py` from the code repository (`pull`, then `load_epochs()`).

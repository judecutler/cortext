---
license: other
pretty_name: CorText P300 EEG
tags: [eeg, bci, p300]
---
# CorText P300 EEG (private research dataset)

Multiple-choice P300 recordings from CorText Lab: 8 channels (Fz C3 Cz C4 Pz PO7 POz PO8), 250 Hz, g.tec Unicorn / BCI Core-8.

One folder per session under `sessions/<session_id>/`: `meta.json`, `trials.csv`, `events.csv`, `epochs.npz`, `eeg_clean.npz`, `eeg_raw.npz`.
Participant IDs are codes (P01), never names. Do not share outside the team without consent.

Load it with `datasets/cortext-p300/cortext_data.py` from the code repository (`pull`, then `load_epochs()`).

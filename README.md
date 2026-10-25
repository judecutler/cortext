# cortext
Attempting to build brain to text translation. 

**Recording EEG? Start with [`docs/DATA_COLLECTION.md`](docs/DATA_COLLECTION.md)**: headset → website → cleaned dataset in a private
Hugging Face repo ([`docs/DATA_STORAGE.md`](docs/DATA_STORAGE.md)) → `cortext_store.py pull` and train
([`datasets/cortext-p300/`](datasets/cortext-p300/README.md)).

## CorText Lab (experiment + demo app)

[`cortext-lab/`](cortext-lab/README.md) is a browser-based experiment for recording P300 / attention data: a
multiple-choice question whose options flash while EEG is recorded, with the participant's click as ground truth.
Flash markers go to LSL (and a CSV backup) through a small stdlib-only Python bridge.

```bash
python cortext-lab/bridge/bridge.py     # open http://localhost:8000
```

See [`cortext-lab/README.md`](cortext-lab/README.md) for the marker schema, [`TEST_PLAN.md`](cortext-lab/TEST_PLAN.md)
and the pre-session [`TIMING_CHECKLIST.md`](cortext-lab/TIMING_CHECKLIST.md).

## Repo layout

| Folder | What |
|---|---|
| `cortext-lab/` | The experiment app (`web/`), bridge, EEG streaming (`eeg/`), cleaning/training pipeline, tests |
| `datasets/cortext-p300/` | Dataset loader and storage tool (data itself is **not** in git) |
| `fusion/` | Self-contained fusion experiments |
| `docs/` | Recording guide, storage decision |
| `notebooks/` | Starter notebook + its pinned environment |
| `PaperModel.ipynb` | Paper reproduction (left at the root so old links keep working) |
| `legacy/` | The original PsychoPy script |

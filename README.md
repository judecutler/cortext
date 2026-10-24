# cortext
Attempting to build brain to text translation. 

## CorText Lab (experiment + demo app)

[`cortext-lab/`](cortext-lab/README.md) is a browser-based experiment for recording P300 / attention data: a
multiple-choice question whose options flash while EEG is recorded, with the participant's click as ground truth.
Flash markers go to LSL (and a CSV backup) through a small stdlib-only Python bridge.

```bash
python cortext-lab/bridge/bridge.py     # open http://localhost:8000
```

See [`cortext-lab/README.md`](cortext-lab/README.md) for the marker schema, [`TEST_PLAN.md`](cortext-lab/TEST_PLAN.md)
and the pre-session [`TIMING_CHECKLIST.md`](cortext-lab/TIMING_CHECKLIST.md).

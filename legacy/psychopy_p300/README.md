# Legacy: first PsychoPy P300 script

`p300_stimulus.py` is the original stand-alone stimulus program (PsychoPy + LSL) that came before CorText Lab.
It is kept for reference only; new recordings should use the web app in `cortext-lab/`.

```bash
python legacy/psychopy_p300/p300_stimulus.py --participant P01
```

Pictures are read from `images/` next to the script (a blue square is shown for a missing file). Marker CSVs are
written to `legacy/psychopy_p300/data/` (git-ignored).

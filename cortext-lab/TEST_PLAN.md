# Test plan

## A. Automated (about 1 minute)

```bash
node --test tests/logic.test.mjs      # scheduler on a simulated 60 Hz clock, flash rules, trial/validity logic, CSV, ZIP
python tests/test_bridge.py           # bridge: static files, WebSocket handshake, clock sync, markers -> CSV
python tests/test_bridge_lsl.py       # needs pylsl: marker arrives on a real LSL inlet with a corrected timestamp
python tests/e2e.py                   # needs playwright + chromium: full session in headless Chrome (screenshots in tests/screenshots/)
python tests/e2e_interrupts.py        # pause mid-flash, hidden tab, bridge killed + restarted mid-session, reload keeps data
```

What they establish: each option flashes exactly `repeats` times and never twice in a row; flash onsets stay within
one frame of schedule on a simulated display and the highlight is always off after an abort; every `flash_on` marker
carries session/trial/option/category/is_target/repeat and reaches IndexedDB, the bridge CSV and (with pylsl) LSL
with no sequence gaps; discard/redo/early-click/export/review behave and never delete data; the timing page runs.

## B. Manual, before the first real session (about 10 minutes)

1. `python bridge/bridge.py`, open the page. Top bar: bridge **connected**; LSL **on** if pylsl is installed.
2. Setup with a test participant, 3 questions, 10 repeats. Start. Check the stage reads clearly from the participant's
   seat (tile size, contrast), at the real display and browser size.
3. Trial 1: press `Space`; confirm only one tile is lit at a time, no animations, no repeats back-to-back; during flashing
   click a tile (nothing happens on screen; dev panel "Early clicks" goes to 1); after flashing click the answer.
4. Post-trial: feedback shows; "EEG indicates … [MOCK …]" is visible; press `Backspace` → "marked INVALID", choose a reason.
5. Trial 2: press `R` during flashing → flashing stops, same question comes back as a retry (dev list shows `redo`).
6. Switch browser tab mid-flash → trial aborts as `tab_hidden`; return, press `Space` to resume.
7. `P` mid-trial → aborted and re-queued; overlay "Paused"; `P` resumes.
8. Press `M`: Demo mode hides the dev panel and enlarges the readout; `M` again restores it.
9. Kill the bridge mid-session: indicator turns red, session continues; markers get `bridge_sent=false`; restart the
   bridge and confirm the page reconnects. Close and reopen the tab: Sessions page still lists everything.
10. Finish. Summary: counts, click accuracy, discard rate. Review: mark a trial invalid, restore it. Export zip; open
    `markers.csv` and check `flash_on` rows (is_target true for the correct option only).
11. With LabRecorder: confirm `CorTextMarkers` appears, record 1 trial, and check the XDF contains the markers
    (`LSL recorder attached` turns green while recording).
12. Run `TIMING_CHECKLIST.md`.

## C. Not covered (needs real hardware/people)

Real EEG alignment and photodiode lag, 8/32-node and Muse streaming, long sessions (30+ min) on the lab laptop, other
browsers/OSes, high-refresh or variable-refresh displays, touch screens.

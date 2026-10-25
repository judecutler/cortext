# Timing checklist (do before every real session, and after any change to computer, display, browser or OS)

Goal: know the constant lag between the marker and the light on the screen, and confirm the jitter is small.
Use the **same** computer, display, resolution, browser window/fullscreen state and power settings as the session.

## Prepare
- [ ] Plug in power; disable battery saver / low-power mode; close other tabs and heavy apps; no screen recording.
- [ ] Use Chrome/Edge, one window, visible (not minimised or behind another window). Prefer fullscreen (`Shift+F`).
- [ ] Browser hardware acceleration ON. Display at its native refresh rate (60 Hz is fine; note it). Turn off display
      "motion smoothing", adaptive/variable refresh and night-light features.
- [ ] `python bridge/bridge.py` running; page shows bridge **connected** (LSL on if you will record markers).

## 1. Software timing (2 minutes) — **Timing** tab
Interval 1000 ms, duration 500 ms, 60 flashes, photodiode square on.
- [ ] Display refresh matches the monitor (e.g. 59.9–60.1 Hz).
- [ ] **Max onset error ≤ 1.5 frames** (≤ 25 ms at 60 Hz) — PASS shown.
- [ ] **Dropped frames = 0** (if not: close apps, plug in power, try again; a few drops mean the machine is too busy).
- [ ] **Markers sent 60/60**, bridge received 60.
- [ ] Sync RTT on localhost is small (typically < 2 ms). If > 5 ms, something is wrong (CPU load, wrong host).
- [ ] Repeat once at the exact interval/duration you will use (e.g. interval 650 ms, duration 500 ms).

## 2. Hardware timing (10 minutes) — marker → light
- [ ] Photodiode taped over the corner square (default bottom-left; choose the corner in the page), wired to an EEG
      aux/trigger channel or a second LSL stream. (Phone slow-motion video of screen + a visible LED on the amplifier
      is a rough fallback.)
- [ ] LabRecorder saves **CorTextMarkers** and the photodiode/EEG stream. Run the timing page for 60 flashes.
- [ ] Extract photodiode onsets (threshold crossing) into a CSV with an `lsl_timestamp` column, then
      `python tools/analyze_timing.py bridge/logs/markers_….csv --event timing_flash --photodiode pd_onsets.csv`
- [ ] Record: **mean lag = ____ ms, SD = ____ ms** (display, refresh, browser, date: ____).
- [ ] SD ≤ about one frame (≈ 8–17 ms at 60 Hz); ideally < 5 ms. Mean lag is stable (±2 ms) across two runs.
- [ ] Write the mean lag into the session notes; subtract it (or use the photodiode onsets) when epoching.

## 3. In-session sanity (every session)
- [ ] Session start shows display refresh in the dev panel (matches expectation).
- [ ] After the first trial: dev panel Timing shows onset error p95 within one frame, dropped frames 0.
- [ ] EEG indicator green (stream found, samples flowing) and **LSL recorder attached** green before pressing Start.
- [ ] After the session: summary says all `flash_on` markers were delivered to the bridge; export the zip and keep
      the bridge CSV.

## If something fails
Dropped frames or large onset errors → power/CPU, background tabs, external display scaling; try fullscreen or a
different browser profile. Sync RTT large → bridge on another machine or Wi-Fi: run it on the stimulus computer.
Large constant lag is fine if it is stable (correct for it); large **jitter** is not (fix the setup or the display).

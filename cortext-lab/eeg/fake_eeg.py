#!/usr/bin/env python3
"""SIMULATED EEG for testing the whole chain without the headset.

    python cortext-lab/eeg/fake_eeg.py                 # 8 channels like the Unicorn, 250 Hz, LSL name CorTextEEG-FAKE
    python cortext-lab/eeg/fake_eeg.py --p300-uv 0     # no P300 at all (predictions should sit at chance)
    python cortext-lab/eeg/fake_eeg.py --muse          # 4 channels like a Muse (TP9 AF7 AF8 TP10), 256 Hz, + Right AUX

It listens to the app's CorTextMarkers LSL stream and adds a P300-shaped bump ~350 ms after every flash of the TRUE
answer (it reads is_target from the marker, which a real brain obviously doesn't). So any accuracy you see with this
stream is NOT evidence; the app and bridge label it "simulated" everywhere. Use it to check wiring, timing and the UI.
"""
import argparse
import json
import math
import random
import threading
import time

import numpy as np
import pylsl

LABELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "POz", "PO8"]
P300_WEIGHT = np.array([0.3, 0.5, 0.8, 0.5, 1.0, 0.6, 0.9, 0.6])   # parietal-central maximum
VEP_WEIGHT = np.array([0.1, 0.2, 0.2, 0.2, 0.5, 0.9, 1.0, 0.9])    # occipital visual response to every flash
ALPHA_WEIGHT = np.array([0.2, 0.3, 0.3, 0.3, 0.7, 1.0, 1.0, 1.0])
BLINK_WEIGHT = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])   # 8 ch: blinks on Fz only (unchanged)
# Muse (referenced to Fpz): the P300 shows at the temporo-parietal TP9/TP10, blinks dominate AF7/AF8.
# The 5th channel "Right AUX" is an unconnected input (muselsl streams it); the bridge drops channels named AUX.
MUSE = {"labels": ["TP9", "AF7", "AF8", "TP10", "Right AUX"],
        "p300": np.array([1.0, 0.3, 0.3, 1.0, 0.0]), "vep": np.array([0.4, 0.1, 0.1, 0.4, 0.0]),
        "alpha": np.array([1.0, 0.3, 0.3, 1.0, 0.0]), "blink": np.array([0.1, 1.0, 1.0, 0.1, 0.0])}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="CorTextEEG-FAKE")
    ap.add_argument("--srate", type=float, default=250.0)
    ap.add_argument("--p300-uv", type=float, default=5.0, help="P300 peak amplitude in microvolts (0 = none)")
    ap.add_argument("--noise-uv", type=float, default=8.0, help="background noise std in microvolts")
    ap.add_argument("--marker-stream", default="CorTextMarkers")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--muse", action="store_true", help="simulate a Muse: TP9 AF7 AF8 TP10 + Right AUX, 256 Hz")
    ap.add_argument("--no-aux", action="store_true", help="with --muse: leave out the Right AUX channel")
    ap.add_argument("--wall-clock", action="store_true",
                    help="stamp samples with time.time() like muselsl without --lsltime (the bridge must correct it)")
    args = ap.parse_args()
    global LABELS, P300_WEIGHT, VEP_WEIGHT, ALPHA_WEIGHT, BLINK_WEIGHT
    if args.muse:
        k = 4 if args.no_aux else 5
        LABELS = MUSE["labels"][:k]
        P300_WEIGHT, VEP_WEIGHT, ALPHA_WEIGHT, BLINK_WEIGHT = (MUSE[x][:k] for x in ("p300", "vep", "alpha", "blink"))
        if args.name == "CorTextEEG-FAKE":
            args.name = "Muse-FAKE"
        if args.srate == 250.0:
            args.srate = 256.0
    rng = np.random.default_rng(args.seed)

    info = pylsl.StreamInfo(args.name, "EEG", len(LABELS), args.srate, "float32", f"fake-{random.randint(0, 1 << 30)}")
    chans = info.desc().append_child("channels")
    for l in LABELS:
        c = chans.append_child("channel")
        c.append_child_value("label", l)
        c.append_child_value("unit", "uV")
        c.append_child_value("type", "EEG")
    info.desc().append_child_value("simulated", "true")
    info.desc().append_child_value("manufacturer", "CorText fake_eeg.py (simulated)")
    outlet = pylsl.StreamOutlet(info, chunk_size=10)

    events = []          # (onset_lsl_s, is_target)
    lock = threading.Lock()

    def marker_listener():
        inlet = None
        while True:
            if inlet is None:
                found = pylsl.resolve_byprop("name", args.marker_stream, timeout=1.0)
                if not found:
                    continue
                inlet = pylsl.StreamInlet(found[0], processing_flags=pylsl.proc_clocksync)
                print(f"listening to markers from '{args.marker_stream}'")
            try:
                s, ts = inlet.pull_sample(timeout=0.5)
            except Exception:
                inlet = None
                continue
            if not s:
                continue
            try:
                m = json.loads(s[0])
            except ValueError:
                continue
            if m.get("event") == "flash_on":
                with lock:
                    events.append((ts, bool(m.get("is_target"))))
                    del events[:-400]

    threading.Thread(target=marker_listener, daemon=True).start()

    n_ch = len(LABELS)
    dc = rng.uniform(-3000, 3000, n_ch)           # electrode offsets, like a real raw stream
    alpha_phase = rng.uniform(0, 2 * math.pi, n_ch)
    pink = np.zeros(n_ch)
    lag = 0.15                                     # generate samples 150 ms behind real time so markers have arrived
    t0 = pylsl.local_clock()
    n = 0
    print(f"SIMULATED EEG streaming as '{args.name}' ({n_ch} ch, {args.srate:g} Hz). Ctrl+C to stop.")
    try:
        while True:
            now = pylsl.local_clock()
            target_n = int((now - lag - t0) * args.srate)
            if target_n <= n:
                time.sleep(0.01)
                continue
            k = target_n - n
            ts = t0 + (n + np.arange(k)) / args.srate
            with lock:
                ev = [e for e in events if e[0] > ts[0] - 1.0]
            x = np.empty((k, n_ch))
            for i, t in enumerate(ts):
                pink = 0.97 * pink + rng.normal(0, args.noise_uv * 0.25, n_ch)
                v = dc + pink + rng.normal(0, args.noise_uv * 0.5, n_ch)
                v += 6 * ALPHA_WEIGHT * np.sin(2 * math.pi * 10 * t + alpha_phase)
                v += 3 * np.sin(2 * math.pi * 60 * t)      # mains hum
                for onset, is_t in ev:
                    dt = t - onset
                    if 0 < dt < 0.9:
                        v += 3 * VEP_WEIGHT * (math.exp(-((dt - 0.1) / 0.03) ** 2) - 0.8 * math.exp(-((dt - 0.17) / 0.04) ** 2))
                        if is_t and args.p300_uv:
                            v += args.p300_uv * P300_WEIGHT * math.exp(-((dt - 0.35) / 0.08) ** 2)
                if rng.random() < 0.0005:                      # an occasional blink on frontal channels
                    v += 150 * BLINK_WEIGHT
                if args.muse and len(LABELS) > 4:
                    v[4] = dc[4] + rng.normal(0, 60)           # floating AUX input: big noise
                x[i] = v
            stamps = ts + (time.time() - pylsl.local_clock()) if args.wall_clock else ts
            outlet.push_chunk(x.astype(np.float32).tolist(), stamps.tolist())
            n = target_n
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

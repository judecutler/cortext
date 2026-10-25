#!/usr/bin/env python3
"""Build the "p3000 function": a fixed, UNTRAINED P300 spike detector saved as a normal model file.

    python cortext-lab/pipeline/make_p3000_function.py             # -> models/library/p3000_function.npz
    python cortext-lab/pipeline/make_p3000_function.py --activate  # ...and make it the running model
    python cortext-lab/pipeline/make_p3000_function.py --muse      # -> models/library/p3000_function_muse.npz

What it does (no calibration, no training data, works from trial 1):
  1. Every picture flash is cut into its own 0-800 ms epoch, so "when did the brain spike?" becomes "which picture was
     on screen ~300 ms before the spike?" (the P300 peaks 250-500 ms after the thing you were looking for appears).
  2. For each flash it measures the spike: mean voltage 240-480 ms after that picture appeared, averaged over the
     parietal/central electrodes where the P300 is biggest (Cz, Pz, POz, PO7, PO8), in microvolts.
  3. The spikes for each option are added up over the repeats; the option with the biggest total is the predicted
     answer. In the bridge, percentages are measured against this trial's own noise (each total in standard errors,
     softmax slope p300.P3000_Z_SLOPE, top capped at 90%), because a fixed TEMPERATURE said 96% on pure 1/f noise.
     TEMPERATURE below is only used by code that scores the file like any other model.

It is stored as a LinearP300 whose weights are exactly that window average, so the bridge, the Agent page's Model
block and the narrator use it like any other model. It is built for the default 0.5-20 Hz filter chain (keep that
chain running) and for the 8-channel Unicorn montage. Unlike a calibrated model it is the same for every person,
so it is never "the wrong participant" - but a calibrated model is expected to beat it once you have one.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bridge"))
import filters as F  # noqa: E402
import models as M   # noqa: E402
import p300 as P     # noqa: E402

NAME = "p3000_function.npz"          # model files can't contain spaces; shown as "p3000_function" on the Agent page
LABELS = list(P.UNICORN_MONTAGE)     # Fz C3 Cz C4 Pz PO7 POz PO8
SPIKE_CHANNELS = ["Cz", "Pz", "POz", "PO7", "PO8"]
# Muse version: the P300 shows at TP9/TP10 (behind the ears; Muse reference is Fpz). AF7/AF8 are forehead channels,
# dominated by blinks and eye movements, so they get no weight. Built for the "Muse P300" filter chain.
MUSE_NAME = "p3000_function_muse.npz"
MUSE_LABELS = list(P.MUSE_MONTAGE)   # TP9 AF7 AF8 TP10
MUSE_SPIKE_CHANNELS = ["TP9", "TP10"]
MUSE_CHAIN = os.path.join(os.path.dirname(HERE), "filters", "library", "muse-p300.json")
SPIKE_WINDOW_MS = (240, 480)         # P300 peak latency, on the 40 ms bin grid
# Softmax slope per summed microvolt (after the causal filter). On simulated EEG (tests/test_p3000.py, 3 repeats):
# a 5 uV P300 gives ~80% correct with stated confidence up to ~65%; with no P300 it is at chance and never says
# more than ~50%. Deliberately soft, so the narrator doesn't announce "100% sure" on noise.
TEMPERATURE = 0.2


def build_muse(temperature=TEMPERATURE):
    return build(MUSE_LABELS, temperature, MUSE_SPIKE_CHANNELS, F.read_file(MUSE_CHAIN), headset="Muse")


def build(labels=LABELS, temperature=TEMPERATURE, spike_channels=SPIKE_CHANNELS, chain=None, headset="Unicorn 8-ch"):
    nb = P.n_bins()
    b0, b1 = (int((ms / 1000 - P.TMIN) * 1000 / P.BIN_MS) for ms in SPIKE_WINDOW_MS)
    chans = [i for i, l in enumerate(labels) if l in spike_channels] or list(range(len(labels)))
    w = np.zeros((len(labels), nb))
    w[chans, b0:b1] = 1.0
    w /= w.sum()                                   # score = plain mean uV over the window and channels
    d = len(labels) * nb
    meta = {"trained_on": "none (fixed spike detector, no training)", "participants": [], "kind": "p3000 function",
            "description": f"mean uV {SPIKE_WINDOW_MS[0]}-{SPIKE_WINDOW_MS[1]} ms after each picture over "
                           f"{', '.join(labels[i] for i in chans)}; biggest summed spike = answer",
            "simulated": False, "headset": headset, "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "n_epochs": 0, "n_trials": 0}
    return P.LinearP300(w.ravel(), 0.0, np.zeros(d), np.ones(d), labels, temperature, meta, chain or F.DEFAULT_SPEC)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--muse", action="store_true", help="build the Muse version (TP9/TP10, Muse P300 filter chain)")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--activate", action="store_true", help="also copy it into models/active/ (the running model)")
    args = ap.parse_args()
    args.out = args.out or os.path.join(M.library_dir(), MUSE_NAME if args.muse else NAME)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    m = build_muse(args.temperature) if args.muse else build(temperature=args.temperature)
    m.save(args.out)
    P.LinearP300.load(args.out)                    # same check the Agent page does
    print("saved", args.out)
    if args.activate:
        print("active:", M.activate_file(args.out))


if __name__ == "__main__":
    main()

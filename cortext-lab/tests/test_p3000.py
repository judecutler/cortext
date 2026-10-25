#!/usr/bin/env python3
"""The "p3000 function" (pipeline/make_p3000_function.py): loads like any model, appears in the model library, and on
SIMULATED EEG picks the attended picture when there is a P300 and sits at chance (with modest confidence) when there
isn't. Simulated accuracy is wiring evidence only, not evidence it works on a brain. Run: python tests/test_p3000.py"""
import os
import shutil
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bridge"))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import make_p3000_function as F3  # noqa: E402
import models as M  # noqa: E402
import p300 as P  # noqa: E402

SR = 250.0
P300_W = np.array([0.3, 0.5, 0.8, 0.5, 1.0, 0.6, 0.9, 0.6])   # same shape as eeg/fake_eeg.py
VEP_W = np.array([0.1, 0.2, 0.2, 0.2, 0.5, 0.9, 1.0, 0.9])


def simulate(rng, p300_uv, n_trials=40, repeats=3, opts=("A", "B", "C", "D"), noise_uv=8.0):
    """Continuous 8-ch EEG of sequential trials (1 s picture + 0.3 s gap), P300 after the true option's flashes."""
    trials, onsets_all = [], []
    t = 1.0
    for _ in range(n_trials):
        true = rng.choice(opts)
        fl = []
        for _ in range(repeats):
            for o in rng.permutation(opts):
                fl.append((t, o))
                t += 1.3
        trials.append((fl, true))
        t += 2.0
    n = int((t + 2) * SR)
    ts = np.arange(n) / SR
    x = rng.normal(0, noise_uv, (n, 8)) + np.cumsum(rng.normal(0, 0.3, (n, 8)), axis=0) + 4000.0   # noise+drift+DC
    k = np.arange(int(0.8 * SR)) / SR
    p3 = np.exp(-0.5 * ((k - 0.35) / 0.07) ** 2)
    vep = np.exp(-0.5 * ((k - 0.12) / 0.03) ** 2) - 0.7 * np.exp(-0.5 * ((k - 0.2) / 0.04) ** 2)
    for fl, true in trials:
        for on, o in fl:
            i = int(on * SR)
            x[i:i + len(k)] += 6.0 * vep[:, None] * VEP_W
            if o == true:
                x[i:i + len(k)] += p300_uv * p3[:, None] * P300_W
    y = P.CausalFilter(SR, 8)(x)
    return ts, y, trials


def run(model, p300_uv, seed):
    rng = np.random.default_rng(seed)
    ts, y, trials = simulate(rng, p300_uv)
    hits, confs = 0, []
    for fl, true in trials:
        eps = P.epochs(ts, y, [f[0] for f in fl])
        scores = [None if e is None else float(model.score(e.ravel())[0]) for e in eps]
        sums, _, _ = P.option_evidence([f[1] for f in fl], scores, ["A", "B", "C", "D"])
        probs = P.softmax(sums, model.temperature)
        best = max(probs, key=probs.get)
        hits += best == true
        confs.append(probs[best])
    return hits / len(trials), float(np.mean(confs)), float(np.max(confs))


def main():
    tmp = tempfile.mkdtemp()
    os.environ["CORTEXT_MODELS_DIR"] = tmp
    try:
        path = os.path.join(M.library_dir(), F3.NAME)
        os.makedirs(M.library_dir())
        F3.build().save(path)
        m = P.LinearP300.load(path)
        info = M.describe(path, labels=list(P.UNICORN_MONTAGE), filter_sig=P.LinearP300(filters=None).filter_signature)
        assert info.get("fits_headset") and info.get("fits_filters"), info
        assert "library/p3000_function.npz" in [d["id"] for d in M.listing(tmp)["library"]]
        assert M.activate_file(path).endswith("p3000_function.npz")
        assert abs(m.w.sum() - 1.0) < 1e-9          # a plain average: score is in microvolts

        acc, mean_c, max_c = run(m, 5.0, 1)
        print(f"simulated 5 uV P300: accuracy {acc:.0%} (chance 25%), mean confidence {mean_c:.0%}, max {max_c:.0%}")
        assert acc >= 0.6, acc
        acc0, mean0, max0 = run(m, 0.0, 2)
        print(f"no P300 (control):   accuracy {acc0:.0%}, mean confidence {mean0:.0%}, max {max0:.0%}")
        assert acc0 <= 0.5, acc0
        assert mean0 < 0.5, mean0                   # stays modest on noise
        print("ok")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

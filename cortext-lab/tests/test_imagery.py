"""Imagery window (settings.imagery_ms > 0): the pipeline writes imagery.npz and load_imagery() reads it back.

Synthetic logs, no bridge or browser:  python cortext-lab/tests/test_imagery.py
A 10 Hz burst on PO7/POz/PO8 is injected only while "dog" is being imagined, so the test can check that each epoch
lines up with its window and carries the right label. Also checks that a session WITHOUT imagery markers gets no
imagery.npz and still builds exactly as before.
"""
import csv
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TMP = tempfile.mkdtemp(prefix="cortext_imagery_")
os.environ["CORTEXT_DATA_DIR"] = os.path.join(TMP, "data")
sys.path.insert(0, os.path.join(HERE, "..", "pipeline"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "datasets", "cortext-p300"))
import build_session  # noqa: E402
import cortext_data  # noqa: E402

LABELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "POz", "PO8"]
SR = 250.0
IMAG_MS = 2000
CONCEPTS = ["dog", "bird", "fish"]


def write_session(log_dir, sid, imagery=True, n_trials=6, seed=0):
    rng = np.random.default_rng(seed)
    t0 = 1000.0
    trial_s = 9.0
    n = int((n_trials * trial_s + 4) * SR)
    ts = t0 + np.arange(n) / SR
    x = rng.normal(0, 5, (n, len(LABELS)))
    markers, seq = [], 0

    def mark(t, event, **kw):
        nonlocal seq
        seq += 1
        markers.append({"seq": seq, "lsl_timestamp": f"{t:.6f}", "event": event, **kw})

    mark(t0 + 0.5, "session_start", session_id=sid)
    for k in range(n_trials):
        concept = CONCEPTS[k % 3]
        ts0 = t0 + 2 + k * trial_s
        base = dict(session_id=sid, trial_id=k + 1, trial_index=k)
        mark(ts0, "trial_start", **base)
        mark(ts0 + 0.1, "prompt_on", **base)
        t_fl = ts0 + 1.0
        if imagery:
            on = ts0 + 1.0
            mark(on, "imagery_on", **base, option_id=concept, category="animal", is_target="true",
                 extra=json.dumps({"imagery_ms": IMAG_MS, "label": concept}))
            mark(on + IMAG_MS / 1000, "imagery_off", **base, option_id=concept, category="animal")
            if concept == "dog":   # 10 Hz, 20 uV on the occipital channels during the window only
                i0, i1 = int((on - t0) * SR), int((on + IMAG_MS / 1000 - t0) * SR)
                x[i0:i1, 5:] += 20 * np.sin(2 * np.pi * 10 * np.arange(i1 - i0) / SR)[:, None]
            t_fl = on + IMAG_MS / 1000 + 1.0
        opts = ["dog", "bird", "fish", "cat"]
        for r in range(2):
            for j, o in enumerate(opts):
                mark(t_fl + (r * 4 + j) * 0.5, "flash_on", **base, option_id=o, option_position=j, category="animal",
                     is_target=str(o == concept).lower(), repeat_index=r, flash_index=r * 4 + j,
                     extra=json.dumps({"mode": "flash"}))
        mark(t_fl + 4.5, "click", **base, option_id=concept)
        mark(t_fl + 4.6, "trial_end", **base, extra=json.dumps({"correct": True}))

    with open(os.path.join(log_dir, f"eeg_{sid}_1.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lsl_time", "page_t_ms"] + LABELS)
        for t, row in zip(ts, x):
            w.writerow([f"{t:.6f}", ""] + [f"{v:.3f}" for v in row])
    keys = sorted({k for m in markers for k in m})
    with open(os.path.join(log_dir, f"markers_{sid}_1.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(markers)


def main():
    log_dir = os.path.join(TMP, "logs")
    os.makedirs(log_dir)
    write_session(log_dir, "IMG1", imagery=True)
    write_session(log_dir, "PLAIN1", imagery=False, seed=1)

    r = build_session.build("IMG1", log_dir=log_dir)
    assert os.path.exists(os.path.join(r["path"], "imagery.npz")), "imagery.npz missing"
    assert r["quality"]["imagery_epochs"] == 6, r["quality"]
    meta = json.load(open(os.path.join(r["path"], "meta.json")))
    assert meta["files"]["imagery.npz"], meta["files"]

    r2 = build_session.build("PLAIN1", log_dir=log_dir)
    assert not os.path.exists(os.path.join(r2["path"], "imagery.npz")), "plain session must not get imagery.npz"
    assert "imagery_epochs" not in r2["quality"]
    assert r2["quality"]["epochs_total"] == 6 * 8, r2["quality"]

    X, y, info = cortext_data.load_imagery(drop_rejected=False)
    assert X.shape == (6, 8, int(0.2 * SR) + int(IMAG_MS / 1000 * SR)), X.shape
    assert list(y) == ["dog", "bird", "fish"] * 2, y
    assert set(info["session_id"]) == {"IMG1"}, "plain session leaked into imagery epochs"
    assert info["times"][0] < 0 and abs(info["times"][-1] - (IMAG_MS / 1000 - 1 / SR)) < 1e-6

    # the injected burst sits inside the window, on the right label and channels
    win = (info["times"] > 0.2) & (info["times"] < 1.8)
    occ = X[:, 5:, :][:, :, win].std(axis=(1, 2))
    front = X[:, :2, :][:, :, win].std(axis=(1, 2))
    dog = y == "dog"
    assert occ[dog].min() > 2 * occ[~dog].max(), (occ, y)
    assert front[dog].max() < 2 * front[~dog].min(), front

    # flash epochs are unchanged by the imagery window
    Xf, yf, _ = cortext_data.load_epochs(sessions=["IMG1"], drop_rejected=False)
    assert Xf.shape[0] == 6 * 8 and yf.sum() == 6 * 2, (Xf.shape, yf.sum())
    print(f"imagery OK: {len(X)} imagery epochs {X.shape[1:]} with labels {sorted(set(y))}; plain session unaffected; "
          f"flash epochs intact ({len(Xf)})")


if __name__ == "__main__":
    main()

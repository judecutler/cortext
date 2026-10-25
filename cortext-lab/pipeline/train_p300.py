#!/usr/bin/env python3
"""Train the live P300 scorer on the shared dataset and report honest cross-validated accuracy.

    python cortext-lab/pipeline/train_p300.py                       # all real sessions -> p300_model.npz
    python cortext-lab/pipeline/train_p300.py --sessions S1 S2      # chosen sessions
    python cortext-lab/bridge/bridge.py --p300-model p300_model.npz # use it live

It rebuilds features exactly the way the bridge computes them live (causal 0.5-20 Hz filter on the RAW recording,
40 ms bins), so the saved model can be dropped into the bridge. Evaluation:
  * grouped cross-validation by trial (no trial is ever in train and test), epoch AUC + answer accuracy per trial
  * shuffled-label control (should sit at AUC 0.5 / chance accuracy; if it doesn't, something leaks)
  * leave-one-session-out when there are 2+ sessions (the number that matters for a new session)
Uses the cleaned dataset only for validity/artifact flags. Your own models can use datasets/cortext-p300 directly.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bridge"))
import p300 as P  # noqa: E402
from build_session import dataset_root  # noqa: E402


def live_features(root, session_folder):
    """Features as the bridge computes them, for every flash epoch of one session (aligned with epochs.npz rows)."""
    raw = np.load(os.path.join(root, session_folder, "eeg_raw.npz"), allow_pickle=False)
    ep = np.load(os.path.join(root, session_folder, "epochs.npz"), allow_pickle=False)
    x, ts = raw["data"].T.astype(float), raw["times"]
    labels = [str(l) for l in raw["labels"]]
    y = P.CausalFilter(float(raw["srate"]), x.shape[1])(x)
    feats, keep = [], []
    for i, onset in enumerate(ep["onset_lsl"]):
        e = P.epoch(ts, y, float(onset))
        if e is not None:
            feats.append(e.ravel())
            keep.append(i)
    keep = np.array(keep, int)
    sid = ep["session_id"][keep] if len(keep) else np.array([])
    return (np.array(feats), ep["y"][keep].astype(int), np.array([f"{s}:{t}" for s, t in zip(sid, ep["trial_id"][keep])]),
            ep["option_id"][keep], ep["trial_valid"][keep] & ~ep["rejected_artifact"][keep], labels, sid)


def answer_accuracy(scores, y, groups, options):
    correct = []
    for g in np.unique(groups):
        m = groups == g
        opts = np.unique(options[m])
        sums = {o: scores[m][options[m] == o].sum() for o in opts}
        truth = options[m][y[m] == 1]
        if len(truth):
            correct.append(max(sums, key=sums.get) == truth[0])
    return float(np.mean(correct)) if correct else float("nan"), len(correct), len(np.unique(options))


def auc(s, y):
    pos, neg = s[y == 1], s[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    r = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def cross_val(X, y, groups, options, labels, folds=5, rng=None):
    ug = np.unique(groups)
    rng = rng or np.random.default_rng(0)
    perm = rng.permutation(ug)
    scores = np.zeros(len(y))
    for k in range(min(folds, len(ug))):
        test_g = set(perm[k::folds])
        te = np.array([g in test_g for g in groups])
        if y[~te].sum() < 2:
            continue
        m = P.LinearP300.fit(X[~te], y[~te], labels)
        scores[te] = m.score(X[te])
    return scores


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="dataset folder (default datasets/cortext-p300)")
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--include-simulated", action="store_true")
    ap.add_argument("-o", "--out", default="p300_model.npz")
    a = ap.parse_args()
    root = a.root or dataset_root()
    sys.path.insert(0, root)
    from cortext_data import list_sessions  # noqa: E402

    rows = list_sessions(root, a.include_simulated)
    if a.sessions:
        rows = [r for r in rows if r["session_id"] in a.sessions]
    if not rows:
        sys.exit(f"no sessions in {root} (pass --include-simulated to use simulated ones)")
    parts, labels = [], None
    for r in rows:
        X, y, g, o, ok, lab, sid = live_features(root, r["folder"])
        if labels is None:
            labels = lab
        if lab != labels:
            print(f"skip {r['session_id']}: different channels {lab}")
            continue
        parts.append((X[ok], y[ok], g[ok], o[ok], sid[ok]))
        print(f"{r['session_id']}: {ok.sum()} usable epochs ({y[ok].sum()} target) from {len(np.unique(g[ok]))} trials")
    X = np.vstack([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    groups = np.concatenate([p[2] for p in parts])
    options = np.concatenate([p[3] for p in parts])
    sids = np.concatenate([p[4] for p in parts])
    if y.sum() < 10:
        sys.exit("not enough target epochs to train (need 10+)")

    s = cross_val(X, y, groups, options, labels)
    acc, n_trials, n_opt = answer_accuracy(s, y, groups, options)
    accs0, aucs0 = [], []
    for seed in range(10):               # repeat: one shuffle of a few dozen trials is a noisy estimate
        rng = np.random.default_rng(100 + seed)
        y_shuf = y.copy()
        for g in np.unique(groups):      # pretend a random option was the target in each trial
            m = groups == g
            y_shuf[m] = (options[m] == rng.choice(np.unique(options[m]))).astype(int)
        s0 = cross_val(X, y_shuf, groups, options, labels, rng=np.random.default_rng(seed))
        accs0.append(answer_accuracy(s0, y_shuf, groups, options)[0])
        aucs0.append(auc(s0, y_shuf))
    print(f"\ncross-validated by trial: epoch AUC {auc(s, y):.3f}, answer accuracy {acc:.2f} over {n_trials} trials "
          f"(chance {1 / n_opt:.2f})")
    print(f"shuffled-label control (10 shuffles): AUC {np.mean(aucs0):.3f}, accuracy {np.mean(accs0):.2f} "
          f"+/- {np.std(accs0):.2f} (should be ~0.5 / chance)")
    if len(np.unique(sids)) > 1:
        for sid in np.unique(sids):
            te = sids == sid
            m = P.LinearP300.fit(X[~te], y[~te], labels)
            a1, n1, _ = answer_accuracy(m.score(X[te]), y[te], groups[te], options[te])
            print(f"leave-one-session-out, test {sid}: answer accuracy {a1:.2f} over {n1} trials")

    model = P.LinearP300.fit(X, y, labels, meta={"trained_on": [r["session_id"] for r in rows], "n_epochs": int(len(y)),
                                                 "cv_auc": round(auc(s, y), 3), "cv_answer_accuracy": round(acc, 3)})
    ev = []
    for g in np.unique(groups):
        m = groups == g
        sums, _, _ = P.option_evidence(list(options[m]), model.score(X[m]), sorted(set(options[m])))
        truth = options[m][y[m] == 1]
        if len(truth):
            ev.append((sums, truth[0]))
    model.temperature = P.fit_temperature(ev)
    model.save(a.out)
    print(f"\nsaved {a.out}  ->  python cortext-lab/bridge/bridge.py --p300-model {a.out}")


if __name__ == "__main__":
    main()

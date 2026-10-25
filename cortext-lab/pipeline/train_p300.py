#!/usr/bin/env python3
"""Train the live P300 scorer on the shared dataset and report honest cross-validated accuracy.

    python cortext-lab/pipeline/train_p300.py                       # all real sessions -> cortext-lab/models/library/
    python cortext-lab/pipeline/train_p300.py --sessions S1 S2      # chosen sessions
    python cortext-lab/pipeline/train_p300.py --activate            # ... and make it the model the bridge runs

Models are saved to the team library (cortext-lab/models/library/, shared through git) named <participants>_<date>.npz.
The bridge runs the one in cortext-lab/models/active/; pick it on the Setup page or use --activate (see models/README.md).

It rebuilds features exactly the way the bridge computes them live (the running filter chain, causal, on the RAW
recording; 40 ms bins), so the saved model can be dropped into the bridge. The chain is the one in
cortext-lab/filters/active/ (what the Agent page shows), else the built-in 0.5-20 Hz default; pick another with
--filters NAME_OR_FILE. The model stores its chain, and the bridge only uses it while that chain runs. Evaluation:
  * grouped cross-validation by trial (no trial is ever in train and test), epoch AUC + answer accuracy per trial
  * shuffled-label control (should sit at AUC 0.5 / chance accuracy; if it doesn't, something leaks)
  * leave-one-session-out when there are 2+ sessions (the number that matters for a new session)
Uses the cleaned dataset only for validity/artifact flags. Needs the raw recordings, so download sessions with
`python datasets/cortext-p300/cortext_store.py pull --all` first. Your own models can use datasets/cortext-p300/cortext_data.py directly.
"""
import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bridge"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "datasets", "cortext-p300"))
import p300 as P  # noqa: E402
import models  # noqa: E402
import filters as F  # noqa: E402
from build_session import dataset_root  # noqa: E402
from cortext_data import list_sessions  # noqa: E402


def live_features(root, session_folder, spec=None):
    """Features as the bridge computes them, for every flash epoch of one session (aligned with epochs.npz rows)."""
    raw = np.load(os.path.join(root, session_folder, "eeg_raw.npz"), allow_pickle=False)
    ep = np.load(os.path.join(root, session_folder, "epochs.npz"), allow_pickle=False)
    x, ts = raw["data"].T.astype(float), raw["times"]
    labels = [str(l) for l in raw["labels"]]
    chain = F.Chain(spec or F.DEFAULT_SPEC, float(raw["srate"]), labels)
    y = chain(x)
    feats, keep = [], []
    for i, onset in enumerate(ep["onset_lsl"]):
        e = P.epoch(ts, y, float(onset), chain.reject_uv)
        if e is not None:
            feats.append(e.ravel())
            keep.append(i)
    keep = np.array(keep, int)
    sid = ep["session_id"][keep] if len(keep) else np.array([])
    return (np.array(feats), ep["y"][keep].astype(int), np.array([f"{s}:{t}" for s, t in zip(sid, ep["trial_id"][keep])]),
            ep["option_id"][keep], ep["trial_valid"][keep] & ~ep["rejected_artifact"][keep], labels, sid)


def resolve_filters(arg):
    """--filters value -> chain spec. None = what the bridge runs (filters/active/, else the built-in default)."""
    if arg is None:
        return F.active_chain()[0]
    if arg.strip().lower() == "default":
        return F.normalize(F.DEFAULT_SPEC)
    if os.path.isfile(arg):
        return F.read_file(arg)
    name = F.safe_name(arg)
    path = os.path.join(F.library_dir(), name)
    if os.path.isfile(path):
        return F.read_file(path)
    have = ", ".join(n[:-5] for n in sorted(os.listdir(F.library_dir()))) if os.path.isdir(F.library_dir()) else "none"
    sys.exit(f"--filters: no chain called {arg!r} (library: {have})")


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
    ap.add_argument("--root", default=None, help="dataset folder (default data/cortext-p300)")
    ap.add_argument("--sessions", nargs="*", default=None)
    ap.add_argument("--include-simulated", action="store_true")
    ap.add_argument("--include-superseded", action="store_true", help="also use sessions that were redone in the app")
    ap.add_argument("-o", "--out", default=None,
                    help="where to save (default: cortext-lab/models/library/<participants>_<date>.npz, the team library)")
    ap.add_argument("--filters", default=None,
                    help="filter chain: a library name (e.g. p300-default), a .json path, or 'default' "
                         "(default: the chain in cortext-lab/filters/active/, i.e. what the bridge runs)")
    ap.add_argument("--activate", action="store_true",
                    help="also make it the model the bridge runs (copies it into cortext-lab/models/active/)")
    a = ap.parse_args()
    root = a.root or dataset_root()
    spec = resolve_filters(a.filters)
    print(f"filters: {spec['name']}: {'; '.join(F.summary(spec))}")

    rows = list_sessions(root, a.include_simulated, include_superseded=a.include_superseded or bool(a.sessions))
    if a.sessions:
        rows = [r for r in rows if r["session_id"] in a.sessions]
    if not rows:
        sys.exit(f"no sessions in {root}. Download them: python datasets/cortext-p300/cortext_store.py pull --all "
                 "(or pass --include-simulated to use simulated ones)")
    no_raw = [r["session_id"] for r in rows if not r["complete"]]
    if no_raw:
        print(f"skipping {len(no_raw)} session(s) downloaded without raw EEG ({', '.join(no_raw)}); "
              "run `cortext_store.py pull --all` to include them")
        rows = [r for r in rows if r["complete"]]
        if not rows:
            sys.exit("no session with raw EEG on this machine")
    parts, labels = [], None
    for r in rows:
        try:
            X, y, g, o, ok, lab, sid = live_features(root, r["folder"], spec)
        except F.FilterError as exc:
            sys.exit(f"{r['session_id']}: the filter chain can't run on this recording: {exc}")
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

    used = [r for r in rows if r["session_id"] in set(np.unique(sids))]
    participants = sorted({str(r.get("participant_id")) for r in used if r.get("participant_id")})
    model = P.LinearP300.fit(X, y, labels, filters=spec, meta={
        "trained_on": [r["session_id"] for r in used], "participants": participants,
        "simulated": any(r.get("simulated") for r in used), "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_epochs": int(len(y)), "n_trials": int(n_trials), "chance": round(1 / n_opt, 3),
        "cv_auc": round(auc(s, y), 3), "cv_answer_accuracy": round(acc, 3),
        "shuffled_accuracy": round(float(np.mean(accs0)), 3)})
    ev = []
    for g in np.unique(groups):
        m = groups == g
        sums, _, _ = P.option_evidence(list(options[m]), model.score(X[m]), sorted(set(options[m])))
        truth = options[m][y[m] == 1]
        if len(truth):
            ev.append((sums, truth[0]))
    model.temperature = P.fit_temperature(ev)
    out = a.out
    if out is None:
        who = "-".join(participants) or "unknown"
        out = os.path.join(models.library_dir(), models.safe_name(
            f"{who}_{time.strftime('%Y%m%d-%H%M')}{'_SIM' if model.meta['simulated'] else ''}.npz"))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    model.save(out)
    print(f"\nsaved {out}")
    if a.activate:
        dst = models.activate_file(out)
        print(f"active: {dst}  (the bridge runs it at its next start, or pick it on the Setup page now)")
    else:
        print("to run it: pick it on the Setup page (P300 model), or add --activate, or copy it into cortext-lab/models/active/")


if __name__ == "__main__":
    main()

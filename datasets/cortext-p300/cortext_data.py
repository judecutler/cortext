"""Load the CorText P300 dataset for model training.

The recordings are NOT in git. They live in a private Hugging Face dataset and in your local copy, data/cortext-p300/
(see docs/DATA_STORAGE.md). Download them first:

    python datasets/cortext-p300/cortext_store.py pull            # epochs only (small, enough to train)
    python datasets/cortext-p300/cortext_store.py pull --all      # also the continuous clean + raw EEG

then:

    import sys; sys.path.append("datasets/cortext-p300")      # path from the repo root
    from cortext_data import load_epochs, list_sessions

    X, y, info = load_epochs()          # all real sessions, valid trials, artifact-free epochs
    # X: (n_epochs, n_channels, n_times) microvolts, noise-reduced, baseline-corrected (-200..0 ms)
    # y: 1 = flash of the answer the participant attended (target), 0 = other option
    # info["groups"]: "session:trial" per epoch -> use GroupKFold so one trial never sits in train AND test
    # info["times"], info["labels"] (channel names), info["option_id"], info["session_id"], ...

    epochs = to_mne(X, info, y)         # optional: an mne.EpochsArray (needs mne)

Only numpy is required. Simulated sessions (fake_eeg.py) are excluded unless include_simulated=True.
"""
import csv
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cortext_store as store  # noqa: E402
from cortext_store import pull, data_root  # noqa: E402,F401  (re-exported: `from cortext_data import pull`)

ROOT = str(data_root())   # this machine's copy; override with the CORTEXT_DATA_DIR environment variable
PER_EPOCH = ["y", "trial_id", "trial_index", "option_id", "option_position", "category", "repeat_index", "flash_index",
             "mode", "trial_valid", "rejected_artifact", "clicked_option_id", "onset_lsl", "session_id"]


def list_sessions(root=ROOT, include_simulated=False):
    """One dict per local session (read from the meta.json files, so it is never out of date)."""
    return [r for r in store.scan_local(root) if include_simulated or not r["simulated"]]


def session_meta(session, root=ROOT):
    return json.load(open(os.path.join(_folder(session, root), "meta.json"), encoding="utf-8"))


def _folder(session, root):
    for r in list_sessions(root, include_simulated=True):
        if r["session_id"] == session or r["folder"] == session:
            return os.path.join(root, r["folder"])
    raise KeyError(f"session {session} is not in {root}. Download it with: python datasets/cortext-p300/cortext_store.py pull")


def load_epochs(sessions=None, root=ROOT, valid_only=True, drop_rejected=True, channels=None, tmin=None, tmax=None,
                include_simulated=False, mode=None, bucket=None):
    """Concatenate epochs from several sessions.

    sessions: list of session ids (default: all real sessions). channels: list of names to keep (in that order).
    valid_only: drop trials the operator marked invalid. drop_rejected: drop epochs flagged as artifacts.
    mode: "flash" or "sequential" to keep one presentation mode only.
    bucket: only sessions labelled with this bucket in the app (e.g. "train" or "test"); a list of buckets also works.
    """
    rows = list_sessions(root, include_simulated)
    nothing_downloaded = not rows
    if bucket is not None:
        want = {bucket} if isinstance(bucket, str) else set(bucket)
        rows = [r for r in rows if r.get("bucket") in want]
    if sessions is not None:
        rows = [r for r in rows if r["session_id"] in sessions]
    Xs, cols, labels, times = [], {k: [] for k in PER_EPOCH}, None, None
    for r in rows:
        z = np.load(os.path.join(root, r["folder"], "epochs.npz"), allow_pickle=False)
        X = z["X"]
        if not len(X):
            continue
        lab = [str(l) for l in z["labels"]]
        t = z["times"]
        if channels is not None:
            missing = [c for c in channels if c not in lab]
            if missing:
                raise ValueError(f"session {r['session_id']} has no channel(s) {missing}")
            X = X[:, [lab.index(c) for c in channels]]
            lab = list(channels)
        keep_t = np.ones(len(t), bool)
        if tmin is not None:
            keep_t &= t >= tmin - 1e-9
        if tmax is not None:
            keep_t &= t <= tmax + 1e-9
        X, t = X[:, :, keep_t], t[keep_t]
        if labels is None:
            labels, times = lab, t
        elif lab != labels or len(t) != len(times):
            raise ValueError("sessions differ in channels or epoch length; pass channels=/tmin=/tmax= to align them")
        m = np.ones(len(X), bool)
        if valid_only:
            m &= z["trial_valid"]
        if drop_rejected:
            m &= ~z["rejected_artifact"]
        if mode is not None:
            m &= z["mode"] == mode
        Xs.append(X[m])
        for k in PER_EPOCH:
            cols[k].append(z[k][m])
    if not Xs:
        if nothing_downloaded:
            warnings.warn(f"No sessions found in {root}. Download them with: python datasets/cortext-p300/cortext_store.py pull", stacklevel=2)
        return np.zeros((0, 0, 0), np.float32), np.zeros(0, np.int8), {"labels": [], "times": np.zeros(0), "groups": np.array([])}
    info = {k: np.concatenate(v) for k, v in cols.items()}
    y = info.pop("y")
    info["groups"] = np.array([f"{s}:{t}" for s, t in zip(info["session_id"], info["trial_id"])])
    info["labels"], info["times"] = labels, times
    return np.concatenate(Xs), y, info


def load_imagery(sessions=None, root=ROOT, valid_only=True, drop_rejected=True, channels=None, include_simulated=False,
                 bucket=None):
    """Epochs from the optional imagery window (sessions recorded with imagery_ms > 0; others are skipped).

    X: (n, channels, time) uV, -200 ms .. imagery_ms after the window opened, baseline -200..0 ms.
    y: the concept the participant was asked to picture (option_id of the correct answer), as strings.
    info["category"], info["groups"] ("session:trial"), info["labels"], info["times"], ...
    Sessions with different window lengths are cut to the shortest.
    """
    rows = list_sessions(root, include_simulated)
    if bucket is not None:
        want = {bucket} if isinstance(bucket, str) else set(bucket)
        rows = [r for r in rows if r.get("bucket") in want]
    if sessions is not None:
        rows = [r for r in rows if r["session_id"] in sessions]
    parts = []
    for r in rows:
        path = os.path.join(root, r["folder"], "imagery.npz")
        if not os.path.exists(path):
            continue
        z = dict(np.load(path, allow_pickle=False))
        if not len(z["X"]):
            continue
        lab = [str(l) for l in z["labels"]]
        if channels is not None:
            missing = [c for c in channels if c not in lab]
            if missing:
                raise ValueError(f"session {r['session_id']} has no channel(s) {missing}")
            z["X"] = z["X"][:, [lab.index(c) for c in channels]]
            lab = list(channels)
        m = np.ones(len(z["X"]), bool)
        if valid_only:
            m &= z["trial_valid"]
        if drop_rejected:
            m &= ~z["rejected_artifact"]
        parts.append((z, m, lab))
    if not parts:
        return np.zeros((0, 0, 0), np.float32), np.zeros(0, str), {"labels": [], "times": np.zeros(0), "groups": np.array([])}
    if len({tuple(p[2]) for p in parts}) > 1:
        raise ValueError("sessions differ in channels; pass channels= to align them")
    n_t = min(p[0]["X"].shape[2] for p in parts)
    keys = ["trial_id", "option_id", "category", "trial_valid", "rejected_artifact", "clicked_option_id", "onset_lsl", "session_id"]
    X = np.concatenate([p[0]["X"][p[1]][:, :, :n_t] for p in parts])
    info = {k: np.concatenate([p[0][k][p[1]] for p in parts]) for k in keys}
    y = info.pop("option_id")
    info["groups"] = np.array([f"{s}:{t}" for s, t in zip(info["session_id"], info["trial_id"])])
    info["labels"], info["times"] = parts[0][2], parts[0][0]["times"][:n_t]
    return X, y, info


def load_continuous(session, root=ROOT, clean=True):
    """(data: channels x samples in uV, times: LSL seconds, labels, srate) for one session."""
    name = "eeg_clean.npz" if clean else "eeg_raw.npz"
    path = os.path.join(_folder(session, root), name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} is not on this machine (sessions are downloaded as epochs only by default). "
                                "Run: python datasets/cortext-p300/cortext_store.py pull --all")
    z = np.load(path, allow_pickle=False)
    return z["data"], z["times"], [str(l) for l in z["labels"]], float(z["srate"])


def load_events(session, root=ROOT):
    with open(os.path.join(_folder(session, root), "events.csv"), newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def trial_option_scores(epoch_scores, info):
    """Average per-epoch scores into one score per (trial, option): what the answer predictor needs.

    Returns {group: {option_id: mean score}}; the predicted answer is the option with the highest score.
    """
    out = {}
    for s, g, o in zip(epoch_scores, info["groups"], info["option_id"]):
        out.setdefault(g, {}).setdefault(o, []).append(float(s))
    return {g: {o: float(np.mean(v)) for o, v in d.items()} for g, d in out.items()}


def to_mne(X, info, y=None):
    """mne.EpochsArray (event id 1 = target, 0 = non-target when y is given)."""
    import mne
    srate = 1.0 / float(np.median(np.diff(info["times"])))
    mi = mne.create_info(list(info["labels"]), srate, "eeg")
    events = np.column_stack([np.arange(len(X)), np.zeros(len(X), int), np.zeros(len(X), int) if y is None else np.asarray(y, int)])
    ep = mne.EpochsArray(X * 1e-6, mi, events=events, tmin=float(info["times"][0]), verbose=False)  # MNE uses volts
    try:
        ep.set_montage("standard_1020", on_missing="ignore")
    except Exception:
        pass
    return ep

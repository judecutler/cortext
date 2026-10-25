"""Load the CorText P300 dataset (this folder) for model training.

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

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
PER_EPOCH = ["y", "trial_id", "trial_index", "option_id", "option_position", "category", "repeat_index", "flash_index",
             "mode", "trial_valid", "rejected_artifact", "clicked_option_id", "onset_lsl", "session_id"]


def list_sessions(root=ROOT, include_simulated=False):
    """Rows of index.csv (one per session) as dicts."""
    path = os.path.join(root, "index.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if include_simulated or r.get("simulated") not in ("True", "true", "1")]


def session_meta(session, root=ROOT):
    return json.load(open(os.path.join(_folder(session, root), "meta.json"), encoding="utf-8"))


def _folder(session, root):
    for r in list_sessions(root, include_simulated=True):
        if r["session_id"] == session or r["folder"] == session:
            return os.path.join(root, r["folder"])
    raise KeyError(f"session {session} not in {root}/index.csv")


def load_epochs(sessions=None, root=ROOT, valid_only=True, drop_rejected=True, channels=None, tmin=None, tmax=None,
                include_simulated=False, mode=None):
    """Concatenate epochs from several sessions.

    sessions: list of session ids (default: all real sessions). channels: list of names to keep (in that order).
    valid_only: drop trials the operator marked invalid. drop_rejected: drop epochs flagged as artifacts.
    mode: "flash" or "sequential" to keep one presentation mode only.
    """
    rows = list_sessions(root, include_simulated)
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
        return np.zeros((0, 0, 0), np.float32), np.zeros(0, np.int8), {"labels": [], "times": np.zeros(0), "groups": np.array([])}
    info = {k: np.concatenate(v) for k, v in cols.items()}
    y = info.pop("y")
    info["groups"] = np.array([f"{s}:{t}" for s, t in zip(info["session_id"], info["trial_id"])])
    info["labels"], info["times"] = labels, times
    return np.concatenate(Xs), y, info


def load_continuous(session, root=ROOT, clean=True):
    """(data: channels x samples in uV, times: LSL seconds, labels, srate) for one session."""
    z = np.load(os.path.join(_folder(session, root), "eeg_clean.npz" if clean else "eeg_raw.npz"), allow_pickle=False)
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

#!/usr/bin/env python3
"""Turn one recorded session into a clean, shareable training dataset entry.

    python cortext-lab/pipeline/build_session.py <session_id>            # from the bridge's logs
    python cortext-lab/pipeline/build_session.py --all                   # every session found in the logs

The bridge runs this automatically when a session ends (and again when the app sends its cleaned trial table), so
normally you never call it by hand. Output (see datasets/cortext-p300/README.md and docs/DATA_STORAGE.md):

    data/cortext-p300/sessions/<session_id>/        (git-ignored; shared through Hugging Face, not GitHub)
        meta.json        who/what/when, stream info, every processing parameter, quality report
        eeg_clean.npz    noise-reduced continuous EEG  (data: channels x samples, microvolts, float32; times: LSL s)
        eeg_raw.npz      the untouched recording (so anyone can redo the cleaning differently)
        epochs.npz       ready to train: X (n, channels, time), y (1 = flash of the attended answer), trial ids, ...
        events.csv       every marker, with its LSL time and sample index into eeg_clean
        trials.csv       one row per trial attempt from the app (validity, click, correct, model output)
    data/cortext-p300/index.csv  one row per session (rebuilt automatically)

Simulated sessions (fake_eeg.py) go to data/cortext-p300/simulated/ instead, so they never mix with real data.

Noise reduction (automatic, zero-phase because this is offline):
  1. split at gaps in the Bluetooth stream (> 0.1 s) and put each piece on an exact 250 Hz grid
  2. detect mains frequency (50 or 60 Hz) from the spectrum, notch it and its harmonics
  3. band-pass 0.5-40 Hz (removes electrode drift and muscle/high-frequency noise)
  4. find bad electrodes (flat, or far noisier than the others) and rebuild them from neighbours
  5. flag artifact spans (blinks, movement: > 150 uV) and mark epochs that touch them as rejected (kept, flagged)
Needs numpy + scipy.
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt, welch

PIPELINE_VERSION = "1.0"
HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.dirname(HERE)
LOG_DIR = os.path.join(LAB, "bridge", "logs")

sys.path.insert(0, os.path.join(os.path.dirname(LAB), "datasets", "cortext-p300"))
import cortext_store as store  # noqa: E402  (where sessions live and how they are shared)

PARAMS = {
    "gap_split_s": 0.1, "min_segment_s": 3.0,
    "bandpass_hz": [0.5, 40.0], "bandpass_order": 4,
    "notch_q": 30.0,
    "flat_std_uv": 0.5, "noisy_std_uv": 100.0, "noisy_vs_median": 4.0,
    "artifact_uv": 150.0, "artifact_pad_s": 0.2,
    "epoch_tmin_s": -0.2, "epoch_tmax_s": 0.8, "baseline_s": [-0.2, 0.0],
}

# Approximate 2-D scalp positions (x right, y towards the nose), used to rebuild bad channels from neighbours.
POS = {"Fp1": (-.3, .95), "Fp2": (.3, .95), "AF3": (-.25, .75), "AF4": (.25, .75), "F7": (-.8, .6), "F3": (-.4, .5),
       "Fz": (0, .5), "F4": (.4, .5), "F8": (.8, .6), "FC1": (-.25, .25), "FCz": (0, .25), "FC2": (.25, .25),
       "T7": (-.95, 0), "C3": (-.5, 0), "Cz": (0, 0), "C4": (.5, 0), "T8": (.95, 0), "CP1": (-.25, -.25),
       "CPz": (0, -.25), "CP2": (.25, -.25), "P7": (-.8, -.6), "P3": (-.4, -.5), "Pz": (0, -.5), "P4": (.4, -.5),
       "P8": (.8, -.6), "PO7": (-.55, -.75), "PO3": (-.25, -.75), "POz": (0, -.75), "PO4": (.25, -.75),
       "PO8": (.55, -.75), "O1": (-.3, -.95), "Oz": (0, -.95), "O2": (.3, -.95), "TP9": (-1, -.3), "TP10": (1, -.3),
       "AF7": (-.55, .8), "AF8": (.55, .8)}


def dataset_root():
    """This machine's copy of the dataset (data/cortext-p300 by default; the CORTEXT_DATA_DIR environment variable overrides)."""
    return str(store.data_root(create=True))


def _safe(sid):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(sid))


def find_logs(session_id, log_dir=LOG_DIR):
    s = _safe(session_id)
    return (sorted(glob.glob(os.path.join(log_dir, f"eeg_{s}_*.csv"))),
            sorted(glob.glob(os.path.join(log_dir, f"markers_{s}_*.csv"))))


def all_logged_sessions(log_dir=LOG_DIR):
    out = set()
    for f in glob.glob(os.path.join(log_dir, "eeg_*.csv")):
        out.add(os.path.basename(f)[4:].rsplit("_", 1)[0])
    return sorted(out)


# ---------------------------------------------------------------- loading
def load_eeg(files):
    labels, rows = None, []
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            r = csv.reader(fh)
            head = next(r, None)
            if not head:
                continue
            labels = labels or head[2:]
            for row in r:
                if len(row) == len(head):
                    rows.append(row)
    if not rows:
        raise ValueError("no EEG samples recorded for this session")
    a = np.array([[float(r[0])] + [float(v) for v in r[2:]] for r in rows])
    page = np.array([float(r[1]) if r[1] else np.nan for r in rows])
    order = np.argsort(a[:, 0], kind="stable")
    a, page = a[order], page[order]
    keep = np.concatenate([[True], np.diff(a[:, 0]) > 0])   # drop duplicate timestamps
    return labels, a[keep, 0], a[keep, 1:], page[keep]


def load_markers(files):
    seen, out = set(), []
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                key = (row.get("seq"), row.get("event"), row.get("trial_id"))
                if key in seen:
                    continue
                seen.add(key)
                out.append(row)
    out.sort(key=lambda r: float(r["lsl_timestamp"]) if r.get("lsl_timestamp") else float("inf"))
    return out


# ---------------------------------------------------------------- cleaning
def regrid(ts, x, srate):
    """Split at gaps and resample each piece onto an exact grid. Returns list of (t_uniform, x_uniform)."""
    gaps = np.where(np.diff(ts) > PARAMS["gap_split_s"])[0]
    bounds = np.split(np.arange(len(ts)), gaps + 1)
    segs = []
    for idx in bounds:
        if len(idx) < 2 or ts[idx[-1]] - ts[idx[0]] < PARAMS["min_segment_s"]:
            continue
        t = np.arange(ts[idx[0]], ts[idx[-1]], 1.0 / srate)
        xu = np.column_stack([np.interp(t, ts[idx], x[idx, c]) for c in range(x.shape[1])])
        segs.append((t, xu))
    return segs


def detect_mains(segs, srate):
    x = np.vstack([s[1] for s in segs])
    f, p = welch(x - x.mean(axis=0), fs=srate, nperseg=min(len(x), int(srate * 4)), axis=0)
    p = p.mean(axis=1)

    def prominence(f0):
        if f0 >= srate / 2 - 1:
            return 0.0
        peak = p[(f >= f0 - 1) & (f <= f0 + 1)].max()
        side = np.median(p[((f >= f0 - 6) & (f <= f0 - 3)) | ((f >= f0 + 3) & (f <= f0 + 6))])
        return float(peak / (side + 1e-12))
    r50, r60 = prominence(50), prominence(60)
    mains = 60 if r60 >= r50 else 50
    return mains, {"prominence_50hz": round(r50, 2), "prominence_60hz": round(r60, 2)}


def clean_segment(x, srate, mains):
    y = x - x.mean(axis=0)
    for f0 in (mains, 2 * mains):
        if f0 < srate / 2 - 1:
            b, a = iirnotch(f0, PARAMS["notch_q"], fs=srate)
            y = filtfilt(b, a, y, axis=0)
    sos = butter(PARAMS["bandpass_order"], PARAMS["bandpass_hz"], btype="bandpass", fs=srate, output="sos")
    return sosfiltfilt(sos, y, axis=0)


def find_bad_channels(clean, labels):
    std = clean.std(axis=0)
    med = np.median(std)
    bad, why = [], {}
    for i, l in enumerate(labels):
        if std[i] < PARAMS["flat_std_uv"]:
            bad.append(i); why[l] = f"flat (std {std[i]:.2f} uV)"
        elif std[i] > PARAMS["noisy_std_uv"] or std[i] > PARAMS["noisy_vs_median"] * med:
            bad.append(i); why[l] = f"noisy (std {std[i]:.1f} uV vs median {med:.1f})"
    return bad, why, {l: round(float(s), 2) for l, s in zip(labels, std)}


def interpolate_bad(clean, labels, bad):
    good = [i for i in range(len(labels)) if i not in bad and labels[i] in POS]
    done = []
    if not good:
        return clean, done
    for i in bad:
        if labels[i] not in POS:
            continue
        p = np.array(POS[labels[i]])
        d = np.array([np.linalg.norm(p - np.array(POS[labels[g]])) for g in good])
        w = 1.0 / (d + 0.05) ** 2
        clean[:, i] = clean[:, good] @ (w / w.sum())
        done.append(labels[i])
    return clean, done


def artifact_mask(clean, srate, good_idx):
    over = (np.abs(clean[:, good_idx]) > PARAMS["artifact_uv"]).any(axis=1) if good_idx else np.zeros(len(clean), bool)
    pad = int(PARAMS["artifact_pad_s"] * srate)
    if over.any() and pad:
        over = np.convolve(over.astype(int), np.ones(2 * pad + 1, int), "same") > 0
    return over


# ---------------------------------------------------------------- validity from markers / app
def trial_validity(markers, trials_from_app):
    """trial_id -> (valid, reason). The app's table (after operator review) wins; else derived from markers."""
    if trials_from_app:
        return {int(t["trial_id"]): (bool(t.get("valid")), t.get("invalid_reason") or "") for t in trials_from_app}
    v = {}
    for m in markers:
        if m.get("trial_id") in (None, ""):
            continue
        tid = int(m["trial_id"])
        ev = m["event"]
        if ev == "trial_start":
            v.setdefault(tid, (False, "no trial_end"))
        elif ev == "trial_end" and v.get(tid, (False, ""))[1] == "no trial_end":
            v[tid] = (True, "")
        elif ev in ("trial_invalid", "trial_aborted", "trial_redo", "no_response"):
            v[tid] = (False, ev)
    return v


# ---------------------------------------------------------------- main build
def build(session_id, trials=None, session=None, out_root=None, log_dir=LOG_DIR, keep_raw=True, stream_info=None):
    eeg_files, marker_files = find_logs(session_id, log_dir)
    if not eeg_files:
        raise FileNotFoundError(f"no EEG log for session {session_id} in {log_dir}")
    labels, ts, x, page = load_eeg(eeg_files)
    markers = load_markers(marker_files)
    srate = float(round(1.0 / np.median(np.diff(ts))))
    if stream_info is None:
        for f in eeg_files:
            try:
                stream_info = json.load(open(f[:-4] + ".json", encoding="utf-8"))
            except Exception:
                pass
    simulated = bool((stream_info or {}).get("simulated"))
    if session and session.get("eeg", {}).get("simulated"):
        simulated = True

    segs = regrid(ts, x, srate)
    if not segs:
        raise ValueError("no continuous stretch of EEG long enough to clean (Bluetooth dropouts?)")
    mains, mains_info = detect_mains(segs, srate)
    t_all = np.concatenate([s[0] for s in segs])
    raw_all = np.vstack([s[1] for s in segs])
    clean_all = np.vstack([clean_segment(s[1], srate, mains) for s in segs])
    seg_id = np.concatenate([np.full(len(s[0]), k) for k, s in enumerate(segs)])

    bad, why, std_before = find_bad_channels(clean_all, labels)
    clean_all, interpolated = interpolate_bad(clean_all, labels, bad)
    good_idx = [i for i in range(len(labels)) if i not in bad]
    art = artifact_mask(clean_all, srate, good_idx)

    # ---- epochs on flash onsets
    valid_map = trial_validity(markers, trials)
    click = {}
    for m in markers:
        if m["event"] == "click" and m.get("trial_id") not in (None, ""):
            click[int(m["trial_id"])] = m.get("option_id")
    flashes = [m for m in markers if m["event"] == "flash_on" and m.get("lsl_timestamp")]
    n0 = int(round(PARAMS["epoch_tmin_s"] * srate))
    n1 = int(round(PARAMS["epoch_tmax_s"] * srate))
    times = np.arange(n0, n1) / srate
    bl = (times >= PARAMS["baseline_s"][0]) & (times < PARAMS["baseline_s"][1])
    X, rows = [], []
    for m in flashes:
        t = float(m["lsl_timestamp"])
        i = int(np.searchsorted(t_all, t))
        if i + n0 < 0 or i + n1 > len(t_all) or abs(t_all[min(i, len(t_all) - 1)] - t) > 1.5 / srate:
            continue
        if seg_id[i + n0] != seg_id[i + n1 - 1]:
            continue  # epoch would cross a gap
        ep = clean_all[i + n0:i + n1].T
        ep = ep - ep[:, bl].mean(axis=1, keepdims=True)
        tid = int(m["trial_id"])
        valid, reason = valid_map.get(tid, (False, "unknown trial"))
        rejected = bool(art[i + n0:i + n1].any())
        X.append(ep.astype(np.float32))
        extra = {}
        try:
            extra = json.loads(m.get("extra") or "{}") or {}
        except ValueError:
            pass
        rows.append({"trial_id": tid, "trial_index": int(m["trial_index"]) if m.get("trial_index") else -1,
                     "option_id": m.get("option_id") or "", "option_position": int(m["option_position"]) if m.get("option_position") else -1,
                     "category": m.get("category") or "", "is_target": m.get("is_target") in ("True", "true", "1"),
                     "repeat_index": int(m["repeat_index"]) if m.get("repeat_index") else -1,
                     "flash_index": int(m["flash_index"]) if m.get("flash_index") else -1,
                     "mode": extra.get("mode", "flash") if isinstance(extra, dict) else "flash",
                     "trial_valid": valid, "invalid_reason": reason, "rejected_artifact": rejected,
                     "clicked_option_id": click.get(tid) or "", "onset_lsl": t, "sample": i})

    # ---- imagery epochs (only when the session used the optional imagery window: imagery_on markers)
    imag = _imagery_epochs(markers, t_all, clean_all, seg_id, art, srate, valid_map, click)

    # ---- write
    root = out_root or dataset_root()
    sub = "simulated" if simulated else "sessions"
    out = os.path.join(root, sub, _safe(session_id))
    tmp = out + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    np.savez_compressed(os.path.join(tmp, "eeg_clean.npz"), data=clean_all.T.astype(np.float32), times=t_all,
                        srate=srate, labels=np.array(labels), segment=seg_id.astype(np.int16), artifact=art)
    if keep_raw:
        np.savez_compressed(os.path.join(tmp, "eeg_raw.npz"), data=x.T.astype(np.float32), times=ts,
                            srate=srate, labels=np.array(labels))
    n = len(rows)
    col = lambda k, dt=None: np.array([r[k] for r in rows], dtype=dt)  # noqa: E731
    np.savez_compressed(
        os.path.join(tmp, "epochs.npz"),
        X=np.array(X, dtype=np.float32).reshape(n, len(labels), len(times)), times=times, labels=np.array(labels),
        srate=srate, y=col("is_target", np.int8), trial_id=col("trial_id", np.int32), trial_index=col("trial_index", np.int32),
        option_id=col("option_id", str), option_position=col("option_position", np.int8), category=col("category", str),
        repeat_index=col("repeat_index", np.int16), flash_index=col("flash_index", np.int16), mode=col("mode", str),
        trial_valid=col("trial_valid", bool), rejected_artifact=col("rejected_artifact", bool),
        clicked_option_id=col("clicked_option_id", str), onset_lsl=col("onset_lsl", float),
        session_id=np.array([str(session_id)] * n))
    if imag is not None:
        np.savez_compressed(os.path.join(tmp, "imagery.npz"), **imag, labels=np.array(labels), srate=srate,
                            session_id=np.array([str(session_id)] * len(imag["X"])))
    with open(os.path.join(tmp, "events.csv"), "w", newline="", encoding="utf-8") as fh:
        keys = ["lsl_time", "sample", "event", "trial_id", "trial_index", "option_id", "option_position", "category",
                "is_target", "repeat_index", "flash_index", "true_answer_id", "extra", "wall_ms"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for m in markers:
            if not m.get("lsl_timestamp"):
                continue
            t = float(m["lsl_timestamp"])
            w.writerow({**m, "lsl_time": f"{t:.6f}", "sample": int(np.searchsorted(t_all, t))})
    if trials:
        keys = sorted({k for t in trials for k in t if not isinstance(t[k], (dict, list))})
        with open(os.path.join(tmp, "trials.csv"), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(trials)

    usable = sum(1 for r in rows if r["trial_valid"] and not r["rejected_artifact"])
    trial_ids = {r["trial_id"] for r in rows}
    report = {
        "duration_s": round(float(ts[-1] - ts[0]), 1), "samples": int(len(ts)), "segments": len(segs),
        "dropped_samples_est": int(max(0, round((ts[-1] - ts[0]) * srate) + 1 - len(ts))),
        "mains_hz": mains, **mains_info,
        "channel_std_uv_after_filter": std_before, "bad_channels": why, "interpolated_channels": interpolated,
        "artifact_fraction": round(float(art.mean()), 3),
        "epochs_total": n, "epochs_target": int(sum(r["is_target"] for r in rows)),
        "epochs_rejected_artifact": int(sum(r["rejected_artifact"] for r in rows)),
        "epochs_in_invalid_trials": int(sum(not r["trial_valid"] for r in rows)),
        "epochs_usable": usable, "trials": len(trial_ids),
        "trials_valid": len({r["trial_id"] for r in rows if r["trial_valid"]}),
        "flash_markers_without_eeg": len(flashes) - n,
    }
    if imag is not None:
        report["imagery_epochs"] = int(len(imag["X"]))
        report["imagery_epochs_rejected_artifact"] = int(imag["rejected_artifact"].sum())
    report["verdict"] = quality_verdict(report, labels)
    meta = {
        "session_id": session_id, "dataset": "cortext-p300", "pipeline_version": PIPELINE_VERSION,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "simulated": simulated,
        "bucket": (session or {}).get("bucket") or "", "participant_id": (session or {}).get("participant_id"), "device": (session or {}).get("device"),
        "created_at": (session or {}).get("created_at"), "notes": (session or {}).get("notes"),
        # set in the app: "Redo this session" marks the old one superseded (kept, but loaders skip it by default)
        "superseded_by": (session or {}).get("superseded_by"), "redo_of": (session or {}).get("redo_of"),
        "previous_session_id": (session or {}).get("previous_session_id"),
        "stream": stream_info or (session or {}).get("eeg"), "labels": labels, "srate": srate, "units": "uV",
        "settings": (session or {}).get("settings"), "processing": PARAMS, "quality": report,
        "sources": {"eeg_logs": [os.path.basename(f) for f in eeg_files], "marker_logs": [os.path.basename(f) for f in marker_files]},
        "files": {"eeg_clean.npz": "noise-reduced continuous EEG", "eeg_raw.npz": "raw recording" if keep_raw else None,
                  "epochs.npz": "epochs on every flash onset",
                  "imagery.npz": "one epoch per imagery window (-200 ms .. imagery_ms after imagery_on)" if imag is not None else None,
                  "events.csv": "all markers", "trials.csv": "app trial table" if trials else None},
    }
    with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    store.rebuild_index(root)
    return {"path": out, "relative_path": store.display_path(out), "session_id": session_id, "simulated": simulated, "quality": report}


def _imagery_epochs(markers, t_all, clean_all, seg_id, art, srate, valid_map, click):
    """One epoch per imagery window: -200 ms .. imagery_ms after imagery_on, baseline -200..0 ms.

    Labels: option_id / category = the correct answer the participant was asked to picture. All epochs in a session get
    the length of the SHORTEST window so they stack. Artifact epochs are flagged, not dropped (same rule as flashes).
    Returns None when the session has no imagery_on markers.
    """
    ons = [m for m in markers if m["event"] == "imagery_on" and m.get("lsl_timestamp")]
    if not ons:
        return None
    durs = []
    for m in ons:
        try:
            durs.append(float((json.loads(m.get("extra") or "{}") or {}).get("imagery_ms")) / 1000.0)
        except (TypeError, ValueError):
            pass
    dur = min(durs) if durs else 2.0
    n0 = int(round(PARAMS["epoch_tmin_s"] * srate))
    n1 = int(round(dur * srate))
    times = np.arange(n0, n1) / srate
    bl = (times >= PARAMS["baseline_s"][0]) & (times < PARAMS["baseline_s"][1])
    X, rows = [], []
    for m in ons:
        t = float(m["lsl_timestamp"])
        i = int(np.searchsorted(t_all, t))
        if i + n0 < 0 or i + n1 > len(t_all) or abs(t_all[min(i, len(t_all) - 1)] - t) > 1.5 / srate:
            continue
        if seg_id[i + n0] != seg_id[i + n1 - 1]:
            continue
        ep = clean_all[i + n0:i + n1].T
        ep = ep - ep[:, bl].mean(axis=1, keepdims=True)
        tid = int(m["trial_id"])
        valid, _ = valid_map.get(tid, (False, "unknown trial"))
        X.append(ep.astype(np.float32))
        rows.append((tid, m.get("option_id") or "", m.get("category") or "", valid, bool(art[i + n0:i + n1].any()),
                     click.get(tid) or "", t))
    n = len(rows)
    col = lambda k, dt=None: np.array([r[k] for r in rows], dtype=dt)  # noqa: E731
    return {"X": np.array(X, dtype=np.float32).reshape(n, clean_all.shape[1], len(times)), "times": times,
            "trial_id": col(0, np.int32), "option_id": col(1, str), "category": col(2, str), "trial_valid": col(3, bool),
            "rejected_artifact": col(4, bool), "clicked_option_id": col(5, str), "onset_lsl": col(6, float)}


def quality_verdict(r, labels):
    msgs = []
    if r["bad_channels"]:
        msgs.append(f"{len(r['bad_channels'])} bad electrode(s): {', '.join(r['bad_channels'])} (rebuilt from neighbours) - check contact/gel")
    if r["artifact_fraction"] > 0.3:
        msgs.append(f"{r['artifact_fraction']:.0%} of the recording has artifacts > {PARAMS['artifact_uv']:.0f} uV - movement, blinking or loose electrodes")
    if r["epochs_total"] and r["epochs_rejected_artifact"] / r["epochs_total"] > 0.3:
        msgs.append(f"{r['epochs_rejected_artifact']} of {r['epochs_total']} epochs rejected for artifacts")
    if r["flash_markers_without_eeg"]:
        msgs.append(f"{r['flash_markers_without_eeg']} flashes had no usable EEG (gaps or recording started late)")
    return {"ok": not msgs, "messages": msgs or ["looks usable"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id", nargs="?")
    ap.add_argument("--all", action="store_true", help="build every session found in the bridge logs")
    ap.add_argument("--logs", default=LOG_DIR)
    ap.add_argument("--out", default=None, help="dataset folder (default: <repo>/data/cortext-p300)")
    ap.add_argument("--no-raw", action="store_true", help="don't include the raw recording")
    ap.add_argument("--push", action="store_true", help="upload the result to the Hugging Face dataset repo")
    a = ap.parse_args()
    ids = all_logged_sessions(a.logs) if a.all else [a.session_id]
    if not ids or ids == [None]:
        ap.error("give a session id or --all")
    for sid in ids:
        try:
            r = build(sid, out_root=a.out, log_dir=a.logs, keep_raw=not a.no_raw)
        except Exception as exc:
            print(f"{sid}: FAILED - {exc}")
            continue
        q = r["quality"]
        print(f"{sid}: {r['relative_path']}  {q['epochs_usable']}/{q['epochs_total']} usable epochs, "
              f"{q['trials_valid']} valid trials. {'; '.join(q['verdict']['messages'])}")
        if a.push:
            try:
                print("uploaded:", store.push_session(sid, root=a.out, allow_simulated=r["simulated"])["url"])
            except store.StorageError as exc:
                print(f"UPLOAD FAILED: {exc}")


if __name__ == "__main__":
    main()

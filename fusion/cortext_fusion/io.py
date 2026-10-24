"""Load raw model outputs from files, folders or Python objects.

Every loader returns ``{trial_id: raw}`` where ``raw`` is whatever one trial's
output looks like (a list, an array, a dict of fields...). Adapters turn ``raw``
into scores. Loading never interprets the numbers; it only splits by trial.

Supported sources
-----------------
* ``.json``  – ``{trial_id: value}``, or a list of records with a trial id field
* ``.jsonl`` – one record per line
* ``.csv``   – long (several rows per trial) or wide (one row per trial)
* ``.npy`` / ``.npz`` – first axis is trials; trial ids from a sidecar file
  (``<stem>_trial_ids.txt|.json|.npy``), an ``trial_ids`` array inside the .npz,
  or ``trial_ids``/``trial_order`` in the config
* a folder of per-trial files (``<trial_id>.json|.npy|.csv``)
* in-memory: a dict ``{trial_id: raw}``, a pandas DataFrame, or a list of records
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

TRIAL_ID_KEYS = ("trial_key", "trial_id", "trial", "trialid", "trial_idx")
SESSION_KEYS = ("session_id", "session", "sessionid")
PER_TRIAL_SUFFIXES = {".json", ".npy", ".csv", ".jsonl"}


def _find_tid_key(keys) -> str | None:
    lower = {str(k).lower(): k for k in keys}
    for k in TRIAL_ID_KEYS:
        if k in lower:
            return lower[k]
    return None


def _find_session_key(keys) -> str | None:
    lower = {str(k).lower(): k for k in keys}
    for k in SESSION_KEYS:
        if k in lower:
            return lower[k]
    return None


def _key(rec: dict[str, Any], tid_key: str, sess_key: str | None) -> str:
    from .stimulus_log import make_key, norm_id
    if sess_key is not None and rec.get(sess_key) not in (None, "") and not (
            isinstance(rec.get(sess_key), float) and np.isnan(rec[sess_key])):
        return make_key(rec[sess_key], rec[tid_key])
    return norm_id(rec[tid_key])


def _record_to_raw(rec: dict[str, Any], tid_key: str, sess_key: str | None = None) -> Any:
    rest = {k: v for k, v in rec.items() if k not in (tid_key, sess_key)}
    if len(rest) == 1:
        return next(iter(rest.values()))
    return rest


def records_to_trials(records: list[dict[str, Any]], tid_key: str | None = None) -> dict[str, Any]:
    if not records:
        return {}
    tid_key = tid_key or _find_tid_key(records[0].keys())
    if tid_key is None:
        raise ValueError(f"records have no trial id field (looked for {TRIAL_ID_KEYS}); keys: {list(records[0])}")
    sess_key = _find_session_key(records[0].keys()) if tid_key.lower() != "trial_key" else None
    out: dict[str, Any] = {}
    multi: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        multi.setdefault(_key(rec, tid_key, sess_key), []).append(rec)
    for tid, recs in multi.items():
        if len(recs) == 1:
            out[tid] = _record_to_raw(recs[0], tid_key, sess_key)
        else:  # several records for one trial -> columns of lists (long format)
            cols = [k for k in recs[0] if k not in (tid_key, sess_key)]
            out[tid] = {c: [r.get(c) for r in recs] for c in cols}
    return out


def frame_to_trials(df: pd.DataFrame, tid_key: str | None = None) -> dict[str, Any]:
    tid_key = tid_key or _find_tid_key(df.columns)
    if tid_key is None:
        raise ValueError(f"table has no trial id column (looked for {TRIAL_ID_KEYS}); columns: {list(df.columns)}")
    sess_key = _find_session_key(df.columns) if str(tid_key).lower() != "trial_key" else None
    df = df.copy()
    keys = [_key(r, tid_key, sess_key) for r in df[[c for c in (tid_key, sess_key) if c]].to_dict("records")]
    df = df.drop(columns=[c for c in (tid_key, sess_key) if c])
    df["__key"] = keys
    out: dict[str, Any] = {}
    for tid, g in df.groupby("__key", sort=False):
        g = g.drop(columns=["__key"])
        if len(g) == 1:
            out[tid] = {c: _py(g[c].iloc[0]) for c in g.columns}
        else:
            out[tid] = {c: [_py(v) for v in g[c].tolist()] for c in g.columns}
    return out


def _py(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):  # a list stored in a CSV cell
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return v
    return v


def _sidecar_ids(path: Path) -> list[str] | None:
    for suf in (".txt", ".json", ".npy", ".csv"):
        cand = path.with_name(path.stem + "_trial_ids" + suf)
        if cand.exists():
            if suf == ".txt":
                return [l.strip() for l in cand.read_text().splitlines() if l.strip()]
            if suf == ".json":
                return [str(x) for x in json.loads(cand.read_text())]
            if suf == ".npy":
                return [str(x) for x in np.load(cand, allow_pickle=False).tolist()]
            return [str(x) for x in pd.read_csv(cand).iloc[:, 0].tolist()]
    return None


def array_to_trials(arr: np.ndarray, trial_ids: list[str] | None) -> dict[str, Any]:
    if trial_ids is None:
        raise ValueError(
            "A .npy file has no trial ids. Provide a sidecar '<name>_trial_ids.txt' (one id per line), "
            "a 'trial_ids' array in an .npz, or set 'trial_order: stimulus_log' in the config "
            "(only if rows are guaranteed to follow the stimulus log order)."
        )
    if len(trial_ids) != arr.shape[0]:
        raise ValueError(f"array has {arr.shape[0]} rows but {len(trial_ids)} trial ids were given")
    return {str(t): arr[i] for i, t in enumerate(trial_ids)}


def load_file(path: str | Path, trial_ids: list[str] | None = None) -> dict[str, Any]:
    """Load one batch file (all trials) into ``{trial_id: raw}``."""
    p = Path(path)
    suf = p.suffix.lower()
    if p.is_dir():
        return load_folder(p)
    if suf == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, list):
            if data and isinstance(data[0], dict):
                return records_to_trials(data)
            return array_to_trials(np.asarray(data, dtype=float), trial_ids or _sidecar_ids(p))
        if isinstance(data, dict):
            for k in ("trials", "outputs", "predictions", "results"):
                if k in data and isinstance(data[k], (list, dict)):
                    inner = data[k]
                    return records_to_trials(inner) if isinstance(inner, list) else {str(a): b for a, b in inner.items()}
            return {str(k): v for k, v in data.items()}
        raise ValueError(f"{p}: unsupported JSON top-level type {type(data).__name__}")
    if suf == ".jsonl":
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        return records_to_trials(recs)
    if suf == ".csv":
        return frame_to_trials(pd.read_csv(p))
    if suf == ".npy":
        arr = np.load(p, allow_pickle=False)
        return array_to_trials(arr, trial_ids or _sidecar_ids(p))
    if suf == ".npz":
        z = np.load(p, allow_pickle=False)
        keys = list(z.keys())
        ids = trial_ids or ([str(x) for x in z["trial_ids"].tolist()] if "trial_ids" in z else _sidecar_ids(p))
        data_keys = [k for k in keys if k != "trial_ids"]
        if len(data_keys) == 1:
            return array_to_trials(z[data_keys[0]], ids)
        arrays = {k: z[k] for k in data_keys}
        if ids is None:
            raise ValueError(f"{p}: .npz has several arrays but no 'trial_ids'")
        return {str(t): {k: a[i] for k, a in arrays.items()} for i, t in enumerate(ids)}
    raise ValueError(f"{p}: unsupported file type '{suf}' (use .json, .jsonl, .csv, .npy, .npz or a folder)")


def load_single_trial_file(path: str | Path) -> Any:
    """Load one per-trial file; the trial id comes from the file name."""
    p = Path(path)
    suf = p.suffix.lower()
    if suf == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            k = _find_tid_key(data.keys())
            if k is not None:
                return _record_to_raw(data, k, _find_session_key(data.keys()))
        return data
    if suf == ".npy":
        return np.load(p, allow_pickle=False)
    if suf == ".csv":
        df = pd.read_csv(p)
        k = _find_tid_key(df.columns)
        if k is not None:
            df = df.drop(columns=[c for c in (k, _find_session_key(df.columns)) if c])
        if len(df) == 1:
            return {c: _py(df[c].iloc[0]) for c in df.columns}
        return {c: [_py(v) for v in df[c].tolist()] for c in df.columns}
    if suf == ".jsonl":
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        k = _find_tid_key(recs[0].keys()) if recs else None
        cols = [c for c in recs[0] if c != k] if recs else []
        return {c: [r.get(c) for r in recs] for c in cols}
    raise ValueError(f"{p}: unsupported per-trial file type '{suf}'")


def trial_id_from_name(path: Path, pattern: str | None = None) -> str:
    import re
    stem = path.stem
    if pattern:
        m = re.search(pattern, stem)
        if not m:
            raise ValueError(f"file name {path.name!r} does not match trial id pattern {pattern!r}")
        return m.group(1) if m.groups() else m.group(0)
    return stem


def load_folder(folder: str | Path, pattern: str | None = None) -> dict[str, Any]:
    folder = Path(folder)
    out: dict[str, Any] = {}
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() in PER_TRIAL_SUFFIXES and p.is_file() and not p.name.startswith("."):
            out[trial_id_from_name(p, pattern)] = load_single_trial_file(p)
    return out


def load_any(source: Any, trial_ids: list[str] | None = None, pattern: str | None = None) -> dict[str, Any]:
    """Load from a path, folder or in-memory object."""
    if isinstance(source, (str, Path)):
        p = Path(source)
        if not p.exists():
            raise FileNotFoundError(f"model output not found: {p}")
        return load_folder(p, pattern) if p.is_dir() else load_file(p, trial_ids)
    if isinstance(source, pd.DataFrame):
        return frame_to_trials(source)
    if isinstance(source, dict):
        return {str(k): v for k, v in source.items()}
    if isinstance(source, list) and source and isinstance(source[0], dict):
        return records_to_trials(source)
    if isinstance(source, np.ndarray):
        return array_to_trials(source, trial_ids)
    raise TypeError(f"cannot load model output from {type(source).__name__}")


def watch_folder(
    folder: str | Path,
    callback: Callable[[str, Any], None] | None = None,
    poll_seconds: float = 0.25,
    pattern: str | None = None,
    stop: Callable[[], bool] | None = None,
    settle_seconds: float = 0.1,
) -> Iterator[tuple[str, Any]]:
    """Poll a folder and yield ``(trial_id, raw)`` for each new per-trial file.

    Uses plain polling (no extra dependency). A file is read once its size has
    stopped changing for ``settle_seconds``, so half-written files are skipped.
    """
    folder = Path(folder)
    seen: dict[Path, float] = {}
    sizes: dict[Path, int] = {}
    done: set[Path] = set()
    while True:
        if stop is not None and stop():
            return
        for p in sorted(folder.iterdir()) if folder.exists() else []:
            if p in done or not p.is_file() or p.suffix.lower() not in PER_TRIAL_SUFFIXES or p.name.startswith("."):
                continue
            size = p.stat().st_size
            prev = seen.get(p)
            now = time.monotonic()
            if prev is None or size != sizes.get(p):
                sizes[p] = size
                seen[p] = now
                continue
            if now - prev < settle_seconds:
                continue
            try:
                raw = load_single_trial_file(p)
            except (json.JSONDecodeError, ValueError, OSError):
                continue  # still being written; retry next poll
            done.add(p)
            tid = trial_id_from_name(p, pattern)
            if callback:
                callback(tid, raw)
            yield tid, raw
        time.sleep(poll_seconds)

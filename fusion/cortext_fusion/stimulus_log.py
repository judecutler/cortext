"""Load the stimulus app's logs into :class:`TrialMeta` objects.

Primary format: **CorText Lab exports** (``cortext-lab/`` in this repo). Each
session exports ``session.json``, ``trials.csv`` and ``markers.csv`` (schema in
``cortext-lab/web/js/schema.js``). Use :func:`load_lab_export` with a session
folder, a folder of session folders ("Export all"), or the downloaded ``.zip``.

* only ``event == flash_on`` marker rows are flashes (other events are ignored)
* trial ids restart at 0 in every session, so trials are keyed
  ``"<session_id>:<trial_id>"`` (the app's own key format). Model outputs can
  use that key, ``session_id`` + ``trial_id`` columns, or the bare trial id
  when only one session is loaded (see ``pipeline.match_trial_ids``).
* option order = ``option_ids_by_position`` (on-screen position A, B, C, D)
* ground truth = ``true_answer_id`` and ``clicked_answer_id``; ``valid``
  and ``invalid_reason`` are kept, and invalid trials are skipped in evaluation
* participant comes from ``session.json`` (needed for hold-out by participant)
* trials with no flash markers (aborted / never run) are kept as invalid

A generic marker log + trial table (any column names from ``ALIASES``) also
works via :func:`load_stimulus_log`.
"""
from __future__ import annotations

import io as _io
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .schema import Flash, TrialMeta

KEY_SEP = ":"

ALIASES: dict[str, list[str]] = {
    "trial_id": ["trial_id", "trial", "trialid", "trial_idx"],
    "option_id": ["option_id", "option", "optionid", "opt"],
    "category": ["category", "true_category", "option_category", "concept"],
    "is_target": ["is_target", "target", "istarget"],
    "repeat_index": ["repeat_index", "repeat", "repetition", "rep", "repeat_idx"],
    "timestamp": ["t_ms", "timestamp", "time", "t", "onset", "lsl_time", "wall_ms"],
    "event": ["event", "event_type", "marker_type"],
    "session_id": ["session_id", "session", "sessionid"],
    "participant_id": ["participant_id", "participant", "subject", "subject_id", "pid"],
    "option_order": ["option_ids_by_position", "option_order", "option_ids", "options"],
    "target_category": ["true_answer_category", "target_category", "prompt_category", "question_category"],
    "target_option": ["true_answer_id", "target_option", "true_option", "correct_option", "answer"],
    "clicked_option": ["clicked_answer_id", "clicked_option", "clicked", "response", "declared_option", "choice"],
    "valid": ["valid", "is_valid"],
    "invalid": ["invalid", "is_invalid", "discarded", "discard"],
    "invalid_reason": ["invalid_reason", "reason", "discard_reason"],
    "prompt": ["prompt", "question"],
}
FLASH_EVENTS = {"flash_on", "flash", "flash_onset", "stim_on"}


def make_key(session_id: Any, trial_id: Any) -> str:
    return f"{norm_id(session_id)}{KEY_SEP}{norm_id(trial_id)}"


def norm_id(v: Any) -> str:
    """'3', 3, 3.0 and '3.0' all become '3' (CSV readers turn int columns with gaps into floats)."""
    if isinstance(v, (float, np.floating)) and float(v).is_integer():
        return str(int(v))
    s = str(v).strip()
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        return s[:-2]
    return s


def _resolve(df: pd.DataFrame, key: str, required: bool = False) -> str | None:
    lower = {str(c).lower().strip(): c for c in df.columns}
    for alias in ALIASES[key]:
        if alias in lower:
            return lower[alias]
    if required:
        raise ValueError(
            f"Stimulus log is missing a '{key}' column (accepted names: {ALIASES[key]}). "
            f"Columns found: {list(df.columns)}"
        )
    return None


def _truthy(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y", "t"}
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    return bool(v)


def _missing(v: Any) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or (isinstance(v, str) and v.strip() == "")


def _parse_list(v: Any) -> list[str] | None:
    if _missing(v):
        return None
    if isinstance(v, (list, tuple)):
        return [norm_id(x) for x in v]
    s = str(v).strip()
    if s.startswith("["):
        try:
            return [norm_id(x) for x in json.loads(s)]
        except json.JSONDecodeError:
            pass
    for sep in ("|", ";", ","):
        if sep in s:
            return [x.strip() for x in s.split(sep) if x.strip()]
    return [s]


def _read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            data = data.get("markers", data.get("trials", data))
        return pd.DataFrame(data)
    if p.suffix.lower() == ".jsonl":
        return pd.read_json(p, lines=True)
    return pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""])


def load_stimulus_log(
    markers: str | Path | pd.DataFrame,
    trials: str | Path | pd.DataFrame | None = None,
    key_mode: str = "auto",
    participants: dict[str, str] | None = None,
) -> dict[str, TrialMeta]:
    """Return ``{trial_key: TrialMeta}`` from a marker log (+ optional trial table).

    ``key_mode``: ``auto`` (``session:trial`` only if trial ids repeat across
    sessions), ``session`` (always ``session:trial``), or ``trial`` (bare id).
    A path to a CorText Lab export (folder or .zip) is accepted as ``markers``.
    """
    if isinstance(markers, (str, Path)):
        p = Path(markers)
        if p.is_dir() or p.suffix.lower() == ".zip":
            return load_lab_export(p)
        if p.suffix.lower() == ".json" and trials is None:
            data = json.loads(p.read_text())
            if isinstance(data, dict) and "markers" in data:
                mdf = pd.DataFrame(data["markers"])
                tdf = pd.DataFrame(data["trials"]) if data.get("trials") else None
                return build_trial_metas(mdf, tdf, key_mode, participants)
    mdf = markers if isinstance(markers, pd.DataFrame) else _read_table(markers)
    tdf = None
    if trials is not None:
        tdf = trials if isinstance(trials, pd.DataFrame) else _read_table(trials)
    return build_trial_metas(mdf, tdf, key_mode, participants)


def build_trial_metas(mdf: pd.DataFrame, tdf: pd.DataFrame | None = None, key_mode: str = "auto",
                      participants: dict[str, str] | None = None) -> dict[str, TrialMeta]:
    c_trial = _resolve(mdf, "trial_id", required=True)
    c_opt = _resolve(mdf, "option_id", required=True)
    c_cat = _resolve(mdf, "category")
    c_tgt = _resolve(mdf, "is_target")
    c_rep = _resolve(mdf, "repeat_index")
    c_ts = _resolve(mdf, "timestamp")
    c_ev = _resolve(mdf, "event")
    c_sess = _resolve(mdf, "session_id")
    c_part = _resolve(mdf, "participant_id")
    participants = participants or {}

    mdf = mdf.copy()
    if c_ev:  # only flash onsets are flashes
        mdf = mdf[mdf[c_ev].astype(str).str.strip().str.lower().isin(FLASH_EVENTS)]
    mdf = mdf[~mdf[c_trial].map(_missing) & ~mdf[c_opt].map(_missing)]
    mdf[c_trial] = mdf[c_trial].map(norm_id)
    mdf[c_opt] = mdf[c_opt].map(norm_id)
    if c_sess:
        mdf[c_sess] = mdf[c_sess].map(lambda v: "" if _missing(v) else norm_id(v))

    t_trial = t_sess = None
    if tdf is not None and len(tdf):
        t_trial = _resolve(tdf, "trial_id", required=True)
        t_sess = _resolve(tdf, "session_id")

    # decide the key
    if key_mode not in ("auto", "session", "trial"):
        raise ValueError("key_mode must be auto, session or trial")
    use_session = key_mode == "session"
    if key_mode == "auto" and c_sess:
        per_trial_sessions = mdf.groupby(c_trial)[c_sess].nunique()
        use_session = bool((per_trial_sessions > 1).any())
    if use_session and not c_sess:
        raise ValueError("key_mode='session' needs a session_id column in the marker log")

    def key_of(sess: Any, tid: Any) -> str:
        return make_key(sess, tid) if use_session else norm_id(tid)

    trial_rows: dict[str, dict[str, Any]] = {}
    if t_trial is not None:
        for rec in tdf.to_dict("records"):
            if _missing(rec[t_trial]):
                continue
            k = key_of(rec.get(t_sess) if t_sess else None, rec[t_trial])
            trial_rows[k] = {str(c).lower().strip(): v for c, v in rec.items()}

    def tget(row: dict[str, Any] | None, key: str) -> Any:
        if not row:
            return None
        for alias in ALIASES[key]:
            if alias in row:
                v = row[alias]
                return None if _missing(v) else v
        return None

    mdf["__key"] = [key_of(s, t) for s, t in zip(mdf[c_sess] if c_sess else [None] * len(mdf), mdf[c_trial])]
    if c_ts:
        mdf["__ts"] = pd.to_numeric(mdf[c_ts], errors="coerce")
        mdf = mdf.sort_values(["__key", "__ts"], kind="stable")

    metas: dict[str, TrialMeta] = {}
    for key, g in mdf.groupby("__key", sort=False):
        trow = trial_rows.get(key)
        order = _parse_list(tget(trow, "option_order"))
        seen_opts = list(dict.fromkeys(g[c_opt].tolist()))
        if order is None:
            order = sorted(seen_opts)
        else:
            missing = set(seen_opts) - set(order)
            if missing:
                raise ValueError(f"trial {key}: marker option_ids {sorted(missing)} not in option order {order}")
        cats: dict[str, str] = {}
        if c_cat:
            for o, c in zip(g[c_opt], g[c_cat]):
                if not _missing(c):
                    cats.setdefault(o, str(c))
        target_opt = tget(trow, "target_option")
        if target_opt is None and c_tgt:
            tg = g[g[c_tgt].map(_truthy)][c_opt].unique().tolist()
            if len(tg) > 1:
                raise ValueError(f"trial {key}: more than one option marked is_target: {tg}")
            target_opt = tg[0] if tg else None
        target_cat = tget(trow, "target_category")
        if target_cat is None and target_opt is not None:
            target_cat = cats.get(norm_id(target_opt))
        valid, reason = _validity(trow, tget)
        flashes = []
        for rec in g.to_dict("records"):
            rep = rec.get(c_rep) if c_rep else None
            ts = rec.get("__ts")
            flashes.append(Flash(option_id=rec[c_opt], repeat_index=int(float(rep)) if not _missing(rep) else -1,
                                 timestamp=None if ts is None or (isinstance(ts, float) and np.isnan(ts)) else float(ts)))
        if c_rep is None or any(f.repeat_index < 0 for f in flashes):  # infer: k-th appearance of each option
            counts: dict[str, int] = {}
            for f in flashes:
                f.repeat_index = counts.get(f.option_id, 0)
                counts[f.option_id] = f.repeat_index + 1
        sess = tget(trow, "session_id") or (g[c_sess].iloc[0] if c_sess else None) or None
        part = tget(trow, "participant_id") or (str(g[c_part].iloc[0]) if c_part and not _missing(g[c_part].iloc[0]) else None)
        part = part or participants.get(str(sess)) if sess else part
        clicked = tget(trow, "clicked_option")
        metas[key] = TrialMeta(
            trial_id=key,
            option_ids=[norm_id(o) for o in order],
            option_categories=[cats.get(norm_id(o), "") for o in order],
            target_category=None if target_cat is None else str(target_cat),
            target_option=None if target_opt is None else norm_id(target_opt),
            clicked_option=None if clicked is None else norm_id(clicked),
            valid=valid, invalid_reason=reason,
            session_id=None if sess is None else str(sess),
            participant_id=None if part is None else str(part),
            prompt=tget(trow, "prompt"), flashes=flashes,
            local_trial_id=norm_id(g[c_trial].iloc[0]),
        )
    # trials in the table with no flash markers: kept, marked invalid (never deleted)
    for key, trow in trial_rows.items():
        if key in metas:
            continue
        order = _parse_list(tget(trow, "option_order"))
        if not order:
            continue
        _, reason = _validity(trow, tget)
        sess = tget(trow, "session_id")
        metas[key] = TrialMeta(
            trial_id=key, option_ids=order, option_categories=[""] * len(order),
            target_category=tget(trow, "target_category"),
            target_option=None if tget(trow, "target_option") is None else norm_id(tget(trow, "target_option")),
            clicked_option=None if tget(trow, "clicked_option") is None else norm_id(tget(trow, "clicked_option")),
            valid=False, invalid_reason=reason or "no flash markers",
            session_id=None if sess is None else str(sess),
            participant_id=participants.get(str(sess)) if sess else None,
            prompt=tget(trow, "prompt"), flashes=[], local_trial_id=norm_id(trow.get(t_trial.lower().strip(), key)),
        )
    return metas


def _validity(trow, tget) -> tuple[bool, str | None]:
    valid = True
    v = tget(trow, "valid")
    if v is not None:
        valid = _truthy(v)
    inv = tget(trow, "invalid")
    if inv is not None and _truthy(inv):
        valid = False
    reason = tget(trow, "invalid_reason")
    return valid, (None if valid else (str(reason) if reason is not None else "unspecified"))


# --------------------------------------------------------------------------- CorText Lab exports

def _session_bundles(path: Path) -> list[dict[str, Any]]:
    """Find (session.json, trials.csv, markers.csv) triples in a folder tree or a zip."""
    bundles: dict[str, dict[str, Any]] = {}
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                base = name.rsplit("/", 1)[-1]
                folder = name[: -len(base)]
                for kind in ("session.json", "trials.csv", "markers.csv"):
                    if base == kind or base.endswith("_" + kind):
                        bundles.setdefault(folder + base[: -len(kind)], {})[kind] = z.read(name)
    else:
        for kind in ("session.json", "trials.csv", "markers.csv"):
            for f in sorted(path.rglob(f"*{kind}")):
                if f.name == kind or f.name.endswith("_" + kind):
                    prefix = str(f.parent / f.name[: -len(kind)])
                    bundles.setdefault(prefix, {})[kind] = f.read_bytes()
    return [b for _, b in sorted(bundles.items()) if "markers.csv" in b]


def load_lab_export(paths: str | Path | Sequence[str | Path]) -> dict[str, TrialMeta]:
    """Load one or more CorText Lab exports (session folder, folder of sessions, or .zip)."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    metas: dict[str, TrialMeta] = {}
    n = 0
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"CorText Lab export not found: {p}")
        for b in _session_bundles(p):
            n += 1
            participants = {}
            if "session.json" in b:
                sj = json.loads(b["session.json"])
                s = sj.get("session", sj)
                if s.get("session_id") is not None:
                    participants[str(s["session_id"])] = str(s.get("participant_id") or "")
            mdf = pd.read_csv(_io.BytesIO(b["markers.csv"]), dtype=str, keep_default_na=False, na_values=[""])
            tdf = pd.read_csv(_io.BytesIO(b["trials.csv"]), dtype=str, keep_default_na=False, na_values=[""]) \
                if "trials.csv" in b else None
            for k, m in build_trial_metas(mdf, tdf, key_mode="session", participants=participants).items():
                if k in metas:
                    raise ValueError(f"trial {k} appears in two exports; load each session once")
                metas[k] = m
    if n == 0:
        raise ValueError(f"no CorText Lab session exports (markers.csv) found under {list(map(str, paths))}")
    return metas


def metas_to_frame(metas: Iterable[TrialMeta]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "trial_key": m.trial_id, "trial_id": m.local_trial_id, "session_id": m.session_id,
            "participant_id": m.participant_id, "option_order": "|".join(m.option_ids),
            "option_categories": "|".join(m.option_categories), "target_category": m.target_category,
            "target_option": m.target_option, "clicked_option": m.clicked_option, "valid": m.valid,
            "invalid_reason": m.invalid_reason, "n_flashes": len(m.flashes), "n_repeats": m.n_repeats,
        }
        for m in metas
    ])

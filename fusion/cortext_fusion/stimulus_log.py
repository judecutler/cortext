"""Load the stimulus app's ("CorText Lab") logs into :class:`TrialMeta` objects.

Two inputs, matching the web-app spec:

* **Marker log** (required): one row per flash onset with
  ``trial_id, option_id, true_category, is_target, repeat_index`` and optionally
  ``timestamp, session_id, participant_id``.
* **Trial table** (optional): one row per trial with any of
  ``option_order, target_category, target_option, clicked_option, valid,
  invalid_reason, session_id, participant_id, prompt``.

Column names are matched case-insensitively against a list of aliases, so small
naming differences in the final web app do not break loading. A JSON session
export ``{"trials": [...], "markers": [...]}`` is accepted too.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .schema import Flash, TrialMeta

ALIASES: dict[str, list[str]] = {
    "trial_id": ["trial_id", "trial", "trialid", "trial_idx"],
    "option_id": ["option_id", "option", "optionid", "opt"],
    "category": ["true_category", "category", "option_category", "label", "concept"],
    "is_target": ["is_target", "target", "istarget"],
    "repeat_index": ["repeat_index", "repeat", "repetition", "rep", "repeat_idx"],
    "timestamp": ["timestamp", "time", "t", "onset", "lsl_time"],
    "session_id": ["session_id", "session", "sessionid"],
    "participant_id": ["participant_id", "participant", "subject", "subject_id", "pid"],
    "option_order": ["option_order", "option_ids", "options"],
    "target_category": ["target_category", "prompt_category", "question_category"],
    "target_option": ["target_option", "true_option", "correct_option", "answer"],
    "clicked_option": ["clicked_option", "clicked", "response", "declared_option", "choice"],
    "valid": ["valid", "is_valid"],
    "invalid": ["invalid", "is_invalid", "discarded", "discard"],
    "invalid_reason": ["invalid_reason", "reason", "discard_reason"],
    "prompt": ["prompt", "question"],
}


def _resolve(df: pd.DataFrame, key: str, required: bool = False) -> str | None:
    lower = {c.lower().strip(): c for c in df.columns}
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


def _parse_list(v: Any) -> list[str] | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    s = str(v).strip()
    if s.startswith("["):
        try:
            return [str(x) for x in json.loads(s)]
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
    return pd.read_csv(p)


def load_stimulus_log(
    markers: str | Path | pd.DataFrame,
    trials: str | Path | pd.DataFrame | None = None,
) -> dict[str, TrialMeta]:
    """Return ``{trial_id: TrialMeta}`` built from the marker log (+ optional trial table)."""
    if isinstance(markers, (str, Path)) and Path(markers).suffix.lower() == ".json" and trials is None:
        data = json.loads(Path(markers).read_text())
        if isinstance(data, dict) and "markers" in data:
            mdf = pd.DataFrame(data["markers"])
            tdf = pd.DataFrame(data["trials"]) if data.get("trials") else None
            return build_trial_metas(mdf, tdf)
    mdf = markers if isinstance(markers, pd.DataFrame) else _read_table(markers)
    tdf = None
    if trials is not None:
        tdf = trials if isinstance(trials, pd.DataFrame) else _read_table(trials)
    return build_trial_metas(mdf, tdf)


def build_trial_metas(mdf: pd.DataFrame, tdf: pd.DataFrame | None = None) -> dict[str, TrialMeta]:
    c_trial = _resolve(mdf, "trial_id", required=True)
    c_opt = _resolve(mdf, "option_id", required=True)
    c_cat = _resolve(mdf, "category")
    c_tgt = _resolve(mdf, "is_target")
    c_rep = _resolve(mdf, "repeat_index")
    c_ts = _resolve(mdf, "timestamp")
    c_sess = _resolve(mdf, "session_id")
    c_part = _resolve(mdf, "participant_id")

    trial_rows: dict[str, dict[str, Any]] = {}
    if tdf is not None and len(tdf):
        t_trial = _resolve(tdf, "trial_id", required=True)
        for rec in tdf.to_dict("records"):
            trial_rows[str(rec[t_trial])] = {k.lower().strip(): v for k, v in rec.items()}

    def tget(row: dict[str, Any] | None, key: str) -> Any:
        if not row:
            return None
        for alias in ALIASES[key]:
            if alias in row:
                v = row[alias]
                if isinstance(v, float) and np.isnan(v):
                    return None
                return v
        return None

    metas: dict[str, TrialMeta] = {}
    mdf = mdf.copy()
    mdf[c_trial] = mdf[c_trial].astype(str)
    mdf[c_opt] = mdf[c_opt].astype(str)
    if c_ts:
        mdf = mdf.sort_values([c_trial, c_ts], kind="stable")
    for tid, g in mdf.groupby(c_trial, sort=False):
        trow = trial_rows.get(tid)
        order = _parse_list(tget(trow, "option_order"))
        seen_opts = list(dict.fromkeys(g[c_opt].tolist()))
        if order is None:
            order = sorted(seen_opts)
        else:
            missing = set(seen_opts) - set(order)
            if missing:
                raise ValueError(f"trial {tid}: marker option_ids {sorted(missing)} not in option_order {order}")
        cats: dict[str, str] = {}
        if c_cat:
            for o, c in zip(g[c_opt], g[c_cat]):
                cats.setdefault(o, str(c))
        target_opt = tget(trow, "target_option")
        if target_opt is None and c_tgt:
            tg = g[g[c_tgt].map(_truthy)][c_opt].unique().tolist()
            if len(tg) > 1:
                raise ValueError(f"trial {tid}: more than one option marked is_target: {tg}")
            target_opt = tg[0] if tg else None
        target_cat = tget(trow, "target_category")
        if target_cat is None and target_opt is not None:
            target_cat = cats.get(str(target_opt))
        valid = True
        v = tget(trow, "valid")
        if v is not None:
            valid = _truthy(v)
        inv = tget(trow, "invalid")
        if inv is not None and _truthy(inv):
            valid = False
        flashes = []
        for i, rec in enumerate(g.to_dict("records")):
            rep = int(rec[c_rep]) if c_rep and not pd.isna(rec[c_rep]) else None
            flashes.append(Flash(option_id=str(rec[c_opt]), repeat_index=rep if rep is not None else -1,
                                 timestamp=float(rec[c_ts]) if c_ts and not pd.isna(rec[c_ts]) else None))
        if c_rep is None:  # infer repeat index as the k-th appearance of each option
            counts: dict[str, int] = {}
            for f in flashes:
                f.repeat_index = counts.get(f.option_id, 0)
                counts[f.option_id] = f.repeat_index + 1
        sess = tget(trow, "session_id") or (str(g[c_sess].iloc[0]) if c_sess else None)
        part = tget(trow, "participant_id") or (str(g[c_part].iloc[0]) if c_part else None)
        clicked = tget(trow, "clicked_option")
        metas[tid] = TrialMeta(
            trial_id=tid,
            option_ids=[str(o) for o in order],
            option_categories=[cats.get(str(o), "") for o in order],
            target_category=None if target_cat is None else str(target_cat),
            target_option=None if target_opt is None else str(target_opt),
            clicked_option=None if clicked is None else str(clicked),
            valid=valid,
            invalid_reason=tget(trow, "invalid_reason"),
            session_id=None if sess is None else str(sess),
            participant_id=None if part is None else str(part),
            prompt=tget(trow, "prompt"),
            flashes=flashes,
        )
    # trials present in the trial table but with no markers
    for tid in trial_rows:
        if tid not in metas:
            raise ValueError(f"trial {tid} is in the trial table but has no flash markers")
    return metas


def metas_to_frame(metas: Iterable[TrialMeta]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "trial_id": m.trial_id, "session_id": m.session_id, "participant_id": m.participant_id,
            "option_order": "|".join(m.option_ids), "option_categories": "|".join(m.option_categories),
            "target_category": m.target_category, "target_option": m.target_option,
            "clicked_option": m.clicked_option, "valid": m.valid, "invalid_reason": m.invalid_reason,
            "n_flashes": len(m.flashes), "n_repeats": m.n_repeats,
        }
        for m in metas
    ])

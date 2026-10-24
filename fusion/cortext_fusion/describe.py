"""`describe`: inspect a sample model output and suggest an adapter + config snippet."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .adapters import ADAPTERS, AdapterError, _is_numeric_seq, detect_candidates, make_adapter
from .io import load_any
from .schema import TrialMeta


INDEX_FIELDS = {"repeat_index", "repeat", "repetition", "rep", "flash_index", "flash", "option_index", "index"}


def _common_prefix(keys: list[str]) -> str | None:
    import os
    if len(keys) < 2:
        return None
    pre = os.path.commonprefix(keys)
    return pre if pre and pre[-1] in "_-:." else None


def _flatten_numbers(x: Any) -> np.ndarray:
    if isinstance(x, dict):
        parts = [_flatten_numbers(v) for k, v in x.items() if str(k).lower() not in INDEX_FIELDS]
        parts = [p for p in parts if p.size]
        return np.concatenate(parts) if parts else np.array([])
    if _is_numeric_seq(x):
        return np.asarray(x, dtype=float).reshape(-1)
    if isinstance(x, (int, float, np.number)) and not isinstance(x, bool):
        return np.array([float(x)])
    return np.array([])


def _shape(x: Any) -> str:
    if isinstance(x, dict):
        inner = ", ".join(f"{k}: {_shape(v)}" for k, v in list(x.items())[:10])
        return "{" + inner + (", ..." if len(x) > 10 else "") + "}"
    if isinstance(x, np.ndarray):
        return f"array{list(x.shape)} {x.dtype}"
    if isinstance(x, (list, tuple)):
        if x and _is_numeric_seq(x):
            a = np.asarray(x, dtype=float)
            return f"list{list(a.shape)} numeric"
        return f"list[{len(x)}] of {type(x[0]).__name__ if x else '?'}"
    return type(x).__name__


def describe(path: str | Path, metas: Mapping[str, TrialMeta] | None = None, class_names: list[str] | None = None,
             n_options: int = 4, model_name: str = "my_model") -> str:
    raw = load_any(path)
    lines = [f"File: {path}", f"Trials found: {len(raw)}"]
    if not raw:
        return "\n".join(lines + ["No trials found."])
    ids = list(raw)
    lines.append(f"Trial id examples: {ids[:3]}")
    first = raw[ids[0]]
    lines.append(f"One trial looks like: {_shape(first)}")
    allv = np.concatenate([_flatten_numbers(raw[t]) for t in ids[:500]]) if ids else np.array([])
    if allv.size:
        fin = allv[np.isfinite(allv)]
        lines.append(f"Numeric values (first {min(len(ids), 500)} trials): n={allv.size}, NaN/inf={allv.size - fin.size}, "
                     f"min={fin.min():.4g}, max={fin.max():.4g}, mean={fin.mean():.4g}, std={fin.std():.4g}")
        lines.append(f"All values in [0,1]: {bool(fin.size and fin.min() >= 0 and fin.max() <= 1)}")
    if metas:
        common = [t for t in ids if t in metas]
        lines.append(f"Trial ids matching the stimulus log: {len(common)}/{len(ids)} "
                     f"(log has {len(metas)})")
        if not common:
            lines.append(f"  !! No ids match. Log ids look like {list(metas)[:3]}.")
            return "\n".join(lines)
        meta = metas[common[0]]
        sample_ids = common
    else:
        oid = [chr(ord("A") + i) for i in range(n_options)]
        meta = TrialMeta(ids[0], oid, [""] * n_options)
        sample_ids = ids
        lines.append(f"(No stimulus log given; assuming {n_options} options {oid} and no flash info. "
                     "Pass --markers for a better guess.)")
    params = {"class_names": class_names} if class_names else {}
    cands = detect_candidates(first if not metas else raw[sample_ids[0]], meta, params)
    lines.append("")
    if not cands:
        lines.append("Could not recognise this format. See README 'Writing your own adapter'.")
        return "\n".join(lines)
    lines.append("Adapter candidates (best first):")
    best_cfg = None
    for name, conf, reason in cands:
        opts: dict[str, Any] = {}
        if name == "class_probabilities" and class_names:
            opts["class_names"] = class_names
        if isinstance(first, dict):
            pre = _common_prefix([str(k) for k in first])
            if pre and name in ("class_probabilities", "per_option", "probabilities"):
                opts["key_prefix"] = pre
        status = ""
        if metas and name != "embedding":
            ok = fail = 0
            err = None
            try:
                ad = make_adapter(name, **opts)
                for t in sample_ids[:200]:
                    try:
                        ad.to_trial_scores(raw[t], metas[t])
                        ok += 1
                    except AdapterError as e:
                        fail += 1
                        err = err or str(e)
            except AdapterError as e:
                err, fail = str(e), 1
            status = f" -> converts {ok}/{ok + fail} trials" + (f"; first error: {err}" if err else "")
        lines.append(f"  {name:20s} confidence {conf:.2f}: {reason}{status}")
        if best_cfg is None:
            best_cfg = {"name": model_name, "source": str(path), "adapter": name}
            if opts:
                best_cfg["options"] = opts
            if name == "class_probabilities" and not class_names and not isinstance(first, dict):
                best_cfg["options"] = {"class_names": ["<class 0>", "<class 1>", "..."]}
            if name == "embedding":
                best_cfg["options"] = {"embeddings": "<path to {category: vector} json>"}
    lines += ["", "Suggested config entry (check it, then paste under 'models:'):", "",
              yaml.safe_dump([best_cfg], sort_keys=False).rstrip(),
              "", "If higher values mean LESS likely (distances, losses), add `lower_is_better: true`."]
    return "\n".join(lines)

"""Glue: stimulus log + model outputs + adapters -> per-model score tables + diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .adapters import Adapter, AdapterError
from .config import Config, ModelConfig
from .fusion import CONSTANT_TOL, Fuser
from .io import load_any
from .schema import FusedPrediction, TrialMeta, TrialScores
from .stimulus_log import load_stimulus_log


@dataclass
class Issue:
    level: str        # "error" | "warning" | "info"
    code: str         # machine-readable, e.g. "missing_trials"
    model: str | None
    message: str
    trial_ids: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        who = f"[{self.model}] " if self.model else ""
        more = ""
        if self.trial_ids:
            shown = ", ".join(self.trial_ids[:5])
            more = f" (e.g. {shown}{', ...' if len(self.trial_ids) > 5 else ''})"
        return f"{self.level.upper():7s} {who}{self.message}{more}"


@dataclass
class ModelData:
    config: ModelConfig
    adapter: Adapter
    raw: dict[str, Any]
    scores: dict[str, TrialScores]


@dataclass
class Dataset:
    metas: dict[str, TrialMeta]
    models: dict[str, ModelData]
    issues: list[Issue]

    def score_table(self, max_repeat: int | None = None) -> dict[str, dict[str, np.ndarray]]:
        """``{model: {trial_id: scores}}``; with ``max_repeat`` per-flash models are re-scored on the first k repeats."""
        out: dict[str, dict[str, np.ndarray]] = {}
        for name, md in self.models.items():
            if max_repeat is None or not md.adapter.per_flash:
                out[name] = {t: s.scores for t, s in md.scores.items()}
                continue
            tab = {}
            for t, raw in md.raw.items():
                meta = self.metas.get(t)
                if meta is None or t not in md.scores:
                    continue
                try:
                    tab[t] = md.adapter.to_trial_scores(raw, meta, model=name, max_repeat=max_repeat).scores
                except AdapterError:
                    pass
            out[name] = tab
        return out

    def per_flash_models(self) -> list[str]:
        return [n for n, m in self.models.items() if m.adapter.per_flash]

    @property
    def max_repeats(self) -> int:
        return max((m.n_repeats for m in self.metas.values()), default=0)


def adapt_model(name: str, adapter: Adapter, raw: Mapping[str, Any], metas: Mapping[str, TrialMeta],
                issues: list[Issue]) -> dict[str, TrialScores]:
    scores: dict[str, TrialScores] = {}
    errors: dict[str, list[str]] = {}
    nan_ids, const_ids, order_ids, top_target = [], [], [], []
    for tid, r in raw.items():
        meta = metas.get(tid)
        if meta is None:
            continue
        try:
            ts = adapter.to_trial_scores(r, meta, model=name)
        except AdapterError as e:
            msg = str(e).split(":", 1)[-1].strip() if f"trial {tid}:" in str(e) else str(e)
            errors.setdefault(msg, []).append(tid)
            continue
        if not np.all(np.isfinite(ts.scores)):
            nan_ids.append(tid)
        elif ts.scores.std() < CONSTANT_TOL * max(1.0, float(np.abs(ts.scores).max())):
            const_ids.append(tid)
        if ts.extras.get("option_order_mismatch"):
            order_ids.append(tid)
        if ts.extras.get("top_is_target_category"):
            top_target.append(tid)
        scores[tid] = ts
    for msg, tids in errors.items():
        issues.append(Issue("error", "adapter_error", name, f"{len(tids)} trial(s) could not be converted: {msg}", tids))
    if nan_ids:
        issues.append(Issue("warning", "nan_scores", name, f"{len(nan_ids)} trial(s) have NaN/inf scores; "
                            "the model is skipped on those trials", nan_ids))
    if const_ids:
        issues.append(Issue("warning", "constant_scores", name, f"{len(const_ids)} trial(s) have scores that are "
                            "constant across options (no information)", const_ids))
    if order_ids:
        issues.append(Issue("warning", "option_order_mismatch", name, f"{len(order_ids)} trial(s) list options in a "
                            "different order from the stimulus log; they were re-ordered by option id. Check that "
                            "list outputs without ids use the stimulus log order.", order_ids))
    if top_target and scores:
        issues.append(Issue("info", "top_class_is_prompt_category", name,
                            f"top decoded class equals the prompt's target category on {len(top_target)}/{len(scores)} "
                            "trials. If the model could be decoding the prompt rather than the choice, test with a "
                            "control where the prompt is hidden or mismatched.", top_target))
    return scores


def check_alignment(name: str, raw: Mapping[str, Any], metas: Mapping[str, TrialMeta], issues: list[Issue]) -> None:
    extra = sorted(set(raw) - set(metas))
    missing = sorted(t for t, m in metas.items() if t not in raw)
    if extra:
        issues.append(Issue("warning", "unknown_trial_ids", name, f"{len(extra)} trial id(s) in the model output are "
                            "not in the stimulus log (ignored). Check the trial id format (e.g. 's1_t003' vs '3').", extra))
    if missing:
        level = "error" if len(missing) == len(metas) else "warning"
        issues.append(Issue(level, "missing_trials", name, f"{len(missing)}/{len(metas)} stimulus-log trial(s) have no "
                            "output from this model", missing))


def load_dataset(cfg: Config, sources: Mapping[str, Any] | None = None,
                 metas: Mapping[str, TrialMeta] | None = None) -> Dataset:
    """Load the stimulus log and every enabled model, adapt to scores, collect issues.

    ``sources`` overrides a model's ``source`` with a path or in-memory object.
    """
    issues: list[Issue] = []
    if metas is None:
        if cfg.markers is None:
            raise ValueError("config needs stimulus_log.markers (the web app's marker log)")
        metas = load_stimulus_log(cfg.resolve(cfg.markers), cfg.resolve(cfg.trials) if cfg.trials else None)
    metas = dict(metas)
    models: dict[str, ModelData] = {}
    for mc in cfg.enabled_models:
        src = (sources or {}).get(mc.name, mc.source)
        if src is None:
            issues.append(Issue("error", "no_source", mc.name, "no 'source' set for this model"))
            continue
        src = cfg.resolve(src) if isinstance(src, str) else src
        trial_ids = mc.trial_ids
        if trial_ids is None and mc.trial_order == "stimulus_log":
            trial_ids = list(metas)
        raw = load_any(src, trial_ids=trial_ids, pattern=mc.file_pattern)
        check_alignment(mc.name, raw, metas, issues)
        adapter = mc.build_adapter()
        scores = adapt_model(mc.name, adapter, raw, metas, issues)
        if getattr(adapter, "reason", None):
            issues.append(Issue("info", "auto_adapter", mc.name, f"auto-detected adapter: {adapter.reason}. "
                                "Write this into the config to make it explicit."))
        models[mc.name] = ModelData(mc, adapter, raw, scores)
    return Dataset(metas=metas, models=models, issues=issues)


def fuse_dataset(ds: Dataset, fuser: Fuser, max_repeat: int | None = None) -> dict[str, FusedPrediction]:
    table = ds.score_table(max_repeat)
    out = {}
    for tid, meta in ds.metas.items():
        out[tid] = fuser.fuse(meta, {m: table[m].get(tid) for m in table})
    return out

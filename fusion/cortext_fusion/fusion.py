"""Per-trial normalization, weighted fusion, calibration and sequential fusion."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax

from .schema import FusedPrediction, TrialMeta

CONSTANT_TOL = 1e-9


# --------------------------------------------------------------------------- normalization

def normalize(scores: Sequence[float] | np.ndarray, method: str = "zscore") -> np.ndarray:
    """Normalize one model's scores across the options of one trial.

    zscore      (s - mean) / std; all zeros if the scores are constant
    logsoftmax  s - logsumexp(s)   (log-probabilities; keeps the model's own scale)
    softmax     probabilities
    center      s - mean          (keeps scale; used for sequential evidence)
    none        unchanged
    """
    s = np.asarray(scores, dtype=float)
    if method == "zscore":
        sd = s.std()
        return np.zeros_like(s) if sd < CONSTANT_TOL else (s - s.mean()) / sd
    if method == "logsoftmax":
        return s - logsumexp(s)
    if method == "softmax":
        return softmax(s)
    if method == "center":
        return s - s.mean()
    if method == "none":
        return s.copy()
    raise ValueError(f"unknown normalization {method!r}")


# --------------------------------------------------------------------------- fuser

@dataclass
class ModelWeight:
    name: str
    weight: float
    normalize: str = "zscore"


@dataclass
class Fuser:
    """Weighted sum of per-trial normalized scores, softmax for confidence."""

    models: list[ModelWeight]
    temperature: float = 1.0
    unsure_threshold: float = 0.4
    source: str = "config"   # "config" or the calibration file it came from

    @classmethod
    def from_config(cls, cfg, calibration: "CalibrationResult | str | Path | None" = None) -> "Fuser":
        models = [ModelWeight(m.name, float(m.weight), m.normalize or cfg.fusion.normalize) for m in cfg.enabled_models]
        f = cls(models=models, temperature=cfg.fusion.temperature, unsure_threshold=cfg.fusion.unsure_threshold)
        cal = calibration
        if cal is None and cfg.fusion.calibration_file:
            p = cfg.resolve(cfg.fusion.calibration_file)
            if p is not None and p.exists():
                cal = p
        if cal is not None:
            if not isinstance(cal, CalibrationResult):
                cal = CalibrationResult.load(cal)
            f.apply_calibration(cal)
        return f

    def apply_calibration(self, cal: "CalibrationResult") -> None:
        for m in self.models:
            m.weight = float(cal.weights.get(m.name, 0.0))
            if m.name in cal.normalize:
                m.normalize = cal.normalize[m.name]
        self.temperature = 1.0  # temperature is absorbed into the fitted weights
        self.source = cal.path or "calibration"

    @property
    def weights(self) -> dict[str, float]:
        return {m.name: m.weight for m in self.models}

    def fuse(self, meta: TrialMeta | Sequence[str], scores: Mapping[str, Sequence[float] | None],
             categories: Sequence[str] | None = None) -> FusedPrediction:
        if isinstance(meta, TrialMeta):
            option_ids, cats, trial_id = list(meta.option_ids), list(meta.option_categories), meta.trial_id
        else:
            option_ids, cats, trial_id = list(meta), list(categories or [""] * len(meta)), "?"
        n = len(option_ids)
        fused = np.zeros(n)
        contributions: dict[str, float] = {}
        per_model: dict[str, list[float]] = {}
        used, warnings = [], []
        normed: dict[str, np.ndarray] = {}
        for m in self.models:
            s = scores.get(m.name)
            if s is None:
                if m.weight != 0:
                    warnings.append(f"{m.name}: no output for this trial (skipped)")
                continue
            s = np.asarray(s, dtype=float)
            if s.shape != (n,):
                warnings.append(f"{m.name}: {s.shape[0] if s.ndim else 0} scores for {n} options (skipped)")
                continue
            if not np.all(np.isfinite(s)):
                warnings.append(f"{m.name}: NaN/inf scores (skipped)")
                continue
            if s.std() < CONSTANT_TOL:
                warnings.append(f"{m.name}: scores are constant across options (no information)")
            z = normalize(s, m.normalize)
            normed[m.name] = z
            per_model[m.name] = z.tolist()
            if m.weight != 0:
                fused += m.weight * z
                used.append(m.name)
        T = max(self.temperature, 1e-6)
        probs = softmax(fused / T)
        if not used:
            warnings.append("no model with non-zero weight produced usable scores")
        order = np.argsort(-probs)
        top = int(order[0])
        margin = float(probs[order[0]] - probs[order[1]]) if n > 1 else 1.0
        conf = float(probs[top])
        if np.allclose(probs, probs[0]):
            pred_idx = None
        else:
            pred_idx = top
        for name, z in normed.items():
            w = self.weights.get(name, 0.0)
            contributions[name] = float(w * (z[top] - z.mean()) / T)
        unsure = pred_idx is None or conf < self.unsure_threshold
        return FusedPrediction(
            trial_id=trial_id,
            predicted_option=option_ids[top] if pred_idx is not None else None,
            predicted_index=pred_idx,
            predicted_category=(cats[top] or None) if pred_idx is not None else None,
            confidence=conf, margin=margin, unsure=unsure,
            probabilities=probs.tolist(), fused_scores=fused.tolist(), option_ids=option_ids,
            contributions=contributions, per_model_scores=per_model, models_used=used, warnings=warnings,
        )


# --------------------------------------------------------------------------- calibration

@dataclass
class CalibrationResult:
    """Weights learned by conditional logistic regression on labeled validation trials.

    ``weights`` already include the temperature (scale). ``temperature`` and
    ``relative_weights`` are reported for readability: weight = relative / temperature.
    """

    weights: dict[str, float]
    normalize: dict[str, str]
    n_trials: int
    nll_per_trial: float
    selected_models: list[str]
    selection: dict[str, Any] = field(default_factory=dict)
    path: str | None = None

    @property
    def temperature(self) -> float:
        s = sum(abs(w) for w in self.weights.values())
        return float(1.0 / s) if s > 0 else float("inf")

    @property
    def relative_weights(self) -> dict[str, float]:
        s = sum(abs(w) for w in self.weights.values())
        return {k: (v / s if s else 0.0) for k, v in self.weights.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": {k: float(v) for k, v in self.weights.items()},
            "normalize": self.normalize,
            "temperature": self.temperature,
            "relative_weights": self.relative_weights,
            "n_trials": self.n_trials,
            "nll_per_trial": self.nll_per_trial,
            "selected_models": self.selected_models,
            "selection": self.selection,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(_plain(self.to_dict()), sort_keys=False))

    @classmethod
    def load(cls, path: str | Path) -> "CalibrationResult":
        d = yaml.safe_load(Path(path).read_text())
        return cls(weights={k: float(v) for k, v in d["weights"].items()}, normalize=d.get("normalize", {}),
                   n_trials=int(d.get("n_trials", 0)), nll_per_trial=float(d.get("nll_per_trial", float("nan"))),
                   selected_models=list(d.get("selected_models", d["weights"].keys())),
                   selection=d.get("selection", {}), path=str(path))


def _plain(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def fit_conditional_logit(X: np.ndarray, y: np.ndarray, l2: float = 1e-2,
                          nonnegative: bool = True) -> tuple[np.ndarray, float]:
    """Fit beta for P(option j | trial) = softmax_j(X[t, j, :] @ beta).

    X: [trials, options, features]; y: index of the correct option per trial.
    This is logistic regression where the choice set is the trial's options
    (McFadden's conditional logit); it learns weights and temperature jointly.
    Weights are constrained to be >= 0 by default: every adapter promises
    "higher = more likely", so a negative weight would only hide a sign error
    (which the evaluation flags) or exploit a chance anti-correlation.
    Returns (beta, mean negative log-likelihood per trial).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    T, J, K = X.shape
    idx = np.arange(T)

    def f(beta):
        u = X @ beta                     # [T, J]
        lse = logsumexp(u, axis=1)
        nll = np.sum(lse - u[idx, y]) + l2 * beta @ beta
        p = np.exp(u - lse[:, None])
        grad = np.einsum("tj,tjk->k", p, X) - X[idx, y].sum(0) + 2 * l2 * beta
        return nll, grad

    bounds = [(0.0, None)] * K if nonnegative else None
    res = minimize(f, np.full(K, 0.1 if nonnegative else 0.0), jac=True, method="L-BFGS-B", bounds=bounds)
    beta = res.x
    u = X @ beta
    nll = float(np.mean(logsumexp(u, axis=1) - u[idx, y]))
    return beta, nll


def conditional_logit_nll(X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
    u = np.asarray(X) @ beta
    return logsumexp(u, axis=1) - u[np.arange(len(y)), np.asarray(y, dtype=int)]


def build_feature_tensor(
    metas: Sequence[TrialMeta], scores: Mapping[str, Mapping[str, Sequence[float]]],
    models: Sequence[str], norm: Mapping[str, str],
) -> tuple[np.ndarray, list[str]]:
    """X[t, j, k] = normalized score of model k for option j of trial t (0 if missing)."""
    J = max(m.n_options for m in metas) if metas else 0
    X = np.zeros((len(metas), J, len(models)))
    kept = []
    for t, meta in enumerate(metas):
        kept.append(meta.trial_id)
        for k, name in enumerate(models):
            s = scores.get(name, {}).get(meta.trial_id)
            if s is None:
                continue
            s = np.asarray(s, dtype=float)
            if s.shape[0] != meta.n_options or not np.all(np.isfinite(s)):
                continue
            X[t, : meta.n_options, k] = normalize(s, norm.get(name, "zscore"))
    return X, kept


def _grouped_cv_nll(X: np.ndarray, y: np.ndarray, groups: np.ndarray, cols: list[int], l2: float) -> float | None:
    ug = np.unique(groups)
    if len(ug) < 2:
        return None
    losses = np.zeros(len(y))
    for g in ug:
        te = groups == g
        tr = ~te
        if tr.sum() == 0:
            return None
        beta, _ = fit_conditional_logit(X[tr][:, :, cols], y[tr], l2)
        losses[te] = conditional_logit_nll(X[te][:, :, cols], y[te], beta)
    return float(losses.mean())


def calibrate(
    metas: Sequence[TrialMeta],
    scores: Mapping[str, Mapping[str, Sequence[float]]],
    models: Sequence[str],
    labels: Mapping[str, int],
    norm: Mapping[str, str] | None = None,
    baseline_models: Sequence[str] | None = None,
    groups: Mapping[str, str] | None = None,
    min_gain: float = 0.01,
    l2: float = 1e-2,
) -> CalibrationResult:
    """Learn fusion weights on labeled validation trials.

    ``baseline_models`` (default: the first model) always get a weight. Every
    other model is added only if it lowers the *held-out* (leave-one-group-out)
    log-loss by at least ``min_gain`` nats per trial. Without at least two
    groups the extra models stay at weight 0, because their value can't be
    shown on held-out data.
    """
    norm = dict(norm or {})
    models = list(models)
    baseline = list(baseline_models or models[:1])
    metas = [m for m in metas if m.trial_id in labels]
    if len(metas) < 5:
        raise ValueError(f"calibration needs labeled validation trials; got {len(metas)}")
    X, ids = build_feature_tensor(metas, scores, models, norm)
    y = np.array([labels[t] for t in ids])
    g = np.array([str((groups or {}).get(t, "all")) for t in ids])

    base_cols = [models.index(b) for b in baseline]
    selected = list(baseline)
    selection: dict[str, Any] = {"baseline": baseline, "candidates": {}}
    base_cv = _grouped_cv_nll(X, y, g, base_cols, l2)
    selection["baseline_cv_nll"] = base_cv
    cur_cols = list(base_cols)
    cur_cv = base_cv
    for k, name in enumerate(models):
        if name in baseline:
            continue
        if cur_cv is None:
            selection["candidates"][name] = {"kept": False, "reason": "fewer than 2 sessions/participants; cannot test on held-out data"}
            continue
        cv = _grouped_cv_nll(X, y, g, cur_cols + [k], l2)
        gain = cur_cv - cv
        keep = gain >= min_gain
        selection["candidates"][name] = {"kept": bool(keep), "cv_nll_with": cv, "cv_nll_without": cur_cv,
                                         "gain_nats_per_trial": gain}
        if keep:
            cur_cols.append(k)
            cur_cv = cv
            selected.append(name)
    beta, nll = fit_conditional_logit(X[:, :, cur_cols], y, l2)
    # If the data say "no information" (all weights 0), keep a tiny weight on the baseline so the
    # prediction is still its best guess, with ~uniform confidence (i.e. "unsure"), not "no answer".
    floor = 1e-3
    for i, c in enumerate(cur_cols):
        if c in base_cols and beta[i] < floor:
            beta[i] = floor
            selection.setdefault("floored", []).append(models[c])
    weights = {name: 0.0 for name in models}
    for c, b in zip(cur_cols, beta):
        weights[models[c]] = float(b)
    selection["final_cv_nll"] = cur_cv
    selection["chance_nll"] = float(np.log(X.shape[1]))
    return CalibrationResult(weights=weights, normalize={m: norm.get(m, "zscore") for m in models},
                             n_trials=len(y), nll_per_trial=nll, selected_models=selected, selection=selection)


# --------------------------------------------------------------------------- sequential

@dataclass
class SequentialState:
    trial_id: str
    option_ids: list[str]
    option_categories: list[str]
    evidence: np.ndarray
    counts: np.ndarray
    repeats_seen: int = 0
    stopped: bool = False
    stop_reason: str | None = None


class SequentialFuser:
    """Accumulate per-flash evidence as repeats arrive; stop early when confident.

    Each flash adds ``scale * value`` to its option's running log-score, where
    ``value`` is a log-odds (``input='prob'`` converts P(target) with the prior
    1/n_options removed) or a raw decision score (``input='score'``). Per-trial
    evidence from other models (e.g. the category model) can be added once with
    :meth:`add_trial_evidence`. Confidence = softmax of the running log-scores.
    """

    def __init__(self, scale: float = 1.0, input: str = "prob", stop_confidence: float = 0.9,
                 min_repeats: int = 3, max_repeats: int | None = None, unsure_threshold: float = 0.4,
                 eps: float = 1e-4):
        if input not in ("prob", "score"):
            raise ValueError("input must be 'prob' or 'score'")
        self.scale, self.input = float(scale), input
        self.stop_confidence, self.min_repeats, self.max_repeats = stop_confidence, min_repeats, max_repeats
        self.unsure_threshold, self.eps = unsure_threshold, eps
        self.states: dict[str, SequentialState] = {}

    def start(self, trial_id: str, option_ids: Sequence[str], option_categories: Sequence[str] | None = None) -> SequentialState:
        n = len(option_ids)
        st = SequentialState(str(trial_id), [str(o) for o in option_ids],
                             list(option_categories or [""] * n), np.zeros(n), np.zeros(n, dtype=int))
        self.states[st.trial_id] = st
        return st

    def _value(self, v: float, n: int) -> float:
        if self.input == "prob":
            p = float(np.clip(v, self.eps, 1 - self.eps))
            prior = 1.0 / n
            return (np.log(p) - np.log1p(-p)) - (np.log(prior) - np.log1p(-prior))
        return float(v)

    def update(self, trial_id: str, option_id: str, value: float, repeat_index: int | None = None) -> FusedPrediction:
        st = self.states.get(str(trial_id))
        if st is None:
            raise KeyError(f"trial {trial_id} not started; call start() first")
        j = st.option_ids.index(str(option_id))
        if not st.stopped:
            st.evidence[j] += self.scale * self._value(value, len(st.option_ids))
            st.counts[j] += 1
            st.repeats_seen = int(st.counts.min()) if repeat_index is None else max(st.repeats_seen, int(repeat_index) + 1)
        return self.current(trial_id)

    def add_trial_evidence(self, trial_id: str, scores: Sequence[float], weight: float, method: str = "zscore") -> None:
        st = self.states[str(trial_id)]
        st.evidence += weight * normalize(scores, method)

    def current(self, trial_id: str) -> FusedPrediction:
        st = self.states[str(trial_id)]
        ev = st.evidence - st.evidence.mean()
        probs = softmax(ev)
        order = np.argsort(-probs)
        top = int(order[0])
        conf = float(probs[top])
        complete = int(st.counts.min()) if len(st.counts) else 0
        if not st.stopped:
            if complete >= self.min_repeats and conf >= self.stop_confidence:
                st.stopped, st.stop_reason = True, "confident"
            elif self.max_repeats is not None and complete >= self.max_repeats:
                st.stopped, st.stop_reason = True, "max_repeats"
        any_ev = bool(np.ptp(ev) > 0)
        return FusedPrediction(
            trial_id=st.trial_id,
            predicted_option=st.option_ids[top] if any_ev else None,
            predicted_index=top if any_ev else None,
            predicted_category=(st.option_categories[top] or None) if any_ev else None,
            confidence=conf, margin=float(probs[order[0]] - probs[order[1]]) if len(probs) > 1 else 1.0,
            unsure=(not any_ev) or conf < self.unsure_threshold,
            probabilities=probs.tolist(), fused_scores=ev.tolist(), option_ids=st.option_ids,
            contributions={}, per_model_scores={}, models_used=["sequential"],
            stopped_early=st.stopped and st.stop_reason == "confident", repeats_used=complete,
        )

    def replay(self, meta: TrialMeta, flash_values: Sequence[float]) -> FusedPrediction:
        """Feed a whole trial's flashes in marker order; return the prediction at the stopping point."""
        self.start(meta.trial_id, meta.option_ids, meta.option_categories)
        pred = self.current(meta.trial_id)
        for f, v in zip(meta.flashes, flash_values):
            pred = self.update(meta.trial_id, f.option_id, v)
            if self.states[meta.trial_id].stopped:
                break
        return pred

    @staticmethod
    def fit_scale(metas: Sequence[TrialMeta], flash_values: Mapping[str, Sequence[float]], labels: Mapping[str, int],
                  input: str = "prob", eps: float = 1e-4) -> float:
        """Fit the per-flash evidence scale on validation trials (conditional logit, one feature)."""
        helper = SequentialFuser(scale=1.0, input=input, eps=eps)
        rows, ys = [], []
        for m in metas:
            if m.trial_id not in labels or m.trial_id not in flash_values:
                continue
            ev = np.zeros(m.n_options)
            for f, v in zip(m.flashes, flash_values[m.trial_id]):
                ev[m.option_index(f.option_id)] += helper._value(v, m.n_options)
            rows.append(ev - ev.mean())
            ys.append(labels[m.trial_id])
        if len(rows) < 5:
            return 1.0
        X = np.array(rows)[:, :, None]
        beta, _ = fit_conditional_logit(X, np.array(ys), l2=1e-3)
        return float(max(beta[0], 1e-3))

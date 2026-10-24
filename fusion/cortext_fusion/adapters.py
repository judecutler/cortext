"""Adapters: turn any model's raw per-trial output into the standard format.

Every adapter implements ``to_scores(raw_output, trial_meta) -> list[float]``:
one real-valued score per option, in ``trial_meta.option_ids`` order, higher =
more likely the option the participant chose.

Ready adapters (names used in the YAML config):

=====================  =====================================================
``per_option``          (a) one score per option, list or dict
``per_flash``           (b) one score per flash, averaged per option
``probabilities``       (c) one probability per option -> logit (or log)
``epoch_target_prob``   (d) p(target) per flash -> mean log-odds per option
``class_probabilities`` (e) probabilities over many categories, looked up by
                        each option's category label
``embedding``           (f) one vector per trial, cosine to each option's vector
``auto``                detect the shape on the first trial, then lock it in
=====================  =====================================================

(g) Any adapter accepts ``lower_is_better: true`` (distances, losses), which
negates the final scores. ``distance`` is an alias for ``per_option`` with that
flag set.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .schema import TrialMeta, TrialScores

EPS_DEFAULT = 1e-4

OPTION_ID_FIELDS = ("option_id", "option", "options", "option_ids", "opt", "stimulus", "flash_option")
SCORE_FIELDS = ("score", "scores", "value", "values", "logit", "logits", "decision", "output", "y")
PROB_FIELDS = ("prob", "probs", "probability", "probabilities", "p", "p_target", "proba")
REPEAT_FIELDS = ("repeat_index", "repeat", "repetition", "rep")
EMB_FIELDS = ("embedding", "embeddings", "vector", "emb", "z", "latent")


class AdapterError(ValueError):
    """Raised with a message that tells the teammate what to change."""

    def __init__(self, msg: str, trial_id: str | None = None, adapter: str | None = None):
        prefix = []
        if adapter:
            prefix.append(f"[{adapter}]")
        if trial_id is not None:
            prefix.append(f"trial {trial_id}:")
        self.has_context = trial_id is not None
        super().__init__(" ".join(prefix + [msg]))


# --------------------------------------------------------------------------- helpers

def _is_numeric_seq(x: Any) -> bool:
    if isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.number) or x.dtype == bool
    if isinstance(x, (list, tuple)) and x:
        try:
            np.asarray(x, dtype=float)
            return True
        except (TypeError, ValueError):
            return False
    return False


def _as_float_array(x: Any, what: str = "values") -> np.ndarray:
    try:
        return np.asarray(x, dtype=float)
    except (TypeError, ValueError) as e:
        raise AdapterError(f"could not convert {what} to numbers: {e}") from e


def _find_field(d: dict, names: tuple[str, ...]) -> str | None:
    lower = {str(k).lower(): k for k in d}
    for n in names:
        if n in lower:
            return lower[n]
    return None


def _strip_prefix(keys, prefix: str | None) -> dict[str, Any]:
    out = {}
    for k in keys:
        s = str(k)
        if prefix and s.startswith(prefix):
            s = s[len(prefix):]
        out[s] = k
    return out


def safe_logit(p: np.ndarray, eps: float = EPS_DEFAULT) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p) - np.log1p(-p)


def safe_log(p: np.ndarray, eps: float = EPS_DEFAULT) -> np.ndarray:
    return np.log(np.clip(np.asarray(p, dtype=float), eps, 1.0))


def _check_prob_range(p: np.ndarray, tol: float = 1e-6) -> None:
    finite = p[np.isfinite(p)]
    if finite.size and (finite.min() < -tol or finite.max() > 1 + tol):
        raise AdapterError(
            f"expected probabilities in [0, 1] but got range [{finite.min():.3g}, {finite.max():.3g}]. "
            "If these are logits/decision values, use adapter 'per_option' or 'per_flash' instead."
        )


def _load_mapping_file(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    if p.suffix == ".json":
        data = json.loads(p.read_text())
        return {str(k): np.asarray(v, dtype=float) for k, v in data.items()}
    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=False)
        if "names" in z and "vectors" in z:
            return {str(n): z["vectors"][i] for i, n in enumerate(z["names"].tolist())}
        return {k: z[k] for k in z.keys()}
    if p.suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(p, index_col=0)
        return {str(i): df.loc[i].to_numpy(dtype=float) for i in df.index}
    raise AdapterError(f"unsupported embedding file {p} (use .json {{name: vector}}, .npz with names/vectors, or .csv)")


# --------------------------------------------------------------------------- base

class Adapter:
    """Base class. Subclasses implement :meth:`_scores`."""

    name = "base"
    per_flash = False  # True if the adapter can truncate to the first k repeats

    def __init__(self, lower_is_better: bool = False, field: str | None = None, **params: Any):
        self.lower_is_better = bool(lower_is_better)
        self.field = field
        self.params = params
        unknown = set(params) - set(self.allowed_params())
        if unknown:
            raise AdapterError(f"unknown option(s) {sorted(unknown)}; allowed: {sorted(self.allowed_params())}",
                               adapter=self.name)

    @classmethod
    def allowed_params(cls) -> set[str]:
        return set()

    # public API -------------------------------------------------------------
    def to_scores(self, raw_output: Any, trial_meta: TrialMeta | dict, max_repeat: int | None = None) -> list[float]:
        meta = trial_meta if isinstance(trial_meta, TrialMeta) else TrialMeta.from_dict(trial_meta)
        return self.to_trial_scores(raw_output, meta, max_repeat=max_repeat).scores.tolist()

    def to_trial_scores(self, raw_output: Any, meta: TrialMeta, model: str = "model",
                        max_repeat: int | None = None) -> TrialScores:
        try:
            raw = self._select(raw_output)
            scores, extras = self._scores(raw, meta, max_repeat)
        except AdapterError as e:
            if getattr(e, "has_context", False):
                raise
            raise AdapterError(str(e), trial_id=meta.trial_id, adapter=self.name) from None
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if scores.shape[0] != meta.n_options:
            raise AdapterError(f"produced {scores.shape[0]} scores but the trial has {meta.n_options} options "
                               f"{meta.option_ids}", trial_id=meta.trial_id, adapter=self.name)
        if self.lower_is_better:
            scores = -scores
        return TrialScores(trial_id=meta.trial_id, model=model, scores=scores, option_ids=list(meta.option_ids),
                           n_repeats_used=extras.pop("n_repeats_used", None), extras=extras)

    # internals --------------------------------------------------------------
    def _select(self, raw: Any) -> Any:
        if self.field is not None:
            if not isinstance(raw, dict) or self.field not in raw:
                keys = list(raw.keys()) if isinstance(raw, dict) else type(raw).__name__
                raise AdapterError(f"field {self.field!r} not found in output (got {keys})")
            return raw[self.field]
        return raw

    def _scores(self, raw: Any, meta: TrialMeta, max_repeat: int | None) -> tuple[np.ndarray, dict]:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {"adapter": self.name, "lower_is_better": self.lower_is_better, "field": self.field, **self.params}


# --------------------------------------------------------------------------- (a) per-option

def _per_option_values(raw: Any, meta: TrialMeta, key_by: str = "auto", prefix: str | None = None,
                       value_fields: tuple[str, ...] = SCORE_FIELDS + PROB_FIELDS) -> tuple[np.ndarray, dict]:
    """Shared parser for (a) and (c): returns values in meta option order."""
    extras: dict[str, Any] = {}
    if isinstance(raw, dict):
        oid_f = _find_field(raw, OPTION_ID_FIELDS)
        val_f = _find_field(raw, value_fields)
        # explicit option ids + values (long format or a record)
        if oid_f is not None and val_f is not None:
            ids = raw[oid_f]
            vals = raw[val_f]
            ids = [str(i) for i in (ids if isinstance(ids, (list, tuple, np.ndarray)) else [ids])]
            vals = _as_float_array(vals).reshape(-1)
            if len(ids) != len(vals):
                raise AdapterError(f"{len(ids)} option ids but {len(vals)} values")
            if len(set(ids)) != len(ids):
                raise AdapterError(f"option ids repeat ({ids}); this looks like per-flash output, use adapter 'per_flash'")
            if set(ids) != set(meta.option_ids):
                raise AdapterError(f"output option ids {ids} do not match stimulus log options {meta.option_ids}")
            if ids != list(meta.option_ids):
                extras["option_order_mismatch"] = ids
            m = dict(zip(ids, vals))
            return np.array([m[o] for o in meta.option_ids]), extras
        if val_f is not None and oid_f is None and _is_numeric_seq(raw[val_f]):
            return _per_option_values(raw[val_f], meta, key_by, prefix, value_fields)
        # mapping keyed by option id or category
        keys = _strip_prefix(raw.keys(), prefix)
        if key_by in ("auto", "option") and set(meta.option_ids) <= set(keys):
            return np.array([float(raw[keys[o]]) for o in meta.option_ids]), extras
        lower_keys = {k.lower(): v for k, v in keys.items()}
        cats = [c.lower() for c in meta.option_categories]
        if key_by in ("auto", "category") and all(cats) and set(cats) <= set(lower_keys):
            return np.array([float(raw[lower_keys[c]]) for c in cats]), extras
        raise AdapterError(
            f"dict output has keys {list(raw.keys())[:12]}; expected the option ids {meta.option_ids}, the option "
            f"categories {meta.option_categories}, or fields like 'option_id' + 'score'. Use 'field:' or 'key_prefix:' "
            "in the config to point at the right part."
        )
    if _is_numeric_seq(raw):
        arr = _as_float_array(raw)
        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        if arr.ndim != 1:
            raise AdapterError(f"expected one value per option but got an array of shape {arr.shape}; "
                               "for [repeats, options] use adapter 'per_flash'")
        if arr.shape[0] != meta.n_options:
            hint = ""
            if meta.flashes and arr.shape[0] == len(meta.flashes):
                hint = " This length matches the number of flashes: use adapter 'per_flash' or 'epoch_target_prob'."
            raise AdapterError(f"expected {meta.n_options} values (one per option) but got {arr.shape[0]}.{hint}")
        return arr, extras
    raise AdapterError(f"cannot read per-option values from {type(raw).__name__}")


class PerOptionAdapter(Adapter):
    """(a) One score per option. List in option order, or a dict keyed by option id/category."""

    name = "per_option"

    @classmethod
    def allowed_params(cls):
        return {"key_by", "key_prefix"}

    def _scores(self, raw, meta, max_repeat):
        return _per_option_values(raw, meta, self.params.get("key_by", "auto"), self.params.get("key_prefix"))


class ProbabilitiesAdapter(Adapter):
    """(c) One probability per option, converted to logits (or log-probs).

    ``transform: logit`` (default) treats each value as an independent
    P(option is target). ``transform: log`` suits a softmax over options.
    ``eps`` clips 0 and 1 so the conversion never produces infinities.
    """

    name = "probabilities"

    @classmethod
    def allowed_params(cls):
        return {"key_by", "key_prefix", "transform", "eps"}

    def _scores(self, raw, meta, max_repeat):
        p, extras = _per_option_values(raw, meta, self.params.get("key_by", "auto"), self.params.get("key_prefix"))
        _check_prob_range(p)
        eps = float(self.params.get("eps", EPS_DEFAULT))
        n_clipped = int(np.sum((p < eps) | (p > 1 - eps)))
        if n_clipped:
            extras["n_clipped"] = n_clipped
        tr = self.params.get("transform", "logit")
        if tr == "logit":
            return safe_logit(p, eps), extras
        if tr == "log":
            return safe_log(p, eps), extras
        raise AdapterError(f"transform must be 'logit' or 'log', got {tr!r}")


# --------------------------------------------------------------------------- (b)/(d) per-flash

def _per_flash_table(raw: Any, meta: TrialMeta, value_fields: tuple[str, ...],
                     target_column: int | None, grid_order: list[str] | None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (option_id per flash, repeat index per flash, value per flash)."""
    if isinstance(raw, dict):
        oid_f = _find_field(raw, OPTION_ID_FIELDS)
        val_f = _find_field(raw, value_fields)
        rep_f = _find_field(raw, REPEAT_FIELDS)
        if val_f is None:
            raise AdapterError(f"per-flash dict needs a value field (one of {value_fields}); got {list(raw.keys())}")
        vals = _as_float_array(raw[val_f])
        if vals.ndim == 2 and target_column is not None:
            vals = vals[:, target_column]
        vals = vals.reshape(-1) if vals.ndim == 1 or (vals.ndim == 2 and 1 in vals.shape) else vals
        if oid_f is None:
            return _per_flash_table(vals, meta, value_fields, target_column, grid_order)
        ids = [str(i) for i in raw[oid_f]]
        if len(ids) != len(vals):
            raise AdapterError(f"{len(ids)} option ids but {len(vals)} flash values")
        reps = np.asarray(raw[rep_f], dtype=int) if rep_f else _infer_repeats(ids)
        unknown = set(ids) - set(meta.option_ids)
        if unknown:
            raise AdapterError(f"flash option ids {sorted(unknown)} are not options of this trial {meta.option_ids}")
        return ids, reps, vals
    arr = _as_float_array(raw)
    if arr.ndim == 2 and target_column is not None and arr.shape[1] == 2 and meta.flashes and arr.shape[0] == len(meta.flashes):
        arr = arr[:, target_column]
    if arr.ndim == 2:
        order = grid_order or meta.option_ids
        if arr.shape[1] != len(order):
            raise AdapterError(f"grid output has shape {arr.shape}; expected [repeats, {len(order)} options]")
        if set(order) != set(meta.option_ids):
            raise AdapterError(f"grid_order {order} does not match trial options {meta.option_ids}")
        reps_n = arr.shape[0]
        ids = [o for _ in range(reps_n) for o in order]
        reps = np.repeat(np.arange(reps_n), len(order))
        return ids, reps, arr.reshape(-1)
    if arr.ndim == 1:
        if not meta.flashes:
            raise AdapterError("a 1-D per-flash array needs the flash order from the stimulus marker log, "
                               "but this trial has no flash markers")
        if arr.shape[0] != len(meta.flashes):
            hint = " (that is one per option: use 'per_option' or 'probabilities')" if arr.shape[0] == meta.n_options else ""
            raise AdapterError(f"got {arr.shape[0]} values but the marker log has {len(meta.flashes)} flashes{hint}")
        return [f.option_id for f in meta.flashes], np.array([f.repeat_index for f in meta.flashes]), arr
    raise AdapterError(f"cannot read per-flash values from an array of shape {arr.shape}")


def _infer_repeats(ids: list[str]) -> np.ndarray:
    counts: dict[str, int] = {}
    out = []
    for i in ids:
        out.append(counts.get(i, 0))
        counts[i] = out[-1] + 1
    return np.array(out)


AGGREGATORS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda v: float(np.mean(v)),
    "sum": lambda v: float(np.sum(v)),
    "median": lambda v: float(np.median(v)),
}


class PerFlashAdapter(Adapter):
    """(b) One score per flash, aggregated per option (mean by default).

    Accepts a 1-D array aligned with the marker log's flash order, a
    ``[repeats, options]`` grid, or a dict with ``option_id`` + ``score``
    (+ optional ``repeat_index``) lists.
    """

    name = "per_flash"
    per_flash = True
    value_fields = SCORE_FIELDS + PROB_FIELDS

    @classmethod
    def allowed_params(cls):
        return {"aggregate", "grid_order", "missing"}

    def _transform(self, v: np.ndarray) -> np.ndarray:
        return v

    def _scores(self, raw, meta, max_repeat):
        ids, reps, vals = _per_flash_table(raw, meta, self.value_fields, self._target_column(), self.params.get("grid_order"))
        vals = self._transform(vals)
        if max_repeat is not None:
            keep = reps < max_repeat
            ids = [i for i, k in zip(ids, keep) if k]
            reps, vals = reps[keep], vals[keep]
        agg = AGGREGATORS.get(self.params.get("aggregate", "mean"))
        if agg is None:
            raise AdapterError(f"aggregate must be one of {list(AGGREGATORS)}")
        out = np.full(meta.n_options, np.nan)
        counts = np.zeros(meta.n_options, dtype=int)
        ids_arr = np.array(ids)
        for j, o in enumerate(meta.option_ids):
            v = vals[ids_arr == o] if len(ids_arr) else np.array([])
            v = v[np.isfinite(v)]
            counts[j] = v.size
            if v.size:
                out[j] = agg(v)
        if (counts == 0).any():
            missing = [o for o, c in zip(meta.option_ids, counts) if c == 0]
            if self.params.get("missing", "error") == "error":
                raise AdapterError(f"no flash values for option(s) {missing}")
        n_used = int(reps.max()) + 1 if len(reps) else 0
        return out, {"n_repeats_used": n_used, "flashes_per_option": counts.tolist()}

    def _target_column(self) -> int | None:
        return None


class EpochTargetProbAdapter(PerFlashAdapter):
    """(d) P(target) for each flash epoch, aggregated per option as mean log-odds.

    Also accepts sklearn-style ``predict_proba`` output shaped ``[flashes, 2]``
    (column ``target_column``, default 1, is P(target)).
    """

    name = "epoch_target_prob"

    @classmethod
    def allowed_params(cls):
        return {"aggregate", "grid_order", "missing", "eps", "target_column"}

    def _target_column(self):
        return int(self.params.get("target_column", 1))

    def _transform(self, v):
        _check_prob_range(v)
        return safe_logit(v, float(self.params.get("eps", EPS_DEFAULT)))


# --------------------------------------------------------------------------- (e) class probabilities

class ClassProbabilitiesAdapter(Adapter):
    """(e) Probabilities over many categories; each option is scored by its category.

    Options:
      class_names:   list of class labels in output order (needed for arrays)
      class_names_file: path to a .txt/.json list instead
      category_map:  {stimulus category: model class name} when names differ
      key_prefix:    strip this from dict keys / CSV columns (e.g. "p_")
      transform:     log (default) | logit | none
      missing:       error (default) | floor  -- option category not in classes
      check_target:  true -> record P(target category) and whether the model's
                     top class equals the prompt's target category (a confound
                     check: the model may be decoding the prompt, not the choice)
    """

    name = "class_probabilities"

    @classmethod
    def allowed_params(cls):
        return {"class_names", "class_names_file", "category_map", "key_prefix", "transform", "missing",
                "check_target", "eps", "renormalize"}

    def _class_names(self) -> list[str] | None:
        if self.params.get("class_names"):
            return [str(c) for c in self.params["class_names"]]
        f = self.params.get("class_names_file")
        if f:
            p = Path(f)
            if p.suffix == ".json":
                return [str(c) for c in json.loads(p.read_text())]
            return [l.strip() for l in p.read_text().splitlines() if l.strip()]
        return None

    def _scores(self, raw, meta, max_repeat):
        names = self._class_names()
        if isinstance(raw, dict):
            pf = _find_field(raw, PROB_FIELDS)
            nf = _find_field(raw, ("class_names", "classes", "labels"))
            if pf is not None and _is_numeric_seq(raw[pf]):
                if nf is not None:
                    names = [str(c) for c in raw[nf]]
                raw = raw[pf]
            else:
                keys = _strip_prefix(raw.keys(), self.params.get("key_prefix"))
                numeric = {k: raw[orig] for k, orig in keys.items() if _scalar_number(raw[orig])}
                names = list(numeric)
                raw = [float(numeric[k]) for k in names]
        p = _as_float_array(raw).reshape(-1)
        if names is None:
            raise AdapterError(f"got {p.shape[0]} class probabilities but no class names; set 'class_names' "
                               "(or 'class_names_file') in the config in the model's output order")
        if len(names) != p.shape[0]:
            raise AdapterError(f"{p.shape[0]} probabilities but {len(names)} class names")
        if self.params.get("transform", "log") != "none":
            _check_prob_range(p)
        lookup = {n.lower(): i for i, n in enumerate(names)}
        cmap = {str(k).lower(): str(v).lower() for k, v in (self.params.get("category_map") or {}).items()}
        eps = float(self.params.get("eps", EPS_DEFAULT))
        if self.params.get("renormalize", False):
            p = p / max(p.sum(), eps)
        vals = np.empty(meta.n_options)
        missing = []
        for j, cat in enumerate(meta.option_categories):
            key = cmap.get(cat.lower(), cat.lower())
            if key in lookup:
                vals[j] = p[lookup[key]]
            else:
                missing.append(cat)
                vals[j] = np.nan
        if missing:
            if self.params.get("missing", "error") == "error":
                raise AdapterError(f"option categories {missing} are not among the model's classes "
                                   f"{names[:20]}{'...' if len(names) > 20 else ''}; add a 'category_map' "
                                   "or set missing: floor")
            vals = np.where(np.isnan(vals), 0.0, vals)
        tr = self.params.get("transform", "log")
        out = safe_log(vals, eps) if tr == "log" else safe_logit(vals, eps) if tr == "logit" else vals
        extras: dict[str, Any] = {}
        if missing:
            extras["missing_categories"] = missing
        if self.params.get("check_target", False) and meta.target_category:
            tkey = cmap.get(meta.target_category.lower(), meta.target_category.lower())
            top = names[int(np.argmax(p))]
            extras["target_category_prob"] = float(p[lookup[tkey]]) if tkey in lookup else None
            extras["top_class"] = top
            extras["top_is_target_category"] = top.lower() == tkey
        return out, extras


def _scalar_number(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float, np.number)):
        return not (isinstance(v, float) and math.isnan(v))
    return False


# --------------------------------------------------------------------------- (f) embeddings

class EmbeddingAdapter(Adapter):
    """(f) One embedding per trial, scored by cosine similarity to each option's embedding.

    Options:
      embeddings:   path (.json {name: vector}, .npz names/vectors, .csv) or an
                    in-memory dict {name: vector}
      key_by:       category (default) | option  -- how to look up each option's vector
      metric:       cosine (default) | dot | neg_euclidean
    """

    name = "embedding"

    @classmethod
    def allowed_params(cls):
        return {"embeddings", "key_by", "metric"}

    def __init__(self, **kw):
        super().__init__(**kw)
        src = self.params.get("embeddings")
        if src is None:
            raise AdapterError("the 'embedding' adapter needs 'embeddings:' (vectors for each category/option)",
                               adapter=self.name)
        table = src if isinstance(src, dict) else _load_mapping_file(src)
        self._table = {str(k).lower(): np.asarray(v, dtype=float).reshape(-1) for k, v in table.items()}
        dims = {v.shape[0] for v in self._table.values()}
        if len(dims) != 1:
            raise AdapterError(f"option embeddings have mixed dimensions {sorted(dims)}", adapter=self.name)
        self.dim = dims.pop()

    def describe(self):
        d = super().describe()
        d["embeddings"] = f"<{len(self._table)} vectors of dim {self.dim}>" if not isinstance(
            self.params.get("embeddings"), str) else self.params["embeddings"]
        return d

    def _scores(self, raw, meta, max_repeat):
        if isinstance(raw, dict):
            f = _find_field(raw, EMB_FIELDS)
            if f is None:
                raise AdapterError(f"no embedding field found (looked for {EMB_FIELDS}); got {list(raw.keys())}")
            raw = raw[f]
        v = _as_float_array(raw).reshape(-1)
        if v.shape[0] != self.dim:
            raise AdapterError(f"trial embedding has dim {v.shape[0]} but option embeddings have dim {self.dim}")
        keys = meta.option_categories if self.params.get("key_by", "category") == "category" else meta.option_ids
        out = np.empty(meta.n_options)
        metric = self.params.get("metric", "cosine")
        for j, k in enumerate(keys):
            e = self._table.get(str(k).lower())
            if e is None:
                raise AdapterError(f"no embedding for {k!r}; have {sorted(self._table)[:20]}")
            if metric == "cosine":
                out[j] = float(v @ e / (np.linalg.norm(v) * np.linalg.norm(e) + 1e-12))
            elif metric == "dot":
                out[j] = float(v @ e)
            elif metric == "neg_euclidean":
                out[j] = -float(np.linalg.norm(v - e))
            else:
                raise AdapterError(f"metric must be cosine, dot or neg_euclidean, got {metric!r}")
        return out, {}


# --------------------------------------------------------------------------- auto-detect

ADAPTERS: dict[str, type[Adapter]] = {
    "per_option": PerOptionAdapter,
    "per_flash": PerFlashAdapter,
    "probabilities": ProbabilitiesAdapter,
    "epoch_target_prob": EpochTargetProbAdapter,
    "class_probabilities": ClassProbabilitiesAdapter,
    "embedding": EmbeddingAdapter,
}


def _summ(raw: Any) -> str:
    if isinstance(raw, dict):
        return f"dict with keys {list(raw.keys())[:8]}"
    if _is_numeric_seq(raw):
        return f"array of shape {np.asarray(raw).shape}"
    return type(raw).__name__


def detect_candidates(raw: Any, meta: TrialMeta, params: dict[str, Any] | None = None) -> list[tuple[str, float, str]]:
    """Rank adapters for one sample. Returns ``[(name, confidence 0-1, reason), ...]`` best first."""
    params = params or {}
    if params.get("field") and isinstance(raw, dict) and params["field"] in raw:
        raw = raw[params["field"]]
    n_opt, n_fl = meta.n_options, len(meta.flashes)
    class_names = params.get("class_names")
    cands: list[tuple[str, float, str]] = []

    def in01(a: np.ndarray) -> bool:
        a = a[np.isfinite(a)]
        return a.size > 0 and a.min() >= 0 and a.max() <= 1

    if isinstance(raw, dict):
        oid_f = _find_field(raw, OPTION_ID_FIELDS)
        val_f = _find_field(raw, SCORE_FIELDS + PROB_FIELDS)
        emb_f = _find_field(raw, EMB_FIELDS)
        if emb_f is not None:
            cands.append(("embedding", 0.8, f"has an embedding field '{emb_f}'"))
        if oid_f is not None and val_f is not None:
            ids = raw[oid_f] if isinstance(raw[oid_f], (list, tuple, np.ndarray)) else [raw[oid_f]]
            vals = _as_float_array(raw[val_f]).reshape(-1)
            p01 = in01(vals)
            if len(ids) == n_opt and len(set(map(str, ids))) == n_opt:
                cands.append(("probabilities" if p01 else "per_option", 0.9,
                              f"'{oid_f}' + '{val_f}' with one value per option" + (" in [0,1]" if p01 else "")))
                if p01:
                    cands.append(("per_option", 0.4, "values in [0,1] could also be plain scores"))
            elif len(ids) > n_opt:
                cands.append(("epoch_target_prob" if p01 else "per_flash", 0.9,
                              f"'{oid_f}' + '{val_f}' with {len(ids)} rows (repeated options = flashes)"
                              + (" in [0,1]" if p01 else "")))
                if p01:
                    cands.append(("per_flash", 0.4, "values in [0,1] could also be plain scores"))
        elif val_f is not None and _is_numeric_seq(raw[val_f]):
            return detect_candidates(raw[val_f], meta, {k: v for k, v in params.items() if k != "field"})
        else:
            keys = _strip_prefix(raw.keys(), params.get("key_prefix"))
            numeric = {k: v for k, v in keys.items() if _scalar_number(raw[v])}
            vals = np.array([float(raw[v]) for v in numeric.values()]) if numeric else np.array([])
            if set(meta.option_ids) <= set(numeric):
                cands.append(("probabilities" if in01(vals) else "per_option", 0.85, "dict keyed by option ids"))
            cats = {c.lower() for c in meta.option_categories if c}
            lk = {k.lower() for k in numeric}
            if cats and cats <= lk and len(numeric) > n_opt:
                cands.append(("class_probabilities", 0.85 if in01(vals) else 0.5,
                              f"dict over {len(numeric)} labels that include the option categories"))
            elif cats and cats <= lk:
                cands.append(("per_option", 0.7, "dict keyed by option categories"))
            elif len(numeric) > n_opt and in01(vals):
                s1 = bool(np.isclose(vals.sum(), 1, atol=1e-3))
                cands.append(("class_probabilities", 0.6 if s1 else 0.4,
                              f"dict over {len(numeric)} labels with values in [0,1]"
                              + (" summing to 1" if s1 else "") + " (pass the stimulus log to check the labels)"))
    elif _is_numeric_seq(raw):
        a = _as_float_array(raw)
        if a.ndim == 2 and 1 in a.shape:
            a = a.reshape(-1)
        if a.ndim == 2:
            if a.shape[1] == n_opt:
                cands.append(("per_flash", 0.85, f"2-D array [{a.shape[0]} repeats, {n_opt} options]"))
            if n_fl and a.shape == (n_fl, 2):
                rs = np.allclose(a.sum(1), 1, atol=1e-3)
                cands.append(("epoch_target_prob", 0.9 if rs else 0.5,
                              f"[{n_fl} flashes, 2] looks like predict_proba (non-target, target)"))
        elif a.ndim == 1:
            p01, s1 = in01(a), bool(np.isclose(a.sum(), 1, atol=1e-3))
            if a.shape[0] == n_opt:
                if p01:
                    cands.append(("probabilities", 0.85 if s1 else 0.7,
                                  f"{n_opt} values in [0,1]" + (" summing to 1" if s1 else "")))
                    cands.append(("per_option", 0.4, "could also be plain scores"))
                else:
                    cands.append(("per_option", 0.9, f"{n_opt} unbounded values, one per option"))
            if n_fl and a.shape[0] == n_fl:
                cands.append(("epoch_target_prob" if p01 else "per_flash", 0.9,
                              f"{n_fl} values = number of flashes in the marker log" + (" in [0,1]" if p01 else "")))
            if class_names and a.shape[0] == len(class_names):
                cands.append(("class_probabilities", 0.9, f"{a.shape[0]} values = number of class_names"))
            elif a.shape[0] not in (n_opt, n_fl) and p01 and s1:
                cands.append(("class_probabilities", 0.6,
                              f"{a.shape[0]} values in [0,1] summing to 1 (needs class_names)"))
            elif a.shape[0] not in (n_opt, n_fl):
                cands.append(("embedding", 0.5, f"{a.shape[0]}-dim vector (needs option embeddings)"))
    cands.sort(key=lambda c: -c[1])
    # dedupe by name keeping the best
    seen, out = set(), []
    for c in cands:
        if c[0] not in seen:
            seen.add(c[0])
            out.append(c)
    return out


class AutoAdapter(Adapter):
    """Detects the output shape on the first trial and then sticks with that adapter.

    Locking in keeps every trial on the same conversion. Use ``describe`` and
    write the chosen adapter into the config once you know it.
    """

    name = "auto"

    @classmethod
    def allowed_params(cls):
        allowed = set()
        for a in ADAPTERS.values():
            allowed |= a.allowed_params()
        return allowed

    def __init__(self, **kw):
        super().__init__(**kw)
        self.chosen: Adapter | None = None
        self.reason: str | None = None

    @property
    def per_flash(self):  # type: ignore[override]
        return bool(self.chosen and self.chosen.per_flash)

    def to_trial_scores(self, raw_output, meta, model="model", max_repeat=None):
        if self.chosen is None:
            cands = detect_candidates(raw_output, meta, {"field": self.field, **self.params})
            if not cands:
                raise AdapterError(f"could not recognise the output format ({_summ(raw_output)}). Run "
                                   "`cortext-fusion describe <file>` and set 'adapter:' in the config.",
                                   trial_id=meta.trial_id, adapter="auto")
            name, conf, reason = cands[0]
            cls = ADAPTERS[name]
            p = {k: v for k, v in self.params.items() if k in cls.allowed_params()}
            self.chosen = cls(lower_is_better=self.lower_is_better, field=self.field, **p)
            self.reason = f"{name} (confidence {conf:.2f}): {reason}"
        return self.chosen.to_trial_scores(raw_output, meta, model=model, max_repeat=max_repeat)

    def describe(self):
        d = super().describe()
        d["chosen"] = self.reason
        return d


def make_adapter(name: str, **options: Any) -> Adapter:
    """Build an adapter by config name. ``distance`` = per_option with lower_is_better."""
    if name == "distance":
        options["lower_is_better"] = True
        name = "per_option"
    if name == "auto":
        return AutoAdapter(**options)
    if name not in ADAPTERS:
        raise AdapterError(f"unknown adapter {name!r}; choose one of {sorted(ADAPTERS) + ['auto', 'distance']}")
    return ADAPTERS[name](**options)

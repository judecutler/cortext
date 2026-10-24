"""Mock data: a fake stimulus log plus fake model outputs in every supported format.

Signal strength is adjustable, so the fusion layer and the evaluation (including
its controls) can be tested before the real models exist. With
``p300_signal=0`` and ``category_signal=0`` everything should score at chance.

The simulated participant attends to the option they later click. Usually that
is the true answer; ``click_error_rate`` makes them pick a different option on
some trials, so "accuracy vs true answer" and "accuracy vs clicked answer" differ.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.special import softmax

from .schema import Flash, TrialMeta

DEFAULT_CATEGORIES = ["dog", "cat", "bird", "fish", "car", "boat", "house", "tree",
                      "flower", "apple", "chair", "hammer"]
INVALID_REASONS = ["misclick", "blink or movement", "distracted", "equipment glitch"]

FORMATS = {
    # name: (model role, adapter, description)
    "p300_per_flash_csv": ("p300", "per_flash", "long CSV: trial_id, option_id, repeat_index, score"),
    "p300_epoch_prob_csv": ("p300", "epoch_target_prob", "long CSV: trial_id, option_id, repeat_index, p_target"),
    "p300_predict_proba_npy": ("p300", "epoch_target_prob", ".npy [trials, flashes, 2] + sidecar trial ids"),
    "p300_grid_npy": ("p300", "per_flash", ".npy [trials, repeats, options] + sidecar trial ids"),
    "p300_peak_amplitude_json": ("p300", "per_option", "JSON {trial_id: {A: uV, B: uV, ...}} (averaged-ERP peak amplitude)"),
    "p300_probs_json": ("p300", "probabilities", "JSON records: trial_id, option_ids, probs (softmax over options)"),
    "p300_distance_json": ("p300", "distance", "JSON {trial_id: [d_A, d_B, ...]} lower is better"),
    "p300_per_trial_folder": ("p300", "per_flash", "folder of <trial_id>.json files with option_id/score lists"),
    "category_class_probs_csv": ("category", "class_probabilities", "wide CSV: trial_id, p_dog, p_cat, ..."),
    "category_class_probs_npy": ("category", "class_probabilities", ".npy [trials, classes] + class_names.txt"),
    "category_embedding_npy": ("category", "embedding", ".npy [trials, dim] + option_embeddings.json"),
}


@dataclass
class MockSpec:
    n_participants: int = 3
    sessions_per_participant: int = 2
    trials_per_session: int = 40
    n_options: int = 4
    n_repeats: int = 10
    categories: tuple[str, ...] = tuple(DEFAULT_CATEGORIES)
    p300_signal: float = 0.6         # per-flash effect size (d') for the attended option
    category_signal: float = 1.0     # logit boost for the attended option's category
    click_error_rate: float = 0.05
    invalid_rate: float = 0.04
    embedding_dim: int = 16
    flash_on_ms: float = 500.0
    flash_off_ms: float = 150.0
    seed: int = 0


def generate_log(spec: MockSpec, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, list[TrialMeta], dict[str, int]]:
    option_ids = [chr(ord("A") + i) for i in range(spec.n_options)]
    markers, trials, metas, attended = [], [], [], {}
    t0 = 1000.0
    for p in range(spec.n_participants):
        pid = f"P{p + 1:02d}"
        for s in range(spec.sessions_per_participant):
            sid = f"{pid}_S{s + 1}"
            for k in range(spec.trials_per_session):
                tid = f"{sid}_T{k + 1:03d}"
                cats = list(rng.choice(spec.categories, size=spec.n_options, replace=False))
                tgt = int(rng.integers(spec.n_options))
                att = tgt
                if rng.random() < spec.click_error_rate:
                    att = int(rng.choice([j for j in range(spec.n_options) if j != tgt]))
                invalid = rng.random() < spec.invalid_rate
                attended[tid] = att
                flashes, last = [], None
                for r in range(spec.n_repeats):
                    order = list(rng.permutation(spec.n_options))
                    if last is not None and order[0] == last:  # no immediate repeats
                        order[0], order[1] = order[1], order[0]
                    for j in order:
                        markers.append({
                            "timestamp": round(t0, 4), "trial_id": tid, "option_id": option_ids[j],
                            "true_category": cats[j], "is_target": int(j == tgt), "repeat_index": r,
                            "session_id": sid, "participant_id": pid,
                        })
                        flashes.append(Flash(option_ids[j], r, round(t0, 4)))
                        t0 += (spec.flash_on_ms + spec.flash_off_ms) / 1000
                    last = order[-1]
                t0 += 3.0
                reason = str(rng.choice(INVALID_REASONS)) if invalid else None
                trials.append({
                    "trial_id": tid, "session_id": sid, "participant_id": pid,
                    "prompt": f"Which one is the {cats[tgt]}?", "option_order": "|".join(option_ids),
                    "target_category": cats[tgt], "target_option": option_ids[tgt],
                    "clicked_option": option_ids[att], "valid": not invalid, "invalid_reason": reason,
                })
                metas.append(TrialMeta(tid, list(option_ids), cats, cats[tgt], option_ids[tgt], option_ids[att],
                                       not invalid, reason, sid, pid, f"Which one is the {cats[tgt]}?", flashes))
    return pd.DataFrame(markers), pd.DataFrame(trials), metas, attended


def _flash_scores(meta: TrialMeta, att: int, d: float, rng) -> np.ndarray:
    return np.array([rng.normal(d if meta.option_index(f.option_id) == att else 0.0, 1.0) for f in meta.flashes])


def generate(out_dir: str | Path, spec: MockSpec | None = None, formats: list[str] | None = None) -> dict[str, Any]:
    """Write a mock stimulus log, model outputs and ready-to-run configs to ``out_dir``."""
    spec = spec or MockSpec()
    rng = np.random.default_rng(spec.seed)
    out = Path(out_dir)
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    mdf, tdf, metas, attended = generate_log(spec, rng)
    mdf.to_csv(out / "markers.csv", index=False)
    tdf.to_csv(out / "trials.csv", index=False)
    formats = formats or list(FORMATS)
    unknown = set(formats) - set(FORMATS)
    if unknown:
        raise ValueError(f"unknown mock formats {sorted(unknown)}; choose from {sorted(FORMATS)}")

    # invalid trials carry no signal (the participant was not doing the task)
    def att_or_none(m: TrialMeta):
        return attended[m.trial_id] if m.valid else -1

    ids = [m.trial_id for m in metas]
    flash_sc = {m.trial_id: _flash_scores(m, att_or_none(m), spec.p300_signal, rng) for m in metas}
    cats = list(spec.categories)
    written: dict[str, str] = {}
    ids_txt = "\n".join(ids) + "\n"

    def opt_mean(m: TrialMeta, vals: np.ndarray) -> np.ndarray:
        return np.array([vals[[f.option_id == o for f in m.flashes]].mean() for o in m.option_ids])

    for fmt in formats:
        path: Path
        if fmt == "p300_per_flash_csv":
            rows = [{"trial_id": m.trial_id, "option_id": f.option_id, "repeat_index": f.repeat_index,
                     "score": round(float(v), 5)} for m in metas for f, v in zip(m.flashes, flash_sc[m.trial_id])]
            path = out / "outputs" / "p300_per_flash.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
        elif fmt == "p300_epoch_prob_csv":
            rows = []
            for m in metas:
                lr = spec.p300_signal * flash_sc[m.trial_id] - spec.p300_signal ** 2 / 2 + np.log(1 / (m.n_options - 1))
                pt = 1 / (1 + np.exp(-lr))
                rows += [{"trial_id": m.trial_id, "option_id": f.option_id, "repeat_index": f.repeat_index,
                          "p_target": round(float(p), 6)} for f, p in zip(m.flashes, pt)]
            path = out / "outputs" / "p300_epoch_prob.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
        elif fmt == "p300_predict_proba_npy":
            arr = []
            for m in metas:
                pt = 1 / (1 + np.exp(-(spec.p300_signal * flash_sc[m.trial_id] - 1.2)))
                arr.append(np.stack([1 - pt, pt], axis=1))
            path = out / "outputs" / "p300_predict_proba.npy"
            np.save(path, np.array(arr))
            (out / "outputs" / "p300_predict_proba_trial_ids.txt").write_text(ids_txt)
        elif fmt == "p300_grid_npy":
            grid = []
            for m in metas:
                g = np.zeros((spec.n_repeats, m.n_options))
                for f, v in zip(m.flashes, flash_sc[m.trial_id]):
                    g[f.repeat_index, m.option_index(f.option_id)] = v
                grid.append(g)
            path = out / "outputs" / "p300_grid.npy"
            np.save(path, np.array(grid))
            (out / "outputs" / "p300_grid_trial_ids.txt").write_text(ids_txt)
        elif fmt == "p300_peak_amplitude_json":
            data = {m.trial_id: {o: round(float(3.0 + 2.0 * v), 4) for o, v in zip(m.option_ids, opt_mean(m, flash_sc[m.trial_id]))}
                    for m in metas}
            path = out / "outputs" / "p300_peak_amplitude.json"
            path.write_text(json.dumps(data, indent=1))
        elif fmt == "p300_probs_json":
            recs = []
            for m in metas:
                p = softmax(3.0 * opt_mean(m, flash_sc[m.trial_id]))
                p[0] = 0.0 if rng.random() < 0.05 else p[0]  # exercise the 0/1-safe logit
                recs.append({"trial_id": m.trial_id, "option_ids": m.option_ids, "probs": [round(float(x), 6) for x in p]})
            path = out / "outputs" / "p300_probs.json"
            path.write_text(json.dumps(recs))
        elif fmt == "p300_distance_json":
            data = {m.trial_id: [round(float(5.0 - v), 5) for v in opt_mean(m, flash_sc[m.trial_id])] for m in metas}
            path = out / "outputs" / "p300_distance.json"
            path.write_text(json.dumps(data))
        elif fmt == "p300_per_trial_folder":
            path = out / "outputs" / "p300_per_trial"
            path.mkdir(exist_ok=True)
            for m in metas:
                (path / f"{m.trial_id}.json").write_text(json.dumps({
                    "option_id": [f.option_id for f in m.flashes],
                    "score": [round(float(v), 5) for v in flash_sc[m.trial_id]]}))
        elif fmt in ("category_class_probs_csv", "category_class_probs_npy"):
            probs = []
            for m in metas:
                logits = rng.normal(0, 1, len(cats))
                a = att_or_none(m)
                if a >= 0:
                    logits[cats.index(m.option_categories[a])] += spec.category_signal
                probs.append(softmax(logits))
            probs = np.array(probs)
            if fmt == "category_class_probs_csv":
                df = pd.DataFrame(np.round(probs, 6), columns=[f"p_{c}" for c in cats])
                df.insert(0, "trial_id", ids)
                path = out / "outputs" / "category_class_probs.csv"
                df.to_csv(path, index=False)
            else:
                path = out / "outputs" / "category_class_probs.npy"
                np.save(path, probs)
                (out / "outputs" / "category_class_probs_trial_ids.txt").write_text(ids_txt)
                (out / "outputs" / "class_names.txt").write_text("\n".join(cats) + "\n")
        elif fmt == "category_embedding_npy":
            emb = {c: rng.normal(0, 1, spec.embedding_dim) for c in cats}
            (out / "outputs" / "option_embeddings.json").write_text(json.dumps({c: v.round(5).tolist() for c, v in emb.items()}))
            vecs = []
            for m in metas:
                a = att_or_none(m)
                base = emb[m.option_categories[a]] * spec.category_signal * 0.5 if a >= 0 else 0
                vecs.append(base + rng.normal(0, 1, spec.embedding_dim))
            path = out / "outputs" / "category_embedding.npy"
            np.save(path, np.array(vecs))
            (out / "outputs" / "category_embedding_trial_ids.txt").write_text(ids_txt)
        written[fmt] = str(path.relative_to(out))

    configs = write_mock_configs(out, written)
    (out / "mock_spec.json").write_text(json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                                                     for k, v in spec.__dict__.items()}, indent=1))
    return {"out_dir": str(out), "n_trials": len(metas), "outputs": written, "configs": configs}


def _model_entry(fmt: str, rel: str, weight: float) -> dict[str, Any]:
    role, adapter, _ = FORMATS[fmt]
    e: dict[str, Any] = {"name": role, "role": role, "source": rel, "adapter": adapter, "weight": weight}
    if adapter == "class_probabilities":
        e["options"] = {"key_prefix": "p_", "check_target": True} if fmt.endswith("csv") else \
            {"class_names_file": "outputs/class_names.txt", "check_target": True}
    if adapter == "embedding":
        e["options"] = {"embeddings": "outputs/option_embeddings.json"}
    if adapter == "probabilities":
        e["options"] = {"transform": "log"}
    return e


def write_mock_configs(out: Path, written: dict[str, str]) -> dict[str, str]:
    """One config per P300 format (paired with the first category format), plus 'mock_config.yaml'."""
    cat_fmts = [f for f in written if FORMATS[f][0] == "category"]
    p300_fmts = [f for f in written if FORMATS[f][0] == "p300"]
    configs = {}
    base = {"stimulus_log": {"markers": "markers.csv", "trials": "trials.csv"},
            "fusion": {"normalize": "zscore", "unsure_threshold": 0.4, "calibration_file": "calibration.yaml"},
            "sequential": {"stop_confidence": 0.9, "min_repeats": 3},
            "evaluation": {"holdout": "session", "n_shuffles": 100, "n_noise": 20, "output_dir": "report"}}
    for i, pf in enumerate(p300_fmts or [None]):
        models = []
        if pf:
            models.append(_model_entry(pf, written[pf], 1.0))
        if cat_fmts:
            models.append(_model_entry(cat_fmts[0], written[cat_fmts[0]], 0.0))
        cfg = dict(base, models=models)
        name = "mock_config.yaml" if i == 0 else f"config_{pf}.yaml"
        (out / name).write_text(yaml.safe_dump(cfg, sort_keys=False))
        configs[pf or "category_only"] = name
    for cf in cat_fmts[1:]:
        models = [_model_entry(p300_fmts[0], written[p300_fmts[0]], 1.0)] if p300_fmts else []
        models.append(_model_entry(cf, written[cf], 0.0))
        (out / f"config_{cf}.yaml").write_text(yaml.safe_dump(dict(base, models=models), sort_keys=False))
        configs[cf] = f"config_{cf}.yaml"
    return configs


class MockPredictor:
    """Fake live predictor for the demo app: biased toward the option it is told is correct.

    ``accuracy`` is the approximate chance of picking the hinted option. With no
    hint it returns uniform-ish random scores. Every response is marked mock.
    """

    def __init__(self, accuracy: float = 0.8, seed: int | None = None):
        self.accuracy = accuracy
        self.rng = np.random.default_rng(seed)

    def scores(self, n_options: int, hint_index: int | None = None) -> np.ndarray:
        s = self.rng.normal(0, 1, n_options)
        if hint_index is not None and 0 <= hint_index < n_options:
            if self.rng.random() < self.accuracy:
                s[hint_index] = s.max() + abs(self.rng.normal(1.0, 0.5))
        return s

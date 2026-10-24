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


MARKER_COLUMNS = ["seq", "session_id", "event", "t_ms", "frame_ms", "wall_ms", "trial_id", "trial_index", "option_id",
                  "option_position", "category", "is_target", "repeat_index", "flash_index", "true_answer_id",
                  "bridge_sent", "extra"]
TRIAL_COLUMNS = ["session_id", "trial_id", "trial_index", "attempt", "type", "status", "valid", "invalid_reason",
                 "invalid_note", "prompt", "tile_content", "option_ids_by_position", "true_answer_id",
                 "true_answer_category", "clicked_answer_id", "clicked_category", "correct", "rt_ms", "early_clicks",
                 "n_flashes", "flash_order", "flash_on_ms", "flash_off_min_ms", "flash_off_max_ms", "repeats",
                 "prediction_source", "prediction_best_option_id", "prediction_confidence", "started_wall_ms",
                 "ended_wall_ms", "notes"]
LAB_INVALID_REASONS = ["misclick", "blink_or_movement", "distracted", "equipment_glitch"]


def generate_log(spec: MockSpec, rng: np.random.Generator, out: Path | None = None
                 ) -> tuple[list[TrialMeta], dict[str, int]]:
    """Simulate CorText Lab sessions; write them as Lab exports under ``out/lab_export/<session_id>/``.

    The files use the app's exact column order (cortext-lab/web/js/schema.js), integer trial ids that
    restart in every session, item ids as option ids, and non-flash marker events, so the fusion
    layer is tested against the real export format.
    """
    from .stimulus_log import make_key
    metas, attended = [], {}
    wall0 = 1.7614e12
    for p in range(spec.n_participants):
        pid = f"P{p + 1:02d}"
        for s in range(spec.sessions_per_participant):
            sid = f"{pid}_20261024{p:02d}{s:02d}00_m{s}"
            markers, trials, seq, t_ms = [], [], 0, 1000.0

            def mk(event, **kw):
                nonlocal seq
                row = {c: "" for c in MARKER_COLUMNS}
                row.update(seq=seq, session_id=sid, event=event, t_ms=round(t_ms, 3), frame_ms=round(t_ms, 3),
                           wall_ms=round(wall0 + t_ms, 1), bridge_sent="true", **kw)
                seq += 1
                markers.append(row)

            mk("session_start", extra=json.dumps({"participant_id": pid}))
            for k in range(spec.trials_per_session):
                cats = list(rng.choice(spec.categories, size=spec.n_options, replace=False))
                items = [f"{c}_{int(rng.integers(1, 4)):02d}" for c in cats]  # item ids, in screen-position order
                tgt = int(rng.integers(spec.n_options))
                att = tgt
                if rng.random() < spec.click_error_rate:
                    att = int(rng.choice([j for j in range(spec.n_options) if j != tgt]))
                invalid = rng.random() < spec.invalid_rate
                key = make_key(sid, k)
                attended[key] = att
                base = dict(trial_id=k, trial_index=k, true_answer_id=items[tgt])
                mk("trial_start", **base)
                flashes, last, fi = [], None, 0
                for r in range(spec.n_repeats):
                    order = list(rng.permutation(spec.n_options))
                    if last is not None and order[0] == last:  # no immediate repeats
                        order[0], order[1] = order[1], order[0]
                    for j in order:
                        fl = dict(base, option_id=items[j], option_position=j, category=cats[j],
                                  is_target="true" if j == tgt else "false", repeat_index=r, flash_index=fi)
                        mk("flash_on", **fl)
                        flashes.append(Flash(items[j], r, round(t_ms, 3)))
                        t_ms += spec.flash_on_ms
                        mk("flash_off", **fl)
                        t_ms += spec.flash_off_ms
                        fi += 1
                    last = order[-1]
                mk("response_open", **base)
                t_ms += 900
                mk("click", **base, option_id=items[att], option_position=att, category=cats[att])
                t_ms += 1500
                reason = str(rng.choice(LAB_INVALID_REASONS)) if invalid else ""
                trials.append({**{c: "" for c in TRIAL_COLUMNS}, **dict(
                    session_id=sid, trial_id=k, trial_index=k, attempt=1, type="mcq_flash", status="complete",
                    valid="false" if invalid else "true", invalid_reason=reason,
                    prompt=f"Which one is the {cats[tgt]}?", tile_content="image",
                    option_ids_by_position="|".join(items), true_answer_id=items[tgt], true_answer_category=cats[tgt],
                    clicked_answer_id=items[att], clicked_category=cats[att], correct=str(att == tgt).lower(),
                    rt_ms=900, early_clicks=0, n_flashes=len(flashes), flash_order="|".join(f.option_id for f in flashes),
                    flash_on_ms=spec.flash_on_ms, flash_off_min_ms=spec.flash_off_ms, flash_off_max_ms=spec.flash_off_ms,
                    repeats=spec.n_repeats)})
                metas.append(TrialMeta(key, items, cats, cats[tgt], items[tgt], items[att], not invalid,
                                       reason or None, sid, pid, f"Which one is the {cats[tgt]}?", flashes,
                                       local_trial_id=str(k)))
            # one trial that was aborted before any flash (must load as invalid, never vanish)
            trials.append({**{c: "" for c in TRIAL_COLUMNS}, **dict(
                session_id=sid, trial_id=spec.trials_per_session, trial_index=spec.trials_per_session, attempt=1,
                type="mcq_flash", status="aborted", valid="false", invalid_reason="aborted",
                option_ids_by_position="|".join(f"{c}_01" for c in spec.categories[: spec.n_options]),
                true_answer_id=f"{spec.categories[0]}_01", true_answer_category=spec.categories[0])})
            if out is not None:
                d = out / "lab_export" / sid
                d.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(markers, columns=MARKER_COLUMNS).to_csv(d / "markers.csv", index=False)
                pd.DataFrame(trials, columns=TRIAL_COLUMNS).to_csv(d / "trials.csv", index=False)
                (d / "session.json").write_text(json.dumps({"schema_version": 1, "session": {
                    "session_id": sid, "participant_id": pid, "device": "MOCK", "status": "complete"},
                    "markers_count": len(markers)}, indent=1))
    return metas, attended


def _flash_scores(meta: TrialMeta, att: int, d: float, rng) -> np.ndarray:
    return np.array([rng.normal(d if meta.option_index(f.option_id) == att else 0.0, 1.0) for f in meta.flashes])


def generate(out_dir: str | Path, spec: MockSpec | None = None, formats: list[str] | None = None) -> dict[str, Any]:
    """Write a mock stimulus log, model outputs and ready-to-run configs to ``out_dir``."""
    spec = spec or MockSpec()
    rng = np.random.default_rng(spec.seed)
    out = Path(out_dir)
    (out / "outputs").mkdir(parents=True, exist_ok=True)
    metas, attended = generate_log(spec, rng, out)
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

    def st(m: TrialMeta) -> dict[str, str]:  # CSV outputs carry session_id + trial_id columns, like the app
        return {"session_id": m.session_id, "trial_id": m.local_trial_id}

    def opt_mean(m: TrialMeta, vals: np.ndarray) -> np.ndarray:
        return np.array([vals[[f.option_id == o for f in m.flashes]].mean() for o in m.option_ids])

    for fmt in formats:
        path: Path
        if fmt == "p300_per_flash_csv":
            rows = [{**st(m), "option_id": f.option_id, "repeat_index": f.repeat_index,
                     "score": round(float(v), 5)} for m in metas for f, v in zip(m.flashes, flash_sc[m.trial_id])]
            path = out / "outputs" / "p300_per_flash.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
        elif fmt == "p300_epoch_prob_csv":
            rows = []
            for m in metas:
                lr = spec.p300_signal * flash_sc[m.trial_id] - spec.p300_signal ** 2 / 2 + np.log(1 / (m.n_options - 1))
                pt = 1 / (1 + np.exp(-lr))
                rows += [{**st(m), "option_id": f.option_id, "repeat_index": f.repeat_index,
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
                (path / f"{m.session_id}__{m.local_trial_id}.json").write_text(json.dumps({
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
                df.insert(0, "trial_id", [m.local_trial_id for m in metas])
                df.insert(0, "session_id", [m.session_id for m in metas])
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
    base = {"stimulus_log": {"lab_export": "lab_export"},
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

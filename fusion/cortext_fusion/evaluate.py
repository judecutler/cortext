"""Evaluation with required controls, held out by session or participant.

What is reported
----------------
* accuracy of each model alone, fused with the config weights, and fused with
  weights calibrated on *other* sessions/participants (leave-one-group-out)
* against two ground truths: the true answer and the participant's clicked
  answer; trials flagged invalid are skipped (and counted)
* accuracy vs number of repeats (per-flash models), plus simulated early stopping
* reliability (calibration) plot + expected calibration error, confusion matrix
* controls: shuffled-label, noise-score (whole pipeline re-run on Gaussian
  noise), prior-only (most common answer position in the training groups)
* diagnostics: everything the loader flagged, plus models with no evidence of
  information or below-chance accuracy
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import softmax

from .config import Config
from .fusion import CalibrationResult, Fuser, ModelWeight, calibrate, fit_conditional_logit, normalize
from .pipeline import Dataset, Issue
from .schema import TrialMeta

GROUND_TRUTHS = ("clicked", "target")


# --------------------------------------------------------------------------- small stats

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, bins: int = 10) -> tuple[float, pd.DataFrame]:
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        acc, cf = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - cf)
        rows.append({"bin_lo": lo, "bin_hi": hi, "n": int(m.sum()), "mean_confidence": cf, "accuracy": acc})
    return float(ece), pd.DataFrame(rows)


# --------------------------------------------------------------------------- core

@dataclass
class EvalInputs:
    metas: list[TrialMeta]                  # valid trials only, in a fixed order
    labels: dict[str, np.ndarray]           # ground truth -> label index per trial
    groups: np.ndarray
    n_options: int
    n_invalid: int
    n_unlabeled: int


def prepare(ds: Dataset, holdout: str) -> EvalInputs:
    valid = [m for m in ds.metas.values() if m.valid]
    n_invalid = len(ds.metas) - len(valid)
    keep = [m for m in valid if m.clicked_option is not None or m.target_option is not None]
    labels = {}
    for gt in GROUND_TRUTHS:
        idx = [(m.clicked_index() if gt == "clicked" else m.target_index()) for m in keep]
        labels[gt] = np.array([-1 if i is None else i for i in idx])
    grp = np.array([str((m.session_id if holdout == "session" else m.participant_id) or "all") for m in keep])
    n_opts = {m.n_options for m in keep}
    if len(n_opts) > 1:
        raise ValueError(f"trials have different numbers of options {sorted(n_opts)}; evaluate them separately")
    return EvalInputs(keep, labels, grp, n_opts.pop() if n_opts else 0, n_invalid, len(valid) - len(keep))


def score_matrix(inp: EvalInputs, table: Mapping[str, Mapping[str, np.ndarray]], model: str) -> np.ndarray:
    """[trials, options] raw scores (NaN rows where the model has no output)."""
    S = np.full((len(inp.metas), inp.n_options), np.nan)
    for t, m in enumerate(inp.metas):
        s = table.get(model, {}).get(m.trial_id)
        if s is not None and len(s) == inp.n_options:
            S[t] = s
    return S


def norm_tensor(inp: EvalInputs, table, models: Sequence[str], norms: Mapping[str, str]) -> np.ndarray:
    X = np.zeros((len(inp.metas), inp.n_options, len(models)))
    for k, name in enumerate(models):
        S = score_matrix(inp, table, name)
        for t in range(len(S)):
            if np.all(np.isfinite(S[t])):
                X[t, :, k] = normalize(S[t], norms.get(name, "zscore"))
    return X


def predict_from_tensor(X: np.ndarray, w: np.ndarray, temperature: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    u = (X @ w) / max(temperature, 1e-6)
    P = softmax(u, axis=1)
    pred = np.argmax(P, axis=1)
    pred[np.all(np.isclose(P, P[:, :1]), axis=1)] = -1  # all-equal -> no prediction
    return pred, P


def _acc(pred: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    m = y >= 0
    n = int(m.sum())
    k = int((pred[m] == y[m]).sum())
    lo, hi = wilson_ci(k, n)
    return {"accuracy": k / n if n else float("nan"), "n": n, "correct": k, "ci95": [lo, hi]}


def shuffled_label_test(pred: np.ndarray, y: np.ndarray, groups: np.ndarray, n: int, rng) -> dict[str, Any]:
    """Permute labels within each group; how often does a random labeling score as high?"""
    m = y >= 0
    obs = float((pred[m] == y[m]).mean()) if m.any() else float("nan")
    null = np.empty(n)
    for i in range(n):
        yp = y.copy()
        for g in np.unique(groups):
            idx = np.where((groups == g) & m)[0]
            yp[idx] = yp[rng.permutation(idx)]
        null[i] = (pred[m] == yp[m]).mean()
    p = (1 + np.sum(null >= obs)) / (n + 1)
    return {"observed": obs, "null_mean": float(null.mean()), "null_95pct": float(np.quantile(null, 0.95)),
            "p_value": float(p), "n_shuffles": n}


def logo_calibrated(inp: EvalInputs, table, models: list[str], norms: dict[str, str], baseline: list[str],
                    y_train: np.ndarray, min_gain: float) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Leave-one-group-out: calibrate on the other groups, predict the held-out group."""
    X = norm_tensor(inp, table, models, norms)
    pred = np.full(len(inp.metas), -1)
    P = np.full((len(inp.metas), inp.n_options), 1.0 / max(inp.n_options, 1))
    folds = []
    ids = [m.trial_id for m in inp.metas]
    for g in np.unique(inp.groups):
        te = inp.groups == g
        tr = (~te) & (y_train >= 0)
        if tr.sum() < 5:
            continue
        tr_metas = [m for m, k in zip(inp.metas, tr) if k]
        labels = {m.trial_id: int(y) for m, y, k in zip(inp.metas, y_train, tr) if k}
        grp = {m.trial_id: str(gg) for m, gg, k in zip(inp.metas, inp.groups, tr) if k}
        cal = calibrate(tr_metas, {mn: table.get(mn, {}) for mn in models}, models, labels, norms,
                        baseline_models=baseline, groups=grp, min_gain=min_gain)
        w = np.array([cal.weights[mn] for mn in models])
        pr, Pg = predict_from_tensor(X[te], w)
        pred[te], P[te] = pr, Pg
        folds.append({"held_out": str(g), "n_train": int(tr.sum()), "n_test": int(te.sum()),
                      "weights": cal.weights, "selected": cal.selected_models})
    return pred, P, folds


def prior_only(inp: EvalInputs, y_train: np.ndarray) -> np.ndarray:
    pred = np.full(len(inp.metas), -1)
    for g in np.unique(inp.groups):
        te = inp.groups == g
        tr = (~te) & (y_train >= 0)
        if not tr.any():
            continue
        pred[te] = np.bincount(y_train[tr], minlength=inp.n_options).argmax()
    return pred


@dataclass
class EvalResult:
    metrics: dict[str, Any]
    predictions: pd.DataFrame
    issues: list[Issue]
    reliability: pd.DataFrame
    confusion: pd.DataFrame
    repeats: pd.DataFrame
    figures: dict[str, str] = field(default_factory=dict)


def evaluate(ds: Dataset, cfg: Config, train_on: str = "clicked", out_dir: str | Path | None = None,
             make_plots: bool = True) -> EvalResult:
    ev = cfg.evaluation
    rng = np.random.default_rng(ev.seed)
    inp = prepare(ds, ev.holdout)
    issues = list(ds.issues)
    if len(inp.metas) == 0:
        raise ValueError("no valid labeled trials to evaluate")
    models = [m.name for m in cfg.enabled_models if m.name in ds.models]
    norms = {m.name: m.normalize or cfg.fusion.normalize for m in cfg.enabled_models}
    cfg_weights = {m.name: m.weight for m in cfg.enabled_models}
    baseline = [m for m in models if cfg_weights.get(m, 0) != 0] or models[:1]
    table = ds.score_table()
    chance = 1.0 / inp.n_options
    y_tr = inp.labels[train_on]
    n_groups = len(np.unique(inp.groups))
    if n_groups < 2:
        issues.append(Issue("warning", "single_group", None, f"only one {ev.holdout} in the data: calibrated "
                            "fusion and prior-only baselines need at least two to hold one out"))

    methods: dict[str, dict[str, Any]] = {}
    preds: dict[str, np.ndarray] = {}
    probs: dict[str, np.ndarray] = {}
    # each model alone (no fitting)
    for name in models:
        S = score_matrix(inp, table, name)
        ok = np.all(np.isfinite(S), axis=1)
        pr = np.where(ok, np.nanargmax(np.where(np.isfinite(S), S, -np.inf), axis=1), -1)
        const = ok & (np.ptp(np.where(ok[:, None], S, 0), axis=1) == 0)
        pr[const] = -1
        preds[f"{name} only"] = pr
        Z = np.zeros_like(S)
        for t in np.where(ok)[0]:
            Z[t] = normalize(S[t], norms[name])
        probs[f"{name} only"] = softmax(Z, axis=1)
    # fused with the config weights (no fitting -> no leakage)
    X = norm_tensor(inp, table, models, norms)
    w_cfg = np.array([cfg_weights.get(m, 0.0) for m in models])
    preds["fused (config weights)"], probs["fused (config weights)"] = predict_from_tensor(X, w_cfg, cfg.fusion.temperature)
    folds: list[dict[str, Any]] = []
    if n_groups >= 2:
        pc, Pc, folds = logo_calibrated(inp, table, models, norms, baseline, y_tr, cfg.fusion.min_gain_logloss)
        preds["fused (calibrated, held-out)"], probs["fused (calibrated, held-out)"] = pc, Pc
        preds["prior-only baseline"] = prior_only(inp, y_tr)

    for gt in GROUND_TRUTHS:
        y = inp.labels[gt]
        for meth, pr in preds.items():
            r = _acc(pr, y)
            r["shuffled_label"] = shuffled_label_test(pr, y, inp.groups, ev.n_shuffles, rng)
            if meth in probs:
                P = probs[meth]
                m = y >= 0
                conf = P.max(axis=1)
                r["mean_confidence"] = float(conf[m].mean())
                unsure = conf < cfg.fusion.unsure_threshold
                r["unsure_rate"] = float(unsure[m].mean())
                sure = m & ~unsure
                r["accuracy_when_sure"] = float((pr[sure] == y[sure]).mean()) if sure.any() else float("nan")
            methods.setdefault(meth, {})[gt] = r

    # noise-score control: the whole pipeline (incl. calibration) on Gaussian noise
    noise_acc = []
    if ev.n_noise > 0:
        for _ in range(ev.n_noise):
            ntab = {mn: {m.trial_id: rng.normal(size=inp.n_options) for m in inp.metas} for mn in models}
            if n_groups >= 2:
                pn, _, _ = logo_calibrated(inp, ntab, models, norms, baseline, y_tr, cfg.fusion.min_gain_logloss)
            else:
                pn, _ = predict_from_tensor(norm_tensor(inp, ntab, models, norms), w_cfg)
            noise_acc.append(_acc(pn, inp.labels[train_on])["accuracy"])
    controls = {
        "chance": chance,
        "noise_score": {"mean_accuracy": float(np.mean(noise_acc)) if noise_acc else None,
                        "95pct": float(np.quantile(noise_acc, 0.95)) if noise_acc else None,
                        "n_runs": len(noise_acc), "ground_truth": train_on},
        "prior_only": methods.get("prior-only baseline"),
    }

    # model information checks
    for name in models:
        r = methods[f"{name} only"][train_on]
        sl = r["shuffled_label"]
        if r["n"] == 0:
            continue
        se = math.sqrt(chance * (1 - chance) / r["n"])
        if r["accuracy"] < chance - 2.5 * se:
            issues.append(Issue("error", "below_chance", name, f"accuracy {r['accuracy']:.2f} is well BELOW chance "
                                f"({chance:.2f}). The sign is probably flipped (set lower_is_better) or the option "
                                "order does not match the stimulus log."))
        elif sl["p_value"] > 0.05:
            issues.append(Issue("warning", "no_information", name, f"accuracy {r['accuracy']:.2f} is not "
                                f"distinguishable from chance ({chance:.2f}; shuffled-label p={sl['p_value']:.2f}). "
                                "This model is not contributing information."))
    if folds:
        for name in models:
            if name in baseline:
                continue
            kept = sum(name in f["selected"] for f in folds)
            issues.append(Issue("info", "calibration_selection", name,
                                f"kept by calibration in {kept}/{len(folds)} held-out folds "
                                f"({'adds value' if kept > len(folds) / 2 else 'stays at weight 0 by default'})"))
    # noise control sanity
    if noise_acc and controls["noise_score"]["mean_accuracy"] > chance + 0.1:
        issues.append(Issue("error", "noise_above_chance", None, "noise inputs score above chance: the evaluation leaks "
                            "label information. Do not trust the other numbers until this is fixed."))

    # accuracy vs repeats
    rep_rows = []
    pf_models = [m for m in ds.per_flash_models() if m in models]
    R = ds.max_repeats
    if pf_models and R > 0:
        cal_w = {f["held_out"]: np.array([f["weights"][mn] for mn in models]) for f in folds}
        for k in range(1, R + 1):
            tk = ds.score_table(max_repeat=k)
            Xk = norm_tensor(inp, tk, models, norms)
            row = {"repeats": k}
            for name in pf_models:
                S = score_matrix(inp, tk, name)
                ok = np.all(np.isfinite(S), axis=1)
                pr = np.where(ok, np.argmax(np.where(np.isfinite(S), S, -np.inf), axis=1), -1)
                for gt in GROUND_TRUTHS:
                    row[f"{name} only [{gt}]"] = _acc(pr, inp.labels[gt])["accuracy"]
            pr, _ = predict_from_tensor(Xk, w_cfg)
            for gt in GROUND_TRUTHS:
                row[f"fused config [{gt}]"] = _acc(pr, inp.labels[gt])["accuracy"]
            if cal_w:
                prc = np.full(len(inp.metas), -1)
                for g, w in cal_w.items():
                    te = inp.groups == g
                    prc[te], _ = predict_from_tensor(Xk[te], w)
                for gt in GROUND_TRUTHS:
                    row[f"fused calibrated [{gt}]"] = _acc(prc, inp.labels[gt])["accuracy"]
            rep_rows.append(row)
    repeats = pd.DataFrame(rep_rows)
    seq = simulate_early_stopping(ds, inp, pf_models[0], y_tr, cfg) if pf_models and R > 1 else None

    # reliability + confusion for the main fused method
    main = "fused (calibrated, held-out)" if "fused (calibrated, held-out)" in probs else "fused (config weights)"
    y = inp.labels[train_on]
    m = y >= 0
    conf = probs[main].max(axis=1)
    correct = (preds[main] == y).astype(float)
    ece, rel = expected_calibration_error(conf[m], correct[m], ev.calibration_bins)
    opts = inp.metas[0].option_ids
    cm = np.zeros((inp.n_options, inp.n_options + 1), dtype=int)
    for p, t in zip(preds[main][m], y[m]):
        cm[t, p if p >= 0 else inp.n_options] += 1
    confusion = pd.DataFrame(cm, index=[f"true {o}" for o in opts], columns=[f"pred {o}" for o in opts] + ["no pred"])

    pred_df = pd.DataFrame({
        "trial_id": [mm.trial_id for mm in inp.metas],
        "group": inp.groups,
        "target_option": [mm.target_option for mm in inp.metas],
        "clicked_option": [mm.clicked_option for mm in inp.metas],
        **{f"pred[{k}]": [opts[i] if i >= 0 else None for i in v] for k, v in preds.items()},
        f"confidence[{main}]": conf,
    })

    metrics = {
        "n_trials_total": len(ds.metas), "n_invalid_skipped": inp.n_invalid, "n_unlabeled_skipped": inp.n_unlabeled,
        "n_evaluated": len(inp.metas), "n_options": inp.n_options, "chance": chance,
        "holdout": ev.holdout, "n_groups": n_groups, "trained_on": train_on,
        "models": models, "config_weights": cfg_weights, "normalize": norms,
        "methods": methods, "controls": controls, "calibration_folds": folds,
        "main_method": main, "ece": ece, "early_stopping": seq,
        "unsure_threshold": cfg.fusion.unsure_threshold,
    }
    res = EvalResult(metrics, pred_df, issues, rel, confusion, repeats)
    if out_dir is not None:
        write_outputs(res, Path(out_dir), make_plots)
    return res


def simulate_early_stopping(ds: Dataset, inp: EvalInputs, model: str, y_train: np.ndarray, cfg: Config) -> dict[str, Any]:
    """Accumulate per-repeat evidence (sum of per-option means x repeats) and stop when confident.

    The evidence scale is fitted on the other groups (held out), like the calibration.
    """
    R = ds.max_repeats
    sums = np.zeros((R, len(inp.metas), inp.n_options))
    for k in range(1, R + 1):
        S = score_matrix(inp, ds.score_table(max_repeat=k), model)
        sums[k - 1] = np.nan_to_num(S - np.nanmean(S, axis=1, keepdims=True)) * k
    sc = cfg.sequential
    stop_k = np.full(len(inp.metas), R)
    pred = np.full(len(inp.metas), -1)
    conf_at = np.zeros(len(inp.metas))
    for g in np.unique(inp.groups):
        te = inp.groups == g
        tr = (~te) & (y_train >= 0)
        if tr.sum() >= 5:
            beta, _ = fit_conditional_logit(sums[-1][tr][:, :, None], y_train[tr], l2=1e-3)
            scale = float(max(beta[0], 1e-3))
        else:
            scale = 1.0
        for t in np.where(te)[0]:
            for k in range(1, R + 1):
                P = softmax(scale * sums[k - 1, t])
                if (k >= sc.min_repeats and P.max() >= sc.stop_confidence) or k == R or \
                        (sc.max_repeats is not None and k >= sc.max_repeats):
                    stop_k[t], pred[t], conf_at[t] = k, int(np.argmax(P)), float(P.max())
                    break
    out = {"model": model, "stop_confidence": sc.stop_confidence, "min_repeats": sc.min_repeats,
           "mean_repeats_used": float(stop_k.mean()), "max_repeats": R,
           "fraction_stopped_early": float((stop_k < R).mean())}
    for gt in GROUND_TRUTHS:
        out[f"accuracy [{gt}]"] = _acc(pred, inp.labels[gt])["accuracy"]
    return out


# --------------------------------------------------------------------------- outputs

def write_outputs(res: EvalResult, out: Path, make_plots: bool = True) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(_jsonable(res.metrics), indent=1))
    res.predictions.to_csv(out / "predictions_heldout.csv", index=False)
    res.reliability.to_csv(out / "reliability.csv", index=False)
    res.confusion.to_csv(out / "confusion.csv")
    if len(res.repeats):
        res.repeats.to_csv(out / "accuracy_vs_repeats.csv", index=False)
    (out / "issues.txt").write_text("\n".join(str(i) for i in res.issues) + "\n")
    if make_plots:
        res.figures = make_figures(res, out)
    from .report import render_report
    (out / "report.md").write_text(render_report(res))


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


def make_figures(res: EvalResult, out: Path) -> dict[str, str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = {}
    m = res.metrics
    gt = m["trained_on"]
    # 1. accuracy by method with chance and controls
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = list(m["methods"])
    accs = [m["methods"][n][gt]["accuracy"] for n in names]
    lo = [a - m["methods"][n][gt]["ci95"][0] for a, n in zip(accs, names)]
    hi = [m["methods"][n][gt]["ci95"][1] - a for a, n in zip(accs, names)]
    ax.barh(names, accs, xerr=[lo, hi], color="#4C72B0", alpha=0.85, capsize=3)
    ax.axvline(m["chance"], color="k", ls="--", lw=1, label="chance")
    nm = m["controls"]["noise_score"]["mean_accuracy"]
    if nm is not None:
        ax.axvline(nm, color="#C44E52", ls=":", lw=1.5, label="noise-score control")
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"accuracy vs {gt} answer (95% CI)")
    ax.invert_yaxis()
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "accuracy_by_method.png", dpi=130)
    plt.close(fig)
    figs["accuracy"] = "accuracy_by_method.png"
    # 2. accuracy vs repeats
    if len(res.repeats):
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for c in res.repeats.columns:
            if c != "repeats" and c.endswith(f"[{gt}]"):
                ax.plot(res.repeats["repeats"], res.repeats[c], marker="o", ms=3, label=c.replace(f" [{gt}]", ""))
        ax.axhline(m["chance"], color="k", ls="--", lw=1, label="chance")
        ax.set_xlabel("repeats per option")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "accuracy_vs_repeats.png", dpi=130)
        plt.close(fig)
        figs["repeats"] = "accuracy_vs_repeats.png"
    # 3. reliability
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    if len(res.reliability):
        ax.plot(res.reliability["mean_confidence"], res.reliability["accuracy"], "o-", color="#4C72B0")
        for _, r in res.reliability.iterrows():
            ax.annotate(str(int(r["n"])), (r["mean_confidence"], r["accuracy"]), fontsize=7,
                        xytext=(3, -8), textcoords="offset points")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title(f"{m['main_method']}\nECE = {m['ece']:.3f}", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=130)
    plt.close(fig)
    figs["reliability"] = "reliability.png"
    # 4. confusion
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    cmv = res.confusion.to_numpy()
    ax.imshow(cmv, cmap="Blues")
    ax.set_xticks(range(cmv.shape[1]), res.confusion.columns, fontsize=8)
    ax.set_yticks(range(cmv.shape[0]), res.confusion.index, fontsize=8)
    for i in range(cmv.shape[0]):
        for j in range(cmv.shape[1]):
            ax.text(j, i, cmv[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cmv[i, j] > cmv.max() / 2 else "black")
    ax.set_title(f"confusion ({m['main_method']})", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=130)
    plt.close(fig)
    figs["confusion"] = "confusion.png"
    return figs

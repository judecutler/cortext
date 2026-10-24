"""Render the one-page evaluation report (Markdown) from an :class:`EvalResult`.

The blank template for hand-filling lives in ``docs/evaluation_report_template.md``;
this module fills the same sections automatically.
"""
from __future__ import annotations

from datetime import date
from typing import Any


def _pct(x: Any) -> str:
    return "n/a" if x is None or x != x else f"{100 * x:.1f}%"


def _p(x: Any) -> str:
    return "n/a" if x is None else (f"{x:.3f}" if x >= 0.001 else "<0.001")


def render_report(res) -> str:
    m = res.metrics
    gt = m["trained_on"]
    other = "target" if gt == "clicked" else "clicked"
    lines = [
        "# CorText fusion: evaluation report",
        "",
        f"*Generated {date.today().isoformat()} by `cortext-fusion evaluate`. Fill in the bracketed items by hand.*",
        "",
        "## 1. Setup",
        f"- Data: **{m['n_evaluated']}** trials evaluated ({m['n_invalid_skipped']} flagged invalid and skipped, "
        f"{m['n_unlabeled_skipped']} without a label), {m['n_options']} options, chance = {_pct(m['chance'])}.",
        f"- Held out by **{m['holdout']}** ({m['n_groups']} groups, leave-one-out). Never random trials.",
        f"- Models: " + ", ".join(f"`{x}` (config weight {m['config_weights'][x]}, {m['normalize'][x]})" for x in m["models"]),
        f"- Calibration trained on the **{gt}** answer. Device / montage / participants: [fill in]",
        "- Data source: [real recording / public dataset / MOCK]. **If MOCK, these numbers mean nothing about the brain.**",
        "",
        "## 2. Accuracy",
        f"| Method | Acc. vs {gt} (95% CI) | Acc. vs {other} | Shuffled-label p | Unsure rate | Acc. when sure |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in m["methods"].items():
        a, b = r[gt], r[other]
        lines.append(f"| {name} | {_pct(a['accuracy'])} ({_pct(a['ci95'][0])}–{_pct(a['ci95'][1])}), n={a['n']} | "
                     f"{_pct(b['accuracy'])} | {_p(a['shuffled_label']['p_value'])} | {_pct(a.get('unsure_rate'))} | "
                     f"{_pct(a.get('accuracy_when_sure'))} |")
    c = m["controls"]
    ns = c["noise_score"]
    lines += [
        "",
        "## 3. Controls (all must be near chance)",
        f"- Chance: {_pct(c['chance'])}",
        f"- Noise-score control (whole pipeline on Gaussian noise, {ns['n_runs']} runs): mean "
        f"{_pct(ns['mean_accuracy'])}, 95th pct {_pct(ns['95pct'])}",
        f"- Shuffled-label control: see the p-values above (null mean for the main method: "
        f"{_pct(m['methods'][m['main_method']][gt]['shuffled_label']['null_mean'])})",
        f"- Prior-only baseline (most common answer position in training groups): "
        f"{_pct(c['prior_only'][gt]['accuracy']) if c.get('prior_only') else 'n/a (needs 2+ groups)'}",
        "- Low-level feature baseline (luminance/position/timing only): [fill in]",
        "",
        "## 4. Accuracy vs repeats and early stopping",
    ]
    if len(res.repeats):
        cols = [col for col in res.repeats.columns if col.endswith(f"[{gt}]")]
        lines.append("| Repeats | " + " | ".join(col.replace(f" [{gt}]", "") for col in cols) + " |")
        lines.append("|---" * (len(cols) + 1) + "|")
        for _, r in res.repeats.iterrows():
            lines.append(f"| {int(r['repeats'])} | " + " | ".join(_pct(r[col]) for col in cols) + " |")
    else:
        lines.append("No per-flash model output, so accuracy vs repeats is not available.")
    es = m.get("early_stopping")
    if es:
        lines += ["", f"Early stopping ({es['model']}, stop at confidence ≥ {es['stop_confidence']}, min "
                  f"{es['min_repeats']} repeats): mean **{es['mean_repeats_used']:.1f}/{es['max_repeats']}** repeats, "
                  f"{_pct(es['fraction_stopped_early'])} of trials stopped early, accuracy {_pct(es[f'accuracy [{gt}]'])}."]
    lines += [
        "",
        "## 5. Calibration and errors",
        f"- Main method: **{m['main_method']}**, expected calibration error **{m['ece']:.3f}** "
        f"(0 = perfectly calibrated). Unsure threshold: {m['unsure_threshold']}.",
    ]
    if m["calibration_folds"]:
        lines.append("- Learned weights per held-out fold: " + "; ".join(
            f"{f['held_out']}: " + ", ".join(f"{k}={v:.2f}" for k, v in f["weights"].items())
            for f in m["calibration_folds"][:6]) + (" ..." if len(m["calibration_folds"]) > 6 else ""))
    lines += ["", "Confusion matrix by on-screen position (rows = true position). A skew toward one position suggests a gaze/position confound:", "", res.confusion.to_markdown()
              if _has_tabulate() else "```\n" + res.confusion.to_string() + "\n```", ""]
    if res.figures:
        lines += ["Figures: " + ", ".join(f"![{k}]({v})" for k, v in res.figures.items()), ""]
    lines += ["## 6. Diagnostics"]
    shown = [i for i in res.issues if i.level in ("error", "warning")]
    lines += [f"- {i}" for i in shown] or ["- No errors or warnings."]
    infos = [i for i in res.issues if i.level == "info"]
    lines += [f"- {i}" for i in infos]
    lines += [
        "",
        "## 7. Verdict (fill in)",
        "- Does fused beat P300 only on held-out data, beyond its CI? [yes/no]",
        "- Do all controls sit at chance? [yes/no]",
        "- Does the category model add value (kept by calibration in most folds)? [yes/no]",
        "- Go / no-go for the next stage, and why: [ ]",
        "",
    ]
    return "\n".join(lines)


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False

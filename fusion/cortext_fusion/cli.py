"""Command line: ``cortext-fusion <command>`` (or ``python -m cortext_fusion <command>``).

describe   inspect a model output file and suggest an adapter
mock       write a fake stimulus log + model outputs in every format + configs
evaluate   held-out evaluation with controls; writes report/ (report.md, plots, metrics.json)
calibrate  learn fusion weights on labeled validation trials -> calibration.yaml
fuse       fuse every trial in the stimulus log -> predictions.csv
serve      live servers: WebSocket for the CorText Lab app + HTTP for the models (--mock to fake scores)
watch      fuse per-trial output files as they appear in folders
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path


def _cmd_describe(a) -> int:
    from .describe import describe
    from .stimulus_log import load_stimulus_log
    metas = load_stimulus_log(a.markers, a.trials) if a.markers else None  # a folder/.zip = CorText Lab export
    cn = [c.strip() for c in a.class_names.split(",")] if a.class_names else None
    if a.class_names_file:
        cn = [l.strip() for l in Path(a.class_names_file).read_text().splitlines() if l.strip()]
    print(describe(a.file, metas, cn, a.n_options, a.name))
    return 0


def _cmd_mock(a) -> int:
    from .mock import MockSpec, generate
    spec = MockSpec(n_participants=a.participants, sessions_per_participant=a.sessions, trials_per_session=a.trials,
                    n_repeats=a.repeats, p300_signal=a.p300_signal, category_signal=a.category_signal, seed=a.seed)
    r = generate(a.out, spec)
    print(f"Wrote {r['n_trials']} mock trials to {r['out_dir']}")
    print("Outputs:")
    for k, v in r["outputs"].items():
        print(f"  {k:28s} {v}")
    print("Configs:", ", ".join(sorted(set(r["configs"].values()))))
    print(f"Try:  cortext-fusion evaluate {Path(a.out) / 'mock_config.yaml'}")
    return 0


def _load(a):
    from .config import load_config
    from .pipeline import load_dataset
    cfg = load_config(a.config)
    ds = load_dataset(cfg)
    return cfg, ds


def _print_issues(issues, quiet_info=False) -> None:
    for i in issues:
        if quiet_info and i.level == "info":
            continue
        print(" ", i)


def _cmd_evaluate(a) -> int:
    from .evaluate import evaluate
    cfg, ds = _load(a)
    out = Path(a.out) if a.out else cfg.resolve(cfg.evaluation.output_dir)
    res = evaluate(ds, cfg, train_on=a.train_on, out_dir=out, make_plots=not a.no_plots)
    m = res.metrics
    print(f"Evaluated {m['n_evaluated']} trials ({m['n_invalid_skipped']} invalid skipped), held out by "
          f"{m['holdout']} ({m['n_groups']} groups). Chance = {m['chance']:.2f}")
    print(f"{'method':32s} {'vs ' + a.train_on:>12s} {'vs ' + ('target' if a.train_on == 'clicked' else 'clicked'):>12s}  shuffled p")
    other = "target" if a.train_on == "clicked" else "clicked"
    for name, r in m["methods"].items():
        print(f"{name:32s} {r[a.train_on]['accuracy']:12.3f} {r[other]['accuracy']:12.3f}  {r[a.train_on]['shuffled_label']['p_value']:.3f}")
    ns = m["controls"]["noise_score"]
    if ns["mean_accuracy"] is not None:
        print(f"{'noise-score control':32s} {ns['mean_accuracy']:12.3f}")
    if res.issues:
        print("Diagnostics:")
        _print_issues(res.issues)
    print(f"Report: {out / 'report.md'}")
    return 1 if any(i.level == "error" for i in res.issues) and a.strict else 0


def _cmd_calibrate(a) -> int:
    from .fusion import calibrate
    cfg, ds = _load(a)
    metas = [m for m in ds.metas.values() if m.valid]
    if a.sessions:
        keep = set(a.sessions.split(","))
        metas = [m for m in metas if m.session_id in keep]
    if a.exclude_sessions:
        drop = set(a.exclude_sessions.split(","))
        metas = [m for m in metas if m.session_id not in drop]
    labels = {}
    for m in metas:
        i = m.clicked_index() if a.train_on == "clicked" else m.target_index()
        if i is not None:
            labels[m.trial_id] = i
    models = [m.name for m in cfg.enabled_models if m.name in ds.models]
    norms = {m.name: m.normalize or cfg.fusion.normalize for m in cfg.enabled_models}
    baseline = [m.name for m in cfg.enabled_models if m.weight != 0 and m.name in ds.models] or models[:1]
    groups = {m.trial_id: (m.session_id if cfg.evaluation.holdout == "session" else m.participant_id) or "all" for m in metas}
    cal = calibrate(metas, ds.score_table(), models, labels, norms, baseline, groups, cfg.fusion.min_gain_logloss)
    out = Path(a.out) if a.out else (cfg.resolve(cfg.fusion.calibration_file) or Path("calibration.yaml"))
    cal.save(out)
    print(f"Calibrated on {cal.n_trials} trials. Weights: " + ", ".join(f"{k}={v:.3f}" for k, v in cal.weights.items()))
    for name, c in cal.selection.get("candidates", {}).items():
        print(f"  {name}: {'KEPT' if c['kept'] else 'weight 0'}"
              + (f" (held-out log-loss gain {c['gain_nats_per_trial']:+.3f} nats/trial)" if "gain_nats_per_trial" in c
                 else f" ({c.get('reason')})"))
    print(f"Wrote {out}. Calibrate on validation sessions only; evaluate on different sessions.")
    return 0


def _cmd_fuse(a) -> int:
    from .fusion import Fuser
    from .pipeline import fuse_dataset
    cfg, ds = _load(a)
    fuser = Fuser.from_config(cfg)
    preds = fuse_dataset(ds, fuser)
    out = Path(a.out)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["trial_id", "predicted_option", "predicted_category", "confidence", "margin", "unsure",
                    "probabilities", "contributions", "warnings", "message"])
        for p in preds.values():
            w.writerow([p.trial_id, p.predicted_option, p.predicted_category, f"{p.confidence:.4f}", f"{p.margin:.4f}",
                        p.unsure, json.dumps([round(x, 4) for x in p.probabilities]),
                        json.dumps({k: round(v, 4) for k, v in p.contributions.items()}), "; ".join(p.warnings),
                        p.message()])
    _print_issues(ds.issues, quiet_info=True)
    print(f"Wrote {len(preds)} predictions to {out} (weights: {fuser.weights}, from {fuser.source})")
    return 0


def _cmd_serve(a) -> int:
    import threading
    from .server import Predictor, serve, serve_ws
    if a.config:
        from .config import load_config
        pred = Predictor.from_config(load_config(a.config), mock=a.mock, mock_accuracy=a.mock_accuracy)
    elif a.mock:
        pred = Predictor.mock_only(mock_accuracy=a.mock_accuracy)
    else:
        print("serve needs a config, or --mock", file=sys.stderr)
        return 2
    servers = [serve(pred, a.host, a.http_port, verbose=a.verbose)]
    if not a.no_ws:
        servers.append(serve_ws(pred, a.host, a.ws_port, verbose=a.verbose))
    mode = "MOCK predictions (random: the app sends no labels)" if pred.mock else "real predictions"
    print(f"cortext-fusion: {mode}; weights {pred.fuser.weights}")
    if not a.no_ws:
        print(f"  CorText Lab app  -> ws://{a.host}:{a.ws_port}   (set the app's predictor to 'ws' with this URL)")
    print(f"  models / scripts -> http://{a.host}:{a.http_port}  (POST /sequential/flash, /predict, ...)")
    print("Ctrl-C to stop.")
    threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers]
    for t in threads:
        t.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for s in servers:
            s.shutdown()
            s.server_close()
    return 0


def _cmd_watch(a) -> int:
    from .config import load_config
    from .io import load_single_trial_file, trial_id_from_name, PER_TRIAL_SUFFIXES
    from .server import Predictor
    from .pipeline import match_trial_ids
    cfg = load_config(a.config)
    pred = Predictor.from_config(cfg)
    folders = {}
    for spec in a.folder:
        name, _, path = spec.partition("=")
        if not path:
            print(f"--folder must be model=path, got {spec!r}", file=sys.stderr)
            return 2
        folders[name] = Path(path)
    required = set(a.require.split(",")) if a.require else {m.name for m in cfg.enabled_models if m.weight != 0} & set(folders)
    pending: dict[str, dict[str, object]] = {}
    seen: set[Path] = set()
    done: set[str] = set()
    out = open(a.out, "a") if a.out else None
    print(f"Watching {', '.join(f'{k}={v}' for k, v in folders.items())}; fusing when {sorted(required)} are present.")
    try:
        while True:
            metas = cfg.load_metas()
            for name, folder in folders.items():
                if not folder.exists():
                    continue
                for p in sorted(folder.iterdir()):
                    if p in seen or p.suffix.lower() not in PER_TRIAL_SUFFIXES:
                        continue
                    try:
                        raw = load_single_trial_file(p)
                    except Exception:
                        continue  # still being written
                    seen.add(p)
                    pending.setdefault(trial_id_from_name(p, cfg.model(name).file_pattern), {})[name] = raw
            remapped, _ = match_trial_ids(pending, metas)
            for tid, outs in list(remapped.items()):
                if tid in done or tid not in metas or not required <= set(outs):
                    continue
                r = pred.predict(metas[tid], outs)
                line = json.dumps({k: r[k] for k in ("trial_id", "predicted_option", "confidence", "unsure", "message",
                                                     "warnings")})
                print(line, flush=True)
                if out:
                    out.write(line + "\n")
                    out.flush()
                done.add(tid)
            if a.once:
                break
            time.sleep(a.poll)
    except KeyboardInterrupt:
        pass
    finally:
        if out:
            out.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cortext-fusion", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("describe", help="inspect a model output file and suggest an adapter")
    d.add_argument("file")
    d.add_argument("--markers", help="CorText Lab export (folder or .zip) or a marker log CSV; improves detection and checks trial ids")
    d.add_argument("--trials", help="stimulus trial table")
    d.add_argument("--class-names", help="comma-separated class names, in model output order")
    d.add_argument("--class-names-file")
    d.add_argument("--n-options", type=int, default=4)
    d.add_argument("--name", default="my_model", help="model name for the suggested config entry")
    d.set_defaults(func=_cmd_describe)

    m = sub.add_parser("mock", help="generate mock data, outputs in every format, and configs")
    m.add_argument("out")
    m.add_argument("--participants", type=int, default=3)
    m.add_argument("--sessions", type=int, default=2, help="sessions per participant")
    m.add_argument("--trials", type=int, default=40, help="trials per session")
    m.add_argument("--repeats", type=int, default=10)
    m.add_argument("--p300-signal", type=float, default=0.6, help="per-flash effect size; 0 = no signal")
    m.add_argument("--category-signal", type=float, default=1.0, help="0 = no signal")
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(func=_cmd_mock)

    e = sub.add_parser("evaluate", help="held-out evaluation with controls -> report")
    e.add_argument("config")
    e.add_argument("--out", help="output folder (default: evaluation.output_dir)")
    e.add_argument("--train-on", choices=["clicked", "target"], default="clicked",
                   help="which answer calibration learns from (both are reported)")
    e.add_argument("--no-plots", action="store_true")
    e.add_argument("--strict", action="store_true", help="exit 1 if any diagnostic is an error")
    e.set_defaults(func=_cmd_evaluate)

    c = sub.add_parser("calibrate", help="learn fusion weights on validation trials")
    c.add_argument("config")
    c.add_argument("--out")
    c.add_argument("--train-on", choices=["clicked", "target"], default="clicked")
    c.add_argument("--sessions", help="only these session ids (comma-separated)")
    c.add_argument("--exclude-sessions", help="leave these session ids out (keep them for testing)")
    c.set_defaults(func=_cmd_calibrate)

    f = sub.add_parser("fuse", help="fuse all trials -> predictions.csv")
    f.add_argument("config")
    f.add_argument("--out", default="predictions.csv")
    f.set_defaults(func=_cmd_fuse)

    s = sub.add_parser("serve", help="local HTTP server for the demo app")
    s.add_argument("config", nargs="?")
    s.add_argument("--mock", action="store_true", help="fake predictions (no models needed)")
    s.add_argument("--mock-accuracy", type=float, default=0.8)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--ws-port", type=int, default=8765, help="WebSocket for the CorText Lab app (its default URL)")
    s.add_argument("--http-port", type=int, default=8766, help="HTTP for models and scripts")
    s.add_argument("--no-ws", action="store_true", help="HTTP only")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=_cmd_serve)

    w = sub.add_parser("watch", help="fuse per-trial output files as they appear")
    w.add_argument("config")
    w.add_argument("--folder", action="append", required=True, help="model=path (repeat for each model)")
    w.add_argument("--require", help="models that must be present before fusing (default: non-zero weight)")
    w.add_argument("--out", help="append JSON lines here")
    w.add_argument("--poll", type=float, default=0.25)
    w.add_argument("--once", action="store_true", help="process what is there and exit")
    w.set_defaults(func=_cmd_watch)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())

import json

import numpy as np
import pandas as pd
import pytest

from cortext_fusion.config import config_from_dict, load_config
from cortext_fusion.evaluate import evaluate, wilson_ci
from cortext_fusion.fusion import Fuser
from cortext_fusion.io import load_any, watch_folder
from cortext_fusion.pipeline import fuse_dataset, load_dataset
from cortext_fusion.stimulus_log import load_lab_export, load_stimulus_log

MOCK_CONFIGS = ["mock_config.yaml", "config_p300_epoch_prob_csv.yaml", "config_p300_predict_proba_npy.yaml",
                "config_p300_grid_npy.yaml", "config_p300_peak_amplitude_json.yaml", "config_p300_probs_json.yaml",
                "config_p300_distance_json.yaml", "config_p300_per_trial_folder.yaml",
                "config_category_class_probs_npy.yaml", "config_category_embedding_npy.yaml"]


def test_lab_export_loading(mock_dir):
    metas = load_lab_export(mock_dir / "lab_export")
    run = [m for m in metas.values() if m.flashes]
    m = run[0]
    assert ":" in m.trial_id and m.local_trial_id == m.trial_id.split(":")[1]
    assert len(m.option_ids) == 4 and m.n_repeats == 6 and len(m.flashes) == 24  # flash_off etc. ignored
    assert m.target_category == m.category_of(m.target_option) and m.participant_id.startswith("P")
    assert m.session_id.startswith(m.participant_id)
    assert any(not x.valid and x.invalid_reason in ("misclick", "blink_or_movement", "distracted",
                                                  "equipment_glitch") for x in run)
    aborted = [x for x in metas.values() if not x.flashes]
    assert aborted and all(not x.valid and x.invalid_reason == "aborted" for x in aborted)  # kept, never dropped
    assert len({x.session_id for x in metas.values()}) == 4


def test_lab_export_zip_and_single_session(mock_dir, tmp_path):
    import zipfile
    zpath = tmp_path / "cortext_all_sessions.zip"
    sessions = sorted((mock_dir / "lab_export").iterdir())
    with zipfile.ZipFile(zpath, "w") as z:
        for sd in sessions:
            for f in sd.iterdir():
                z.write(f, f"{sd.name}/{f.name}")
    assert len(load_lab_export(zpath)) == len(load_lab_export(mock_dir / "lab_export"))
    one = load_lab_export(sessions[0])  # a single session folder
    assert len({m.session_id for m in one.values()}) == 1


def test_trial_id_matching(mock_dir):
    from cortext_fusion.pipeline import match_trial_ids
    all_metas = load_lab_export(mock_dir / "lab_export")
    k = next(iter(all_metas))
    s, t = k.split(":")
    out, tr = match_trial_ids({f"{s}__{t}": 1}, all_metas)
    assert list(out) == [k] and tr
    # bare trial id is ambiguous across sessions -> left unmatched
    out, _ = match_trial_ids({t: 1}, all_metas)
    assert list(out) == [t]
    one = {kk: m for kk, m in all_metas.items() if m.session_id == s}
    out, _ = match_trial_ids({f"{float(t)}": 1}, one)
    assert list(out) == [k]


def test_stimulus_log_aliases_and_inferred_repeats():
    mdf = pd.DataFrame({"Trial": ["x"] * 8, "Option": list("ABCDBADC"), "Category": list("wxyzxwzy"),
                        "Target": [0, 1, 0, 0, 1, 0, 0, 0]})
    m = load_stimulus_log(mdf)["x"]
    assert m.target_option == "B" and m.option_categories == list("wxyz")
    assert [f.repeat_index for f in m.flashes] == [0, 0, 0, 0, 1, 1, 1, 1]


def test_generic_log_composite_keys_when_ids_repeat():
    mdf = pd.DataFrame({"session_id": ["s1"] * 4 + ["s2"] * 4, "trial_id": [0] * 8, "event": ["flash_on"] * 8,
                        "option_id": list("ABCD") * 2, "category": list("wxyz") * 2})
    metas = load_stimulus_log(mdf)
    assert sorted(metas) == ["s1:0", "s2:0"]
    assert sorted(load_stimulus_log(mdf, key_mode="session")) == ["s1:0", "s2:0"]


@pytest.mark.parametrize("cfg_name", MOCK_CONFIGS)
def test_every_mock_format_loads_and_beats_chance(mock_dir, cfg_name):
    cfg = load_config(mock_dir / cfg_name)
    ds = load_dataset(cfg)
    assert not [i for i in ds.issues if i.level == "error"], [str(i) for i in ds.issues]
    preds = fuse_dataset(ds, Fuser.from_config(cfg, calibration=None))
    valid = [m for m in ds.metas.values() if m.valid]
    acc = np.mean([preds[m.trial_id].predicted_option == m.clicked_option for m in valid])
    assert acc > 0.5, f"{cfg_name}: accuracy {acc}"


def test_evaluate_with_signal(mock_dir, tmp_path):
    cfg = load_config(mock_dir / "config_category_embedding_npy.yaml")
    cfg.evaluation.n_shuffles, cfg.evaluation.n_noise = 50, 5
    res = evaluate(load_dataset(cfg), cfg, out_dir=tmp_path / "rep")
    m = res.metrics
    p3 = m["methods"]["p300 only"]["clicked"]
    cal = m["methods"]["fused (calibrated, held-out)"]["clicked"]
    assert p3["accuracy"] > 0.6 and p3["shuffled_label"]["p_value"] < 0.05
    assert cal["accuracy"] >= p3["accuracy"] - 0.05
    assert abs(m["controls"]["noise_score"]["mean_accuracy"] - 0.25) < 0.1
    assert abs(m["controls"]["prior_only"]["clicked"]["accuracy"] - 0.25) < 0.15
    assert m["n_invalid_skipped"] > 0
    assert len(res.repeats) == 6 and res.repeats.iloc[-1, 1] >= res.repeats.iloc[0, 1]
    assert m["early_stopping"]["mean_repeats_used"] <= 6
    for f in ["report.md", "metrics.json", "predictions_heldout.csv", "accuracy_vs_repeats.png", "reliability.png",
              "confusion.png", "accuracy_by_method.png"]:
        assert (tmp_path / "rep" / f).exists(), f
    json.loads((tmp_path / "rep" / "metrics.json").read_text())
    assert "Controls" in (tmp_path / "rep" / "report.md").read_text()


def test_evaluate_null_data_is_at_chance_and_flagged(null_mock_dir):
    cfg = load_config(null_mock_dir / "mock_config.yaml")
    cfg.evaluation.n_shuffles, cfg.evaluation.n_noise = 50, 3
    res = evaluate(load_dataset(cfg), cfg, make_plots=False)
    for name, r in res.metrics["methods"].items():
        assert r["clicked"]["accuracy"] < 0.4, name
    codes = {(i.code, i.model) for i in res.issues}
    assert ("no_information", "p300") in codes and ("no_information", "category") in codes


def test_flipped_sign_is_flagged(mock_dir):
    cfg = load_config(mock_dir / "config_p300_distance_json.yaml")
    cfg.model("p300").adapter = "per_option"   # forget that it is a distance
    cfg.evaluation.n_shuffles, cfg.evaluation.n_noise = 20, 0
    res = evaluate(load_dataset(cfg), cfg, make_plots=False)
    assert any(i.code == "below_chance" and i.model == "p300" for i in res.issues)


def test_alignment_diagnostics(mock_dir, tmp_path):
    raw = load_any(mock_dir / "outputs" / "p300_peak_amplitude.json")
    ids = list(raw)
    bad = {k: v for k, v in raw.items() if k not in ids[:10]}
    bad["not_a_trial"] = [1, 2, 3, 4]
    bad[ids[20]] = [1, 1, 1, 1]                                # constant
    bad[ids[21]] = [float("nan"), 1, 2, 3]                     # NaN
    bad[ids[22]] = [1, 2, 3]                                   # wrong length
    cfg = config_from_dict({"stimulus_log": {"lab_export": str(mock_dir / "lab_export")},
                            "models": [{"name": "p300", "adapter": "per_option", "source": "x"}]})
    ds = load_dataset(cfg, sources={"p300": bad})
    codes = {i.code for i in ds.issues}
    assert {"unknown_trial_ids", "missing_trials", "constant_scores", "nan_scores", "adapter_error"} <= codes


def test_config_validation(tmp_path):
    with pytest.raises(ValueError, match="unknown key"):
        config_from_dict({"models": [{"name": "a", "adaptor": "per_option"}]})
    with pytest.raises(ValueError, match="never random"):
        config_from_dict({"models": [{"name": "a"}], "evaluation": {"holdout": "random"}})
    with pytest.raises(Exception, match="unknown adapter"):
        config_from_dict({"models": [{"name": "a", "adapter": "nope"}]})


def test_example_config_parses():
    from pathlib import Path
    cfg = load_config(Path(__file__).parents[1] / "configs" / "example.yaml")
    assert [m.weight for m in cfg.models] == [1.0, 0.0]


def test_folder_watcher(tmp_path):
    (tmp_path / "t1.json").write_text("[1, 2, 3, 4]")
    got = []
    gen = watch_folder(tmp_path, poll_seconds=0.01, settle_seconds=0.0, stop=lambda: len(got) >= 1)
    for tid, raw in gen:
        got.append((tid, raw))
    assert got == [("t1", [1, 2, 3, 4])]


def test_wilson():
    lo, hi = wilson_ci(50, 100)
    assert lo < 0.5 < hi and hi - lo < 0.21

import numpy as np
import pytest

from cortext_fusion.fusion import (CalibrationResult, Fuser, ModelWeight, SequentialFuser, calibrate,
                                   fit_conditional_logit, normalize)
from cortext_fusion.schema import Flash, TrialMeta


def test_normalize_methods():
    s = np.array([1.0, 2.0, 3.0, 6.0])
    z = normalize(s, "zscore")
    assert np.isclose(z.mean(), 0) and np.isclose(z.std(), 1)
    assert np.isclose(np.exp(normalize(s, "logsoftmax")).sum(), 1)
    assert np.isclose(normalize(s, "softmax").sum(), 1)
    assert np.allclose(normalize([2, 2, 2, 2], "zscore"), 0)
    with pytest.raises(ValueError):
        normalize(s, "bogus")


def test_fuser_default_p300_only(meta):
    f = Fuser([ModelWeight("p300", 1.0), ModelWeight("category", 0.0)], unsure_threshold=0.4)
    p = f.fuse(meta, {"p300": [0, 3, 0, 0], "category": [9, 0, 0, 0]})
    assert p.predicted_option == "B" and p.predicted_category == "dog"
    assert p.models_used == ["p300"] and p.contributions["category"] == 0
    assert p.message().startswith("EEG indicates B (dog), confidence")
    assert 0 < p.margin < 1 and np.isclose(sum(p.probabilities), 1)


def test_fuser_unsure_and_warnings(meta):
    f = Fuser([ModelWeight("p300", 1.0), ModelWeight("category", 0.5)], unsure_threshold=0.9)
    p = f.fuse(meta, {"p300": [0, 0.1, 0, 0], "category": [np.nan, 0, 0, 0]})
    assert p.unsure and "unsure" in p.message()
    assert any("NaN" in w for w in p.warnings)
    p2 = f.fuse(meta, {"p300": [1, 1, 1, 1]})
    assert p2.predicted_option is None and p2.unsure
    assert any("constant" in w for w in p2.warnings) and any("no output" in w for w in p2.warnings)


def _synthetic(n=300, w=(2.0, 0.8, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    metas, scores, labels, groups = [], {"a": {}, "b": {}, "c": {}}, {}, {}
    for t in range(n):
        tid = f"t{t}"
        y = int(rng.integers(4))
        m = TrialMeta(tid, list("ABCD"), ["x"] * 4)
        for k, name in enumerate("abc"):
            s = rng.gumbel(size=4) / 2 + rng.normal(size=4)
            s[y] += w[k]
            scores[name][tid] = s
        metas.append(m)
        labels[tid] = y
        groups[tid] = f"g{t % 4}"
    return metas, scores, labels, groups


def test_conditional_logit_recovers_ordering():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4, 2))
    beta_true = np.array([1.5, 0.3])
    u = X @ beta_true
    y = np.array([rng.choice(4, p=np.exp(r) / np.exp(r).sum()) for r in u])
    beta, nll = fit_conditional_logit(X, y, l2=1e-4)
    assert np.allclose(beta, beta_true, atol=0.15) and nll < np.log(4)


def test_calibrate_selects_useful_models_only():
    metas, scores, labels, groups = _synthetic()
    cal = calibrate(metas, scores, ["a", "b", "c"], labels, baseline_models=["a"], groups=groups)
    assert "b" in cal.selected_models and "c" not in cal.selected_models
    assert cal.weights["c"] == 0 and cal.weights["a"] > cal.weights["b"] > 0
    assert cal.selection["candidates"]["c"]["kept"] is False


def test_calibrate_single_group_keeps_baseline_only():
    metas, scores, labels, _ = _synthetic(n=100)
    cal = calibrate(metas, scores, ["a", "b"], labels, baseline_models=["a"], groups=None)
    assert cal.selected_models == ["a"] and cal.weights["b"] == 0
    assert "cannot test" in cal.selection["candidates"]["b"]["reason"]


def test_calibration_roundtrip(tmp_path):
    metas, scores, labels, groups = _synthetic(n=120)
    cal = calibrate(metas, scores, ["a", "b"], labels, groups=groups)
    cal.save(tmp_path / "cal.yaml")
    back = CalibrationResult.load(tmp_path / "cal.yaml")
    assert back.weights == pytest.approx(cal.weights)
    f = Fuser([ModelWeight("a", 1.0), ModelWeight("b", 0.0)])
    f.apply_calibration(back)
    assert f.weights == pytest.approx(cal.weights) and f.temperature == 1.0


def test_sequential_accumulates_and_stops_early():
    seq = SequentialFuser(input="prob", stop_confidence=0.9, min_repeats=2)
    seq.start("t", list("ABCD"), ["cat", "dog", "car", "tree"])
    stopped_at = None
    for r in range(10):
        for o in "ABCD":
            p = seq.update("t", o, 0.85 if o == "C" else 0.15)
            if p.stopped_early and stopped_at is None:
                stopped_at = r
    assert p.predicted_option == "C" and p.stopped_early and stopped_at is not None and stopped_at < 9
    assert p.repeats_used == stopped_at + 1  # evidence frozen once stopped


def test_sequential_min_repeats_and_trial_evidence():
    seq = SequentialFuser(input="score", stop_confidence=0.5, min_repeats=3)
    seq.start("t", list("ABCD"))
    for o in "ABCD":
        p = seq.update("t", o, 5.0 if o == "A" else 0.0)
    assert not p.stopped_early  # confident but only 1 repeat
    seq.add_trial_evidence("t", [0, 0, 0, 10], weight=5.0)
    assert seq.current("t").predicted_option == "D"


def test_sequential_replay_and_fit_scale(meta):
    vals = [0.8 if f.option_id == "B" else 0.2 for f in meta.flashes]
    p = SequentialFuser(stop_confidence=0.99, min_repeats=1).replay(meta, vals)
    assert p.predicted_option == "B"
    scale = SequentialFuser.fit_scale([meta] * 6, {meta.trial_id: vals}, {meta.trial_id: 1})
    assert scale > 0

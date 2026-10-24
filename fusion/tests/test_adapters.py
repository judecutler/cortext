import json

import numpy as np
import pytest

from cortext_fusion.adapters import AdapterError, detect_candidates, make_adapter


def scores(name, raw, meta, **kw):
    return np.array(make_adapter(name, **kw).to_scores(raw, meta))


# (a) per-option -------------------------------------------------------------
def test_per_option_list_and_dicts(meta):
    assert scores("per_option", [1, 2, 3, 4], meta).tolist() == [1, 2, 3, 4]
    assert scores("per_option", {"D": 4, "C": 3, "B": 2, "A": 1}, meta).tolist() == [1, 2, 3, 4]
    assert scores("per_option", {"tree": 4, "car": 3, "dog": 2, "cat": 1}, meta).tolist() == [1, 2, 3, 4]
    assert scores("per_option", {"score_A": 1, "score_B": 2, "score_C": 3, "score_D": 4}, meta,
                  key_prefix="score_").tolist() == [1, 2, 3, 4]


def test_per_option_reorders_by_explicit_ids(meta):
    ts = make_adapter("per_option").to_trial_scores({"option_id": ["D", "C", "B", "A"], "score": [4, 3, 2, 1]}, meta)
    assert ts.scores.tolist() == [1, 2, 3, 4]
    assert ts.extras["option_order_mismatch"] == ["D", "C", "B", "A"]


def test_per_option_errors_are_helpful(meta):
    with pytest.raises(AdapterError, match="trial t1.*expected 4 values.*per_flash"):
        scores("per_option", list(range(12)), meta)
    with pytest.raises(AdapterError, match="do not match stimulus log options"):
        scores("per_option", {"option_id": ["A", "B", "C", "E"], "score": [1, 2, 3, 4]}, meta)


def test_field_selection(meta):
    assert scores("per_option", {"scores": [1, 2, 3, 4], "meta": "x"}, meta, field="scores").tolist() == [1, 2, 3, 4]
    with pytest.raises(AdapterError, match="field 'nope' not found"):
        scores("per_option", {"scores": [1, 2, 3, 4]}, meta, field="nope")


# (b) per-flash --------------------------------------------------------------
def test_per_flash_1d_aligned_with_markers(meta):
    vals = [10 if f.option_id == "B" else 0 for f in meta.flashes]
    assert scores("per_flash", vals, meta).tolist() == [0, 10, 0, 0]


def test_per_flash_grid_and_truncation(meta):
    grid = np.array([[0, 1, 0, 0], [0, 3, 0, 0], [0, 5, 0, 0]], float)  # [repeats, options]
    ad = make_adapter("per_flash")
    assert ad.to_scores(grid, meta) == [0, 3, 0, 0]
    assert ad.to_scores(grid, meta, max_repeat=1) == [0, 1, 0, 0]
    assert ad.to_trial_scores(grid, meta, max_repeat=2).n_repeats_used == 2


def test_per_flash_long_dict_and_aggregates(meta):
    raw = {"option_id": ["A", "A", "B", "B", "C", "D"], "score": [1, 3, 2, 2, 0, 9]}
    assert scores("per_flash", raw, meta).tolist() == [2, 2, 0, 9]
    assert scores("per_flash", raw, meta, aggregate="sum").tolist() == [4, 4, 0, 9]


def test_per_flash_missing_option_errors(meta):
    with pytest.raises(AdapterError, match="no flash values for option"):
        scores("per_flash", {"option_id": ["A", "B", "C"], "score": [1, 2, 3]}, meta)


# (c) probabilities ------------------------------------------------------------
def test_probabilities_logit_safe_at_0_and_1(meta):
    s = scores("probabilities", [0.0, 1.0, 0.5, 0.25], meta)
    assert np.all(np.isfinite(s)) and s[1] > s[2] > s[3] > s[0]
    assert abs(s[2]) < 1e-12


def test_probabilities_rejects_logits(meta):
    with pytest.raises(AdapterError, match="expected probabilities"):
        scores("probabilities", [-2, 3, 0.5, 1], meta)


# (d) epoch target probability ---------------------------------------------------
def test_epoch_target_prob_mean_log_odds(meta):
    p = [0.9 if f.option_id == "C" else 0.1 for f in meta.flashes]
    s = scores("epoch_target_prob", p, meta)
    assert np.argmax(s) == 2 and np.isclose(s[2], np.log(9))


def test_epoch_target_prob_predict_proba_shape(meta):
    pt = np.array([0.8 if f.option_id == "D" else 0.2 for f in meta.flashes])
    s = scores("epoch_target_prob", np.stack([1 - pt, pt], 1), meta)
    assert np.argmax(s) == 3


# (e) class probabilities ----------------------------------------------------------
def test_class_probabilities_lookup_and_target_check(meta):
    names = ["tree", "dog", "bird", "cat", "car"]
    p = [0.1, 0.5, 0.1, 0.2, 0.1]
    ts = make_adapter("class_probabilities", class_names=names, check_target=True).to_trial_scores(p, meta)
    assert np.allclose(np.exp(ts.scores), [0.2, 0.5, 0.1, 0.1])  # cat, dog, car, tree
    assert ts.extras["top_is_target_category"] is True
    d = {"p_tree": 0.1, "p_dog": 0.5, "p_bird": 0.1, "p_cat": 0.2, "p_car": 0.1}
    assert np.allclose(scores("class_probabilities", d, meta, key_prefix="p_"), ts.scores)


def test_class_probabilities_missing_category(meta):
    with pytest.raises(AdapterError, match="not among the model's classes"):
        scores("class_probabilities", [0.5, 0.5], meta, class_names=["dog", "cat"])
    s = scores("class_probabilities", [0.7, 0.3], meta, class_names=["dog", "kitty"], category_map={"cat": "kitty"},
               missing="floor")
    assert np.argmax(s) == 1


# (f) embeddings -------------------------------------------------------------------
def test_embedding_cosine(meta, tmp_path):
    emb = {"cat": [1, 0, 0], "dog": [0, 1, 0], "car": [0, 0, 1], "tree": [1, 1, 0]}
    f = tmp_path / "e.json"
    f.write_text(json.dumps(emb))
    s = scores("embedding", [0.1, 1.0, 0.0], meta, embeddings=str(f))
    assert np.argmax(s) == 1
    with pytest.raises(AdapterError, match="dim"):
        scores("embedding", [1, 2], meta, embeddings=emb)


# (g) lower is better --------------------------------------------------------------
def test_lower_is_better_and_distance_alias(meta):
    assert scores("per_option", [4, 1, 3, 2], meta, lower_is_better=True).tolist() == [-4, -1, -3, -2]
    assert np.argmax(scores("distance", [4, 1, 3, 2], meta)) == 1
    assert np.argmax(scores("distance", [4, 1, 3, 2], meta, lower_is_better=False)) == 1  # alias always negates


# auto-detection -------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ([0.3, -1.2, 2.0, 0.1], "per_option"),
    ([0.1, 0.6, 0.2, 0.1], "probabilities"),
    (list(np.random.default_rng(0).normal(size=12)), "per_flash"),
    (list(np.random.default_rng(0).uniform(size=12)), "epoch_target_prob"),
    (np.zeros((3, 4)) + np.arange(4), "per_flash"),
    ({"A": 1.5, "B": -1, "C": 2, "D": 0}, "per_option"),
    ({"option_id": list("ABCDABCD"), "score": list(range(8))}, "per_flash"),
    ({"embedding": [0.1] * 16}, "embedding"),
])
def test_detect(meta, raw, expected):
    assert detect_candidates(raw, meta)[0][0] == expected


def test_detect_class_probs(meta):
    p = np.full(10, 1 / 10)  # 12 would equal the fixture's flash count
    assert detect_candidates(p, meta)[0][0] == "class_probabilities"
    d = {c: 0.2 for c in ["cat", "dog", "car", "tree", "bird"]}
    assert detect_candidates(d, meta)[0][0] == "class_probabilities"


def test_auto_adapter_locks_in(meta):
    ad = make_adapter("auto")
    assert ad.to_scores([0.1, 0.6, 0.2, 0.1], meta)[1] > 0
    assert ad.chosen.name == "probabilities" and "probabilities" in ad.describe()["chosen"]


def test_unknown_adapter_and_option():
    with pytest.raises(AdapterError, match="unknown adapter"):
        make_adapter("magic")
    with pytest.raises(AdapterError, match="unknown option"):
        make_adapter("per_option", agregate="mean")

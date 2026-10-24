import numpy as np
import pytest

from cortext_fusion.mock import MockSpec, generate
from cortext_fusion.schema import Flash, TrialMeta


@pytest.fixture
def meta():
    """A 4-option trial with 3 repeats in a fixed flash order."""
    opts = ["A", "B", "C", "D"]
    order = [["B", "A", "D", "C"], ["A", "C", "B", "D"], ["D", "B", "C", "A"]]
    flashes = [Flash(o, r) for r, row in enumerate(order) for o in row]
    return TrialMeta("t1", opts, ["cat", "dog", "car", "tree"], target_category="dog", target_option="B",
                     clicked_option="B", session_id="S1", participant_id="P1", flashes=flashes)


@pytest.fixture(scope="session")
def mock_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("mock")
    spec = MockSpec(n_participants=2, sessions_per_participant=2, trials_per_session=30, n_repeats=6,
                    p300_signal=0.8, category_signal=1.5, seed=1)
    generate(d, spec)
    return d


@pytest.fixture(scope="session")
def null_mock_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("mock_null")
    spec = MockSpec(n_participants=2, sessions_per_participant=2, trials_per_session=30, n_repeats=6,
                    p300_signal=0.0, category_signal=0.0, seed=2)
    generate(d, spec, formats=["p300_per_flash_csv", "category_class_probs_csv"])
    return d

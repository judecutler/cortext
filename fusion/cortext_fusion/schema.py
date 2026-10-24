"""Core data structures shared across the package.

The *standard internal format* is a :class:`TrialScores`: for one trial and one
model, one real-valued score per option, in the stimulus log's option order,
where higher means "more likely to be the option the participant chose".
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np


@dataclass
class Flash:
    """One flash event from the stimulus app's marker log."""

    option_id: str
    repeat_index: int
    timestamp: float | None = None


@dataclass
class TrialMeta:
    """Everything the fusion layer knows about a trial from the stimulus log.

    ``option_ids`` defines the canonical option order for the trial. Every
    adapter must return scores in exactly this order.
    """

    trial_id: str
    option_ids: list[str]
    option_categories: list[str]
    target_category: str | None = None
    target_option: str | None = None      # the true answer (is_target in the marker log)
    clicked_option: str | None = None     # what the participant declared afterwards
    valid: bool = True
    invalid_reason: str | None = None
    session_id: str | None = None
    participant_id: str | None = None
    prompt: str | None = None
    flashes: list[Flash] = field(default_factory=list)  # in presentation order
    local_trial_id: str | None = None  # the app's own trial id (trial_id above may be "session:trial")

    @property
    def n_options(self) -> int:
        return len(self.option_ids)

    @property
    def n_repeats(self) -> int:
        if not self.flashes:
            return 0
        return max(f.repeat_index for f in self.flashes) + 1

    def option_index(self, option_id: str) -> int:
        try:
            return self.option_ids.index(str(option_id))
        except ValueError as e:
            raise KeyError(
                f"trial {self.trial_id}: option_id {option_id!r} is not one of {self.option_ids}"
            ) from e

    def category_of(self, option_id: str) -> str:
        return self.option_categories[self.option_index(option_id)]

    def target_index(self) -> int | None:
        return None if self.target_option is None else self.option_index(self.target_option)

    def clicked_index(self) -> int | None:
        return None if self.clicked_option is None else self.option_index(self.clicked_option)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrialMeta":
        d = dict(d)
        flashes = [f if isinstance(f, Flash) else Flash(**f) for f in d.pop("flashes", []) or []]
        d["option_ids"] = [str(o) for o in d["option_ids"]]
        d["option_categories"] = [str(c) for c in d.get("option_categories") or [""] * len(d["option_ids"])]
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(flashes=flashes, **known)


@dataclass
class TrialScores:
    """Standard internal format: one score per option for one model and trial."""

    trial_id: str
    model: str
    scores: np.ndarray                 # shape [n_options], higher = more likely correct
    option_ids: list[str]
    n_repeats_used: int | None = None  # how many repeats contributed (per-flash adapters)
    extras: dict[str, Any] = field(default_factory=dict)  # adapter diagnostics


@dataclass
class FusedPrediction:
    trial_id: str
    predicted_option: str | None
    predicted_index: int | None
    predicted_category: str | None
    confidence: float
    margin: float
    unsure: bool
    probabilities: list[float]
    fused_scores: list[float]
    option_ids: list[str]
    contributions: dict[str, float]            # per-model contribution to the predicted option
    per_model_scores: dict[str, list[float]]  # normalized scores per model
    models_used: list[str]
    warnings: list[str] = field(default_factory=list)
    stopped_early: bool | None = None
    repeats_used: int | None = None

    def message(self, participant: str = "EEG") -> str:
        """Human-readable demo line, e.g. 'EEG indicates B (dog), confidence 0.72'."""
        if self.unsure or self.predicted_option is None:
            return f"{participant}: unsure (best guess {self.predicted_option}, confidence {self.confidence:.2f})"
        cat = f" ({self.predicted_category})" if self.predicted_category else ""
        who = "EEG indicates" if participant == "EEG" else f"{participant}'s EEG indicates"
        return f"{who} {self.predicted_option}{cat}, confidence {self.confidence:.2f}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["message"] = self.message()
        return d

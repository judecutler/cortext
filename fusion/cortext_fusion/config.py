"""YAML configuration: which adapter each model uses, plus fusion/evaluation settings.

Relative paths in the config are resolved against the config file's folder.
See ``configs/example.yaml`` for an annotated example.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .adapters import Adapter, make_adapter

NORMALIZERS = ("zscore", "logsoftmax", "softmax", "center", "none")


@dataclass
class ModelConfig:
    name: str
    adapter: str = "auto"
    source: str | None = None           # file, folder of per-trial files, or None (in-memory/server)
    options: dict[str, Any] = field(default_factory=dict)
    lower_is_better: bool = False
    field: str | None = None
    weight: float = 1.0
    normalize: str | None = None        # overrides fusion.normalize for this model
    trial_ids: list[str] | None = None  # for .npy without sidecar ids
    trial_order: str | None = None      # "stimulus_log": rows follow the stimulus log order
    file_pattern: str | None = None     # regex extracting trial_id from per-trial file names
    enabled: bool = True
    role: str | None = None             # "p300" | "category" (used for reports/default weights)

    def build_adapter(self) -> Adapter:
        return make_adapter(self.adapter, lower_is_better=self.lower_is_better, field=self.field, **self.options)


@dataclass
class FusionConfig:
    normalize: str = "zscore"
    unsure_threshold: float = 0.4       # confidence below this -> "unsure"
    temperature: float = 1.0
    calibration_file: str | None = None # learned weights (written by `cortext-fusion calibrate`)
    min_gain_logloss: float = 0.01      # a model keeps weight only if it improves held-out log-loss this much


@dataclass
class SequentialConfig:
    stop_confidence: float = 0.9
    min_repeats: int = 3
    max_repeats: int | None = None
    input: str = "prob"          # live per-flash values: "prob" = P(target) per epoch, "score" = raw decision score
    scale: float = 1.0           # evidence per unit of log-odds/score (fit with SequentialFuser.fit_scale)
    final_wait_s: float = 1.0    # after trial_end, wait this long for late scores (the app times out at 3 s)


@dataclass
class EvaluationConfig:
    holdout: str = "session"            # "session" | "participant"
    n_shuffles: int = 200
    n_noise: int = 50
    output_dir: str = "report"
    seed: int = 0
    calibration_bins: int = 10


@dataclass
class Config:
    models: list[ModelConfig]
    markers: str | None = None
    trials: str | None = None
    lab_export: list[str] | None = None   # CorText Lab exports: session folders, "export all" folders or .zip files
    key_mode: str = "auto"
    fusion: FusionConfig = field(default_factory=FusionConfig)
    sequential: SequentialConfig = field(default_factory=SequentialConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    base_dir: Path = field(default_factory=Path.cwd)

    def resolve(self, p: str | None) -> Path | None:
        if p is None:
            return None
        pp = Path(p).expanduser()
        return pp if pp.is_absolute() else (self.base_dir / pp)

    def model(self, name: str) -> ModelConfig:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(f"no model named {name!r} in config (have {[m.name for m in self.models]})")

    def load_metas(self):
        """Load the stimulus log named in the config -> {trial_key: TrialMeta}."""
        from .stimulus_log import load_lab_export, load_stimulus_log
        if self.lab_export:
            import glob
            paths: list[str] = []
            for p in self.lab_export:
                rp = str(self.resolve(p))
                hits = sorted(glob.glob(rp))
                paths += hits or [rp]
            return load_lab_export(paths)
        if self.markers is None:
            raise ValueError("config needs stimulus_log.lab_export (CorText Lab exports) or stimulus_log.markers")
        return load_stimulus_log(self.resolve(self.markers), self.resolve(self.trials) if self.trials else None,
                                 key_mode=self.key_mode)

    @property
    def enabled_models(self) -> list[ModelConfig]:
        return [m for m in self.models if m.enabled]


def _dc(cls, data: dict[str, Any] | None, where: str):
    data = dict(data or {})
    allowed = set(cls.__dataclass_fields__)
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"config section '{where}': unknown key(s) {sorted(unknown)}; allowed: {sorted(allowed)}")
    return cls(**data)


def config_from_dict(d: dict[str, Any], base_dir: str | Path | None = None) -> Config:
    d = dict(d or {})
    models_raw = d.pop("models", None)
    if not models_raw:
        raise ValueError("config needs a 'models:' section with at least one model")
    if isinstance(models_raw, dict):
        models_raw = [{"name": k, **(v or {})} for k, v in models_raw.items()]
    models = []
    for m in models_raw:
        mc = _dc(ModelConfig, m, f"models.{m.get('name', '?')}")
        if mc.normalize is not None and mc.normalize not in NORMALIZERS:
            raise ValueError(f"models.{mc.name}.normalize must be one of {NORMALIZERS}")
        models.append(mc)
    names = [m.name for m in models]
    if len(set(names)) != len(names):
        raise ValueError(f"model names must be unique, got {names}")
    stim = d.pop("stimulus_log", {}) or {}
    fusion = _dc(FusionConfig, d.pop("fusion", None), "fusion")
    if fusion.normalize not in NORMALIZERS:
        raise ValueError(f"fusion.normalize must be one of {NORMALIZERS}")
    seq = _dc(SequentialConfig, d.pop("sequential", None), "sequential")
    ev = _dc(EvaluationConfig, d.pop("evaluation", None), "evaluation")
    if ev.holdout not in ("session", "participant"):
        raise ValueError("evaluation.holdout must be 'session' or 'participant' (never random trials)")
    if d:
        raise ValueError(f"unknown top-level config key(s): {sorted(d)}")
    unknown_stim = set(stim) - {"markers", "trials", "lab_export", "key_mode"}
    if unknown_stim:
        raise ValueError(f"stimulus_log: unknown key(s) {sorted(unknown_stim)}")
    lab = stim.get("lab_export")
    if isinstance(lab, str):
        lab = [lab]
    cfg = Config(models=models, markers=stim.get("markers"), trials=stim.get("trials"), lab_export=lab,
                 key_mode=stim.get("key_mode", "auto"), fusion=fusion,
                 sequential=seq, evaluation=ev, base_dir=Path(base_dir) if base_dir else Path.cwd())
    for m in cfg.models:  # validate adapters early so typos fail at load time
        m.options = {k: (str(cfg.resolve(v)) if k in ("embeddings", "class_names_file") and isinstance(v, str) else v)
                     for k, v in m.options.items()}
        m.build_adapter()
    return cfg


def load_config(path: str | Path) -> Config:
    p = Path(path)
    data = yaml.safe_load(p.read_text())
    return config_from_dict(data, base_dir=p.parent.resolve())

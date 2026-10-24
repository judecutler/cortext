"""CorText fusion layer: adapters, fusion, calibration, evaluation, mock data and a live server.

Quick use from Python::

    from cortext_fusion import load_config, load_dataset, Fuser, fuse_dataset, evaluate
    cfg = load_config("configs/example.yaml")
    ds = load_dataset(cfg)
    preds = fuse_dataset(ds, Fuser.from_config(cfg))
"""
from .adapters import (ADAPTERS, Adapter, AdapterError, AutoAdapter, ClassProbabilitiesAdapter, EmbeddingAdapter,
                       EpochTargetProbAdapter, PerFlashAdapter, PerOptionAdapter, ProbabilitiesAdapter,
                       detect_candidates, make_adapter)
from .config import Config, ModelConfig, config_from_dict, load_config
from .evaluate import evaluate
from .fusion import CalibrationResult, Fuser, ModelWeight, SequentialFuser, calibrate, normalize
from .io import load_any, watch_folder
from .pipeline import Dataset, Issue, fuse_dataset, load_dataset
from .schema import Flash, FusedPrediction, TrialMeta, TrialScores
from .server import Predictor
from .stimulus_log import load_lab_export, load_stimulus_log

__version__ = "0.1.0"

"""Live prediction: a Python API (:class:`Predictor`) and a small local HTTP server.

Endpoints (JSON in, JSON out; CORS enabled so the browser demo app can call them):

GET  /health                    mode, models, weights
POST /predict                   one trial's model outputs -> fused prediction
POST /sequential/start          begin a live trial (options)
POST /sequential/flash          one flash's value -> running prediction (+ stop flag)
POST /sequential/evidence       add a per-trial model's scores to a live trial
POST /sequential/end            forget a live trial

Trial description (used by /predict and /sequential/start)::

    {"trial_id": "P01_S1_T001",
     "options": [{"option_id": "A", "category": "cat"}, {"option_id": "B", "category": "dog"}, ...],
     "target_category": "dog",                  # optional
     "flashes": [{"option_id": "C", "repeat_index": 0}, ...]}   # only for 1-D per-flash arrays

/predict adds ``"outputs": {"p300": <raw>, "category": <raw>}`` where ``<raw>``
is exactly what that model writes for one trial. In **mock mode** outputs are
ignored and scores are faked (biased toward ``"hint_option"`` if given), so the
demo works before any model exists. Every mock response has ``"mock": true``.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Mapping

import numpy as np

from .adapters import Adapter, AdapterError
from .config import Config
from .fusion import Fuser, ModelWeight, SequentialFuser
from .mock import MockPredictor
from .schema import Flash, TrialMeta


def meta_from_request(d: Mapping[str, Any]) -> TrialMeta:
    if "trial_id" not in d:
        raise ValueError("request needs 'trial_id'")
    if "options" in d:
        opts = d["options"]
        ids = [str(o["option_id"] if isinstance(o, dict) else o) for o in opts]
        cats = [str(o.get("category", "")) if isinstance(o, dict) else "" for o in opts]
    elif "option_ids" in d:
        ids = [str(o) for o in d["option_ids"]]
        cats = [str(c) for c in d.get("option_categories", [""] * len(ids))]
    else:
        raise ValueError("request needs 'options' (list of {option_id, category}) or 'option_ids'")
    target_opt = d.get("target_option")
    if target_opt is None and "options" in d:
        for o in d["options"]:
            if isinstance(o, dict) and o.get("is_target"):
                target_opt = str(o["option_id"])
    flashes = [Flash(str(f["option_id"]), int(f.get("repeat_index", -1))) for f in d.get("flashes", [])]
    return TrialMeta(trial_id=str(d["trial_id"]), option_ids=ids, option_categories=cats,
                     target_category=d.get("target_category"), target_option=target_opt, flashes=flashes,
                     session_id=d.get("session_id"), participant_id=d.get("participant_id"))


class Predictor:
    """Fuse one trial at a time. Used by the server, the folder watcher and notebooks.

    >>> p = Predictor.from_config(load_config("configs/example.yaml"))
    >>> p.predict(meta, {"p300": raw_p300, "category": raw_cat}).message()
    """

    def __init__(self, fuser: Fuser, adapters: Mapping[str, Adapter], sequential: SequentialFuser | None = None,
                 mock: bool = False, mock_accuracy: float = 0.8, seed: int | None = None):
        self.fuser, self.adapters = fuser, dict(adapters)
        self.sequential = sequential or SequentialFuser()
        self.mock = mock
        self.mock_predictor = MockPredictor(mock_accuracy, seed)
        self.lock = threading.Lock()
        self._seq_meta: dict[str, TrialMeta] = {}

    @classmethod
    def from_config(cls, cfg: Config, mock: bool = False, **kw) -> "Predictor":
        fuser = Fuser.from_config(cfg)
        adapters = {m.name: m.build_adapter() for m in cfg.enabled_models}
        sc = cfg.sequential
        seq = SequentialFuser(stop_confidence=sc.stop_confidence, min_repeats=sc.min_repeats,
                              max_repeats=sc.max_repeats, unsure_threshold=cfg.fusion.unsure_threshold)
        return cls(fuser, adapters, seq, mock=mock, **kw)

    @classmethod
    def mock_only(cls, unsure_threshold: float = 0.4, **kw) -> "Predictor":
        fuser = Fuser([ModelWeight("mock", 1.0, "zscore")], unsure_threshold=unsure_threshold)
        return cls(fuser, {}, mock=True, **kw)

    def _hint(self, meta: TrialMeta, d: Mapping[str, Any]) -> int | None:
        h = d.get("hint_option", meta.target_option)
        return meta.option_ids.index(str(h)) if h is not None and str(h) in meta.option_ids else None

    def predict(self, meta: TrialMeta, outputs: Mapping[str, Any] | None = None,
                request: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if self.mock:
            s = self.mock_predictor.scores(meta.n_options, self._hint(meta, request or {}))
            fuser = Fuser([ModelWeight("mock", 1.0, "zscore")], unsure_threshold=self.fuser.unsure_threshold)
            out = fuser.fuse(meta, {"mock": s}).to_dict()
            out["mock"] = True
            return out
        outputs = outputs or {}
        scores, warnings = {}, []
        unknown = set(outputs) - set(self.adapters)
        if unknown:
            warnings.append(f"outputs for unknown model(s) {sorted(unknown)} ignored (configured: {sorted(self.adapters)})")
        for name, raw in outputs.items():
            ad = self.adapters.get(name)
            if ad is None:
                continue
            try:
                scores[name] = ad.to_trial_scores(raw, meta, model=name).scores
            except AdapterError as e:
                warnings.append(str(e))
        pred = self.fuser.fuse(meta, scores)
        pred.warnings = warnings + pred.warnings
        out = pred.to_dict()
        out["mock"] = False
        return out

    # sequential --------------------------------------------------------------
    def seq_start(self, meta: TrialMeta) -> dict[str, Any]:
        with self.lock:
            self.sequential.start(meta.trial_id, meta.option_ids, meta.option_categories)
            self._seq_meta[meta.trial_id] = meta
            out = self.sequential.current(meta.trial_id).to_dict()
        out["mock"] = self.mock
        return out

    def seq_flash(self, d: Mapping[str, Any]) -> dict[str, Any]:
        tid = str(d["trial_id"])
        with self.lock:
            value = d.get("value")
            if value is None:
                if not self.mock:
                    raise ValueError("'value' is required (P(target) or a score for this flash)")
                meta = self._seq_meta[tid]
                hint = self._hint(meta, d)
                j = meta.option_ids.index(str(d["option_id"]))
                z = self.mock_predictor.rng.normal(0.8 if j == hint else 0.0, 1.0)
                value = float(1 / (1 + np.exp(-(z - 1.5))))
                if self.sequential.input == "score":
                    value = z
            out = self.sequential.update(tid, str(d["option_id"]), float(value), d.get("repeat_index")).to_dict()
        out["mock"] = self.mock
        return out

    def seq_evidence(self, d: Mapping[str, Any]) -> dict[str, Any]:
        tid, name = str(d["trial_id"]), d["model"]
        with self.lock:
            meta = self._seq_meta[tid]
            scores = self.adapters[name].to_trial_scores(d["output"], meta, model=name).scores \
                if "output" in d else np.asarray(d["scores"], dtype=float)
            w = float(d.get("weight", self.fuser.weights.get(name, 0.0)))
            self.sequential.add_trial_evidence(tid, scores, w)
            out = self.sequential.current(tid).to_dict()
        out["mock"] = self.mock
        return out

    def seq_end(self, tid: str) -> dict[str, Any]:
        with self.lock:
            self.sequential.states.pop(str(tid), None)
            self._seq_meta.pop(str(tid), None)
        return {"ok": True}

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "mock": self.mock, "models": sorted(self.adapters),
                "weights": self.fuser.weights, "unsure_threshold": self.fuser.unsure_threshold,
                "weights_source": self.fuser.source}


def make_handler(predictor: Predictor):
    class Handler(BaseHTTPRequestHandler):
        server_version = "cortext-fusion/0.1"

        def log_message(self, fmt, *args):  # quieter default logging
            if getattr(self.server, "verbose", False):
                super().log_message(fmt, *args)

        def _send(self, code: int, body: Any) -> None:
            data = json.dumps(body, default=_json_default).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):
            self._send(204, {})

        def do_GET(self):
            if self.path.rstrip("/") in ("", "/health"):
                return self._send(200, predictor.health())
            self._send(404, {"error": f"unknown path {self.path}"})

        def do_POST(self):
            try:
                n = int(self.headers.get("Content-Length", 0))
                d = json.loads(self.rfile.read(n) or b"{}")
                path = self.path.rstrip("/")
                if path == "/predict":
                    return self._send(200, predictor.predict(meta_from_request(d), d.get("outputs"), d))
                if path == "/sequential/start":
                    return self._send(200, predictor.seq_start(meta_from_request(d)))
                if path == "/sequential/flash":
                    return self._send(200, predictor.seq_flash(d))
                if path == "/sequential/evidence":
                    return self._send(200, predictor.seq_evidence(d))
                if path == "/sequential/end":
                    return self._send(200, predictor.seq_end(d["trial_id"]))
                self._send(404, {"error": f"unknown path {self.path}"})
            except (ValueError, KeyError, AdapterError, json.JSONDecodeError) as e:
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def serve(predictor: Predictor, host: str = "127.0.0.1", port: int = 8765, verbose: bool = False) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(predictor))
    httpd.verbose = verbose  # type: ignore[attr-defined]
    return httpd

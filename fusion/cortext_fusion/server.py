"""Live prediction: a Python API (:class:`Predictor`) plus two local servers that share it.

1. **WebSocket, for the CorText Lab app** (default ``ws://localhost:8765``, the
   app's default predictor URL). It speaks the app's protocol exactly
   (``cortext-lab/tools/mock_predictor_server.py``)::

     app -> us   {"type":"trial_start","session_id","trial_id","options":[{"id","category","label","position"}]}
     app -> us   {"type":"flash","trial_id","option_id","flash_index","repeat_index","t_ms","wall_ms"}
     app -> us   {"type":"trial_end","trial_id"}
     us  -> app  {"type":"live","trial_id","p300":{option_id: score},"category_scores":{category: score}, ...}
     us  -> app  {"type":"prediction","trial_id","best_option_id","confidence","p300","category_scores", ...}

   The app never sends ``is_target`` or the true answer, so neither the models
   nor this server can see labels. Extra fields we add (``unsure``,
   ``probabilities``, ``message``, ``mock``) are ignored by the app.

2. **HTTP, for the models** (default ``http://localhost:8766``, CORS enabled):

   =========================  =================================================
   GET  /health               mode, models, weights, live trials
   POST /predict              one trial's raw model outputs -> fused prediction
   POST /sequential/flash     one flash's score -> running prediction
   POST /sequential/evidence  a per-trial model's output (e.g. category) for a live trial
   POST /sequential/start     begin a trial without the app (tests, other front ends)
   POST /sequential/end       finish a trial -> final prediction
   =========================  =================================================

   A model refers to a live trial by the app's ``trial_id`` (plus ``session_id``
   if it has it) and to a flash by ``option_id`` or the app's ``flash_index``.
   ``/predict`` for a live trial needs no options: they come from the app, and the
   result is also pushed to the app as its prediction.

**Live flow**: the app sends trial_start/flash events here; the P300 process
reads the EEG + LSL markers and POSTs one score per flash (``value`` =
P(target), or a raw score with ``sequential_input: score``) or one ``/predict``
per trial. Each update is pushed to the app as a ``live`` message. On
``trial_end`` we wait up to ``final_wait_s`` (the app gives up after 3 s) for the
last scores, then send the ``prediction``.

**Mock mode** fakes the scores so the demo runs without models. It has no labels
(the app sends none), so its answers are random unless a ``hint_option`` is
given over HTTP. Every mock message carries ``"mock": true``.
"""
from __future__ import annotations

import json
import socketserver
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Mapping

import numpy as np

from . import ws as wsproto
from .adapters import Adapter, AdapterError
from .config import Config
from .fusion import Fuser, ModelWeight, SequentialFuser, normalize
from .mock import MockPredictor
from .schema import Flash, TrialMeta
from .stimulus_log import make_key, norm_id


def meta_from_request(d: Mapping[str, Any]) -> TrialMeta:
    """Build a TrialMeta from an app ``trial_start`` message or an HTTP trial body."""
    if "trial_id" not in d:
        raise ValueError("request needs 'trial_id'")
    if "options" in d:
        opts = list(d["options"])
        if opts and isinstance(opts[0], dict) and all("position" in o for o in opts):
            opts.sort(key=lambda o: float(o["position"]))
        ids = [norm_id(o.get("option_id", o.get("id")) if isinstance(o, dict) else o) for o in opts]
        cats = [str(o.get("category", "")) if isinstance(o, dict) else "" for o in opts]
    elif "option_ids" in d:
        opts = []
        ids = [norm_id(o) for o in d["option_ids"]]
        cats = [str(c) for c in d.get("option_categories", [""] * len(ids))]
    else:
        raise ValueError("request needs 'options' (list of {option_id or id, category}) or 'option_ids'")
    target_opt = d.get("target_option")
    for o in opts:
        if isinstance(o, dict) and o.get("is_target") and target_opt is None:
            target_opt = norm_id(o.get("option_id", o.get("id")))
    flashes = [Flash(norm_id(f["option_id"]), int(f.get("repeat_index", -1))) for f in d.get("flashes", [])]
    sess = d.get("session_id")
    key = make_key(sess, d["trial_id"]) if sess not in (None, "") else norm_id(d["trial_id"])
    return TrialMeta(trial_id=key, option_ids=ids, option_categories=cats, target_category=d.get("target_category"),
                     target_option=target_opt, flashes=flashes, session_id=None if sess is None else str(sess),
                     participant_id=d.get("participant_id"), local_trial_id=norm_id(d["trial_id"]))


@dataclass
class LiveTrial:
    meta: TrialMeta
    raw_trial_id: Any                       # exactly as the app sent it (the app compares with ===)
    flash_options: dict[int, str] = field(default_factory=dict)   # app flash_index -> option_id
    n_flash_events: int = 0
    n_scored: int = 0
    category_scores: dict[str, float] = field(default_factory=dict)
    pending: dict[int, tuple[float, Any]] = field(default_factory=dict)  # scores that arrived before their flash event
    final: dict[str, Any] | None = None     # set by /predict for this trial
    ended: bool = False
    started: float = field(default_factory=time.monotonic)


class Predictor:
    """Fuse one trial at a time; also tracks live trials coming from the app.

    >>> p = Predictor.from_config(load_config("configs/example.yaml"))
    >>> p.predict(meta, {"p300": raw_p300, "category": raw_cat})["message"]
    """

    def __init__(self, fuser: Fuser, adapters: Mapping[str, Adapter], sequential: SequentialFuser | None = None,
                 mock: bool = False, mock_accuracy: float = 0.8, seed: int | None = None,
                 final_wait_s: float = 1.0, sequential_model: str | None = None):
        self.fuser, self.adapters = fuser, dict(adapters)
        self.sequential = sequential or SequentialFuser()
        self.mock = mock
        self.mock_predictor = MockPredictor(mock_accuracy, seed)
        self.final_wait_s = final_wait_s
        self.sequential_model = sequential_model or next(iter(self.adapters), "p300")
        self.lock = threading.RLock()
        self.live: dict[str, LiveTrial] = {}
        self.listeners: list[Callable[[dict[str, Any]], None]] = []   # e.g. WebSocket broadcast

    @classmethod
    def from_config(cls, cfg: Config, mock: bool = False, **kw) -> "Predictor":
        fuser = Fuser.from_config(cfg)
        adapters = {m.name: m.build_adapter() for m in cfg.enabled_models}
        sc = cfg.sequential
        seq = SequentialFuser(stop_confidence=sc.stop_confidence, min_repeats=sc.min_repeats,
                              max_repeats=sc.max_repeats, unsure_threshold=cfg.fusion.unsure_threshold,
                              input=sc.input, scale=sc.scale)
        primary = next((m.name for m in cfg.enabled_models if m.weight != 0), None)
        return cls(fuser, adapters, seq, mock=mock, final_wait_s=sc.final_wait_s, sequential_model=primary, **kw)

    @classmethod
    def mock_only(cls, unsure_threshold: float = 0.4, **kw) -> "Predictor":
        fuser = Fuser([ModelWeight("mock", 1.0, "zscore")], unsure_threshold=unsure_threshold)
        return cls(fuser, {}, SequentialFuser(unsure_threshold=unsure_threshold), mock=True, **kw)

    # ------------------------------------------------------------------ helpers
    def _emit(self, msg: dict[str, Any]) -> None:
        for fn in list(self.listeners):
            try:
                fn(msg)
            except Exception:  # a dead client must not break the others
                pass

    def _hint(self, meta: TrialMeta, d: Mapping[str, Any]) -> int | None:
        h = d.get("hint_option", meta.target_option)
        return meta.option_ids.index(norm_id(h)) if h is not None and norm_id(h) in meta.option_ids else None

    def find_live(self, d: Mapping[str, Any]) -> LiveTrial:
        """Find a live trial from a model's message (trial_id [+ session_id])."""
        tid = d.get("trial_id")
        if tid is None:
            raise ValueError("'trial_id' is required")
        sess = d.get("session_id")
        cands = [make_key(sess, tid)] if sess not in (None, "") else [norm_id(tid)]
        for k in cands:
            if k in self.live:
                return self.live[k]
        same = [lt for lt in self.live.values() if lt.meta.local_trial_id == norm_id(tid)]
        if len(same) == 1:
            return same[0]
        if same:  # several sessions reuse the id: take the most recent
            return max(same, key=lambda lt: lt.started)
        raise KeyError(f"no live trial {tid!r} (live: {[lt.meta.trial_id for lt in self.live.values()][:5]})")

    def _app_payload(self, lt: LiveTrial, pred: dict[str, Any]) -> dict[str, Any]:
        ids = lt.meta.option_ids
        p300 = dict(zip(ids, (round(float(x), 4) for x in pred.get("fused_scores", [0.0] * len(ids)))))
        return {"trial_id": lt.raw_trial_id, "p300": p300, "category_scores": dict(lt.category_scores),
                "probabilities": dict(zip(ids, pred.get("probabilities", []))), "confidence": pred.get("confidence"),
                "unsure": pred.get("unsure"), "message": pred.get("message"), "mock": self.mock,
                "repeats_used": pred.get("repeats_used")}

    # ------------------------------------------------------------------ batch / per-trial
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

    def predict_request(self, d: Mapping[str, Any]) -> dict[str, Any]:
        """HTTP /predict: options in the body, or taken from the live trial the app started."""
        with self.lock:
            lt = None
            if "options" not in d and "option_ids" not in d:
                lt = self.find_live(d)
                meta = lt.meta
            else:
                meta = meta_from_request(d)
                lt = self.live.get(meta.trial_id)
            out = self.predict(meta, d.get("outputs"), d)
            if lt is not None:
                cat_name = next((n for n in self.adapters if n != self.sequential_model and n in (d.get("outputs") or {})), None)
                if cat_name and cat_name in out.get("per_model_scores", {}):
                    lt.category_scores = dict(zip(lt.meta.option_categories, out["per_model_scores"][cat_name]))
                lt.final = out
                if lt.ended:
                    self._send_prediction(lt)
                else:
                    self._emit({"type": "live", **self._app_payload(lt, out)})
            return out

    # ------------------------------------------------------------------ live trials
    def start(self, meta: TrialMeta, raw_trial_id: Any = None) -> dict[str, Any]:
        with self.lock:
            lt = LiveTrial(meta, meta.local_trial_id if raw_trial_id is None else raw_trial_id)
            self.live[meta.trial_id] = lt
            self.sequential.start(meta.trial_id, meta.option_ids, meta.option_categories)
            # keep the live table small
            if len(self.live) > 200:
                for k in sorted(self.live, key=lambda k: self.live[k].started)[:50]:
                    self.live.pop(k, None)
                    self.sequential.states.pop(k, None)
            out = self.sequential.current(meta.trial_id).to_dict()
            out["mock"] = self.mock
            self._emit({"type": "live", **self._app_payload(lt, out)})
            return out

    def app_flash(self, d: Mapping[str, Any]) -> None:
        """A flash event from the app (no score: the model provides it, or the mock in mock mode)."""
        with self.lock:
            lt = self.find_live(d)
            oid = norm_id(d["option_id"])
            if "flash_index" in d and d["flash_index"] is not None:
                lt.flash_options[int(d["flash_index"])] = oid
            rep = d.get("repeat_index")
            lt.meta.flashes.append(Flash(oid, int(rep) if rep is not None else -1))
            lt.n_flash_events += 1
            fi = d.get("flash_index")
            if fi is not None and int(fi) in lt.pending:  # the model was faster than the app's event
                value, prep = lt.pending.pop(int(fi))
                self._score(lt, oid, value, prep if prep is not None else rep)
            elif self.mock:
                j = lt.meta.option_ids.index(oid)
                hint = self._hint(lt.meta, d)
                z = self.mock_predictor.rng.normal(0.8 if j == hint else 0.0, 1.0)
                value = z if self.sequential.input == "score" else float(1 / (1 + np.exp(-(z - 1.5))))
                self._score(lt, oid, value, rep)

    def _score(self, lt: LiveTrial, option_id: str, value: float, repeat_index: Any) -> dict[str, Any]:
        out = self.sequential.update(lt.meta.trial_id, option_id, float(value),
                                     None if repeat_index is None else int(repeat_index)).to_dict()
        lt.n_scored += 1
        out["mock"] = self.mock
        self._emit({"type": "live", **self._app_payload(lt, out)})
        return out

    def model_flash(self, d: Mapping[str, Any]) -> dict[str, Any]:
        """A model's score for one flash: {trial_id, [session_id], option_id or flash_index, value}."""
        with self.lock:
            lt = self.find_live(d)
            if d.get("value") is None:
                raise ValueError("'value' is required (P(target), or a score if sequential.input is 'score')")
            oid = d.get("option_id")
            if oid is None:
                fi = d.get("flash_index")
                if fi is None:
                    raise ValueError("give 'option_id' or 'flash_index'")
                if int(fi) not in lt.flash_options:  # flash event not here yet: hold the score until it is
                    lt.pending[int(fi)] = (float(d["value"]), d.get("repeat_index"))
                    out = self.sequential.current(lt.meta.trial_id).to_dict()
                    out.update(mock=self.mock, queued=True)
                    return out
                oid = lt.flash_options[int(fi)]
            oid = norm_id(oid)
            if oid not in lt.meta.option_ids:
                raise ValueError(f"option {oid!r} is not in trial {lt.meta.trial_id} {lt.meta.option_ids}")
            return self._score(lt, oid, float(d["value"]), d.get("repeat_index"))

    def model_evidence(self, d: Mapping[str, Any]) -> dict[str, Any]:
        """A per-trial model's output for a live trial (e.g. the category model)."""
        with self.lock:
            lt = self.find_live(d)
            name = d.get("model")
            if "output" in d:
                if name not in self.adapters:
                    raise ValueError(f"unknown model {name!r}; configured: {sorted(self.adapters)}")
                scores = self.adapters[name].to_trial_scores(d["output"], lt.meta, model=name).scores
            else:
                scores = np.asarray(d["scores"], dtype=float)
            w = float(d.get("weight", self.fuser.weights.get(name, 0.0)))
            z = normalize(scores, "zscore")
            lt.category_scores = dict(zip(lt.meta.option_categories, (round(float(x), 4) for x in z)))
            self.sequential.add_trial_evidence(lt.meta.trial_id, scores, w)
            out = self.sequential.current(lt.meta.trial_id).to_dict()
            out["mock"] = self.mock
            out["applied_weight"] = w
            self._emit({"type": "live", **self._app_payload(lt, out)})
            return out

    def final_prediction(self, lt: LiveTrial) -> dict[str, Any]:
        if lt.final is not None:
            return lt.final
        out = self.sequential.current(lt.meta.trial_id).to_dict()
        out["mock"] = self.mock
        if lt.n_scored == 0:
            out["unsure"] = True
            out["warnings"] = ["no model scores arrived for this trial"]
        return out

    def _send_prediction(self, lt: LiveTrial) -> dict[str, Any]:
        out = self.final_prediction(lt)
        msg = {"type": "prediction", **self._app_payload(lt, out), "best_option_id": out.get("predicted_option")}
        if out.get("predicted_option") is None and out.get("probabilities"):
            msg["best_option_id"] = None
        self._emit(msg)
        return msg

    def end(self, d: Mapping[str, Any], wait: bool = True) -> dict[str, Any]:
        """Finish a live trial: wait (briefly) for outstanding scores, then send the prediction."""
        with self.lock:
            lt = self.find_live(d)
            lt.ended = True
        if wait and not self.mock:
            deadline = time.monotonic() + self.final_wait_s
            while time.monotonic() < deadline:
                with self.lock:
                    if lt.final is not None or (lt.n_flash_events and lt.n_scored >= lt.n_flash_events):
                        break
                time.sleep(0.02)
        with self.lock:
            return self._send_prediction(lt)

    def health(self) -> dict[str, Any]:
        return {"status": "ok", "mock": self.mock, "models": sorted(self.adapters),
                "weights": self.fuser.weights, "unsure_threshold": self.fuser.unsure_threshold,
                "weights_source": self.fuser.source, "sequential_input": self.sequential.input,
                "live_trials": [lt.meta.trial_id for lt in list(self.live.values())[-10:]]}


# ---------------------------------------------------------------------- HTTP (models)

def make_http_handler(predictor: Predictor):
    class Handler(BaseHTTPRequestHandler):
        server_version = "cortext-fusion/0.1"

        def log_message(self, fmt, *args):
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
                    return self._send(200, predictor.predict_request(d))
                if path == "/sequential/start":
                    meta = meta_from_request(d)
                    return self._send(200, predictor.start(meta, d["trial_id"]))
                if path == "/sequential/flash":
                    if predictor.mock and d.get("value") is None:
                        predictor.app_flash(d)
                        with predictor.lock:
                            lt = predictor.find_live(d)
                            out = predictor.sequential.current(lt.meta.trial_id).to_dict()
                        out["mock"] = True
                        return self._send(200, out)
                    return self._send(200, predictor.model_flash(d))
                if path == "/sequential/evidence":
                    return self._send(200, predictor.model_evidence(d))
                if path == "/sequential/end":
                    return self._send(200, predictor.end(d))
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


def serve(predictor: Predictor, host: str = "127.0.0.1", port: int = 8766, verbose: bool = False) -> ThreadingHTTPServer:
    """HTTP server for models (and tests). Call ``serve_forever()`` on the result."""
    httpd = ThreadingHTTPServer((host, port), make_http_handler(predictor))
    httpd.verbose = verbose  # type: ignore[attr-defined]
    return httpd


# ---------------------------------------------------------------------- WebSocket (CorText Lab app)

class _WSServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_ws_handler(predictor: Predictor, verbose: bool = False):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            if wsproto.handshake(self.request) is None:
                return
            send_lock = threading.Lock()

            def raw(op: int, payload: bytes) -> None:
                with send_lock:
                    self.request.sendall(wsproto.build_frame(op, payload))

            def send(obj: dict[str, Any]) -> None:
                raw(0x1, json.dumps(obj, default=_json_default).encode())

            predictor.listeners.append(send)
            try:
                while True:
                    try:
                        text = wsproto.read_message(self.rfile, raw)
                    except (ConnectionError, OSError):
                        return
                    if text is None:
                        return
                    try:
                        msg = json.loads(text)
                        t = msg.get("type")
                        if t == "trial_start":
                            predictor.start(meta_from_request(msg), msg.get("trial_id"))
                        elif t == "flash":
                            predictor.app_flash(msg)
                        elif t == "trial_end":
                            threading.Thread(target=predictor.end, args=(msg,), daemon=True).start()
                        elif verbose:
                            print(f"ws: ignoring message type {t!r}")
                    except (ValueError, KeyError, json.JSONDecodeError) as e:
                        send({"type": "error", "error": f"{type(e).__name__}: {e}"})
            finally:
                try:
                    predictor.listeners.remove(send)
                except ValueError:
                    pass

    return Handler


def serve_ws(predictor: Predictor, host: str = "127.0.0.1", port: int = 8765, verbose: bool = False) -> _WSServer:
    """WebSocket server for the CorText Lab app. Call ``serve_forever()`` on the result."""
    return _WSServer((host, port), make_ws_handler(predictor, verbose))

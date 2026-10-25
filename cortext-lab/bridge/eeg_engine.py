"""EEG side of the bridge: finds the headset's LSL stream, records it per session, measures signal quality and band
power for the brain map, and scores P300 evidence for each answer option live.

Data flow
  headset --(eeg/stream_eeg.py, gpype)--> LSL "CorTextEEG" --> EEGEngine (this file)
     * ring buffer: raw, filtered by the running filter chain (filters.py, switchable on the Agent page), and a fixed
       0.5-20 Hz reference copy used only for electrode quality (so quality chips mean the same whatever chain runs)
     * logs/eeg_<session>.csv  (lsl_time, page_t_ms, Fz, C3, ...)  while a session is running
     * quality + band power per channel -> /api/status, the app's setup page, dev panel and brain map
  flash markers from the app (with LSL timestamps computed by the bridge) --> P300Scorer --> /predict WebSocket

Labels: flash markers carry is_target. The scorer NEVER reads it for the trial it is predicting; labels are only moved
into the adaptive training pool after that trial has ended (its 'trial_end' marker), and invalid/aborted trials are
excluded.

Only pylsl is needed to record; numpy + scipy are needed for quality, band power and scoring.
"""
import csv
import os
import threading
import time

import filters as F

try:
    import numpy as np
    import p300 as P
    NUMPY_ERROR = None
except Exception as exc:  # numpy/scipy missing
    np = None
    P = None
    NUMPY_ERROR = repr(exc)

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
RING_SECONDS = 90


def _labels_from_info(info, n):
    labels, units = [], []
    try:
        ch = info.desc().child("channels").child("channel")
        while not ch.empty() and len(labels) < n:
            labels.append(ch.child_value("label") or "")
            units.append(ch.child_value("unit") or "")
            ch = ch.next_sibling("channel")
    except Exception:
        pass
    if len(labels) != n or not all(labels):
        labels = list(P.UNICORN_MONTAGE) if (P and n == 8) else [f"Ch{i + 1}" for i in range(n)]
        units = ["uV"] * n
    return labels, units


class EEGEngine:
    def __init__(self, state, pylsl, stream_name=None, model_path=None, scorer_mode="auto", log_dir="logs",
                 filter_spec=None, filter_source="built-in default"):
        self.state = state
        self.pylsl = pylsl
        self.stream_name = stream_name
        self.log_dir = log_dir
        self.lock = threading.RLock()
        self.info = {"found": False}
        self.labels, self.srate = [], 0.0
        self.scale = 1.0
        self.ts = None
        self.raw = None
        self.filt = None
        self.qfilt = None
        self.filter = None
        self.qfilter = None
        self.filter_spec = F.normalize(filter_spec or F.DEFAULT_SPEC)
        self.filter_sig = F.signature(self.filter_spec)
        self.filter_source = filter_source
        self.filter_error = None
        self.filter_changed_at = None
        self.session_id = None
        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None
        self.csv_paths = {}       # session_id -> [paths]
        self.samples_recorded = 0
        self.scorer = P300Scorer(self, model_path, scorer_mode, log_dir) if P else None
        self.evidence = None   # latest live/final per-option evidence (for the neural monitor page)

    # ---------- session recording ----------
    def start_session(self, session_id):
        with self.lock:
            if session_id == self.session_id and self.csv_file:
                return self.csv_path
            self._close_csv()
            self.session_id = session_id
            self.samples_recorded = 0
            if self.labels:
                self._open_csv()
            return self.csv_path

    def stop_session(self, session_id):
        """Stop recording this session (if it is the active one) and close its file, so the files can be deleted."""
        with self.lock:
            if self.session_id == session_id:
                self._close_csv()
                self.session_id = None
            self.csv_paths.pop(session_id, None)

    def _open_csv(self):
        os.makedirs(self.log_dir, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(self.session_id))
        path = os.path.join(self.log_dir, f"eeg_{safe}_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        self.csv_file = open(path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["lsl_time", "page_t_ms"] + self.labels)
        self.csv_path = path
        self.csv_paths.setdefault(self.session_id, []).append(path)
        try:  # sidecar with the stream description, read by pipeline/build_session.py
            import json
            info = {k: v for k, v in self.info.items() if k not in ("quality", "bands", "recording")}
            with open(path[:-4] + ".json", "w", encoding="utf-8") as fh:
                json.dump({**info, "labels": self.labels, "srate": self.srate}, fh, indent=1)
        except Exception:
            pass

    def _close_csv(self):
        if self.csv_file:
            try:
                self.csv_file.close()
            except Exception:
                pass
        self.csv_file = self.csv_writer = self.csv_path = None

    def session_files(self, session_id):
        return [p for p in self.csv_paths.get(session_id, []) if os.path.exists(p)] or self._find_files(session_id)

    def _find_files(self, session_id):
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(session_id))
        try:
            return sorted(os.path.join(self.log_dir, f) for f in os.listdir(self.log_dir)
                          if f.startswith(f"eeg_{safe}_") and f.endswith(".csv"))
        except OSError:
            return []

    # ---------- acquisition thread ----------
    def run(self):
        pl = self.pylsl
        inlet = None
        last_sample = None
        count, t_window = 0, time.time()
        last_quality = 0.0
        while True:
            try:
                if inlet is None:
                    inlet = self._connect()
                    if inlet is None:
                        time.sleep(1.0)
                        continue
                    count, t_window, last_sample = 0, time.time(), None
                chunk, stamps = inlet.pull_chunk(timeout=0.05, max_samples=1024)
                now = time.time()
                if stamps:
                    self._ingest(chunk, stamps)
                    count += len(stamps)
                    last_sample = now
                if now - t_window >= 1.0:
                    self.info["measured_srate"] = round(count / (now - t_window), 1)
                    count, t_window = 0, now
                self.info["age_ms"] = None if last_sample is None else round((now - last_sample) * 1000)
                lt = self.last_ts()
                if lt is not None:  # how far the newest EEG sample is behind "now" (Bluetooth + LSL delay)
                    self.info["delay_ms"] = round((pl.local_clock() - lt) * 1000)
                if last_sample is not None and now - last_sample > 5.0:
                    raise ConnectionError("no EEG samples for 5 s")
                if P and now - last_quality > 0.5:
                    last_quality = now
                    self._update_quality()
                if self.scorer:
                    self.scorer.tick()
            except Exception as exc:
                self.info = {"found": False, "error": repr(exc)}
                inlet = None
                with self.lock:
                    self._close_csv()
                time.sleep(1.0)

    def _connect(self):
        pl = self.pylsl
        found = (pl.resolve_byprop("name", self.stream_name, timeout=1.0) if self.stream_name
                 else pl.resolve_byprop("type", "EEG", timeout=1.0))
        if not found:
            self.info = {"found": False}
            return None
        found.sort(key=lambda s: ("FAKE" in s.name().upper(), s.name()))  # prefer a real headset over the simulator
        inlet = pl.StreamInlet(found[0], max_buflen=30, processing_flags=pl.proc_clocksync | pl.proc_dejitter)
        full = inlet.info(timeout=2.0)
        n = full.channel_count()
        labels, units = _labels_from_info(full, n)
        simulated = "FAKE" in full.name().upper() or (full.desc().child_value("simulated") == "true")
        scale = 1e6 if units and all(u.lower() in ("v", "volts") for u in units) else (1e3 if units and units[0].lower() == "mv" else 1.0)
        with self.lock:
            self.labels, self.srate, self.scale = labels, full.nominal_srate() or 250.0, scale
            self.ts = self.raw = self.filt = self.qfilt = None
            self.qfilter = P.CausalFilter(self.srate, n) if P else None
            self.filter = self._build_chain(self.filter_spec) if P else None
            self.info = {"found": True, "name": full.name(), "type": full.type(), "channels": n, "labels": labels,
                         "nominal_srate": self.srate, "measured_srate": None, "age_ms": None, "simulated": simulated,
                         "source_id": full.source_id(), "hostname": full.hostname(),
                         "device": "Simulated EEG (fake_eeg.py)" if simulated else self._guess_device(full, labels),
                         "quality": [], "bands": {}, "recording": None}
            if self.session_id and not self.csv_file:
                self._open_csv()
        if self.scorer:
            self.scorer.on_stream(labels)
        return inlet

    @staticmethod
    def _guess_device(info, labels):
        serial = ""
        try:
            serial = info.desc().child("device").child_value("serial")
        except Exception:
            pass
        if labels == ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "POz", "PO8"]:
            return "g.tec Unicorn / BCI Core-8" + (f" ({serial})" if serial else "")
        return f"{info.name()} ({info.channel_count()} ch)"

    def _ingest(self, chunk, stamps):
        pl = self.pylsl
        offset = self.state.clock_offset_ms
        lsl_minus_perf_ms = pl.local_clock() * 1000.0 - time.perf_counter() * 1000.0
        with self.lock:
            if self.csv_writer:
                for row, t in zip(chunk, stamps):
                    page_t = "" if offset is None else round(t * 1000.0 - lsl_minus_perf_ms - offset, 3)
                    self.csv_writer.writerow([f"{t:.6f}", page_t] + [f"{v * self.scale:.3f}" for v in row])
                self.samples_recorded += len(stamps)
                self.csv_file.flush()
                self.info["recording"] = {"session_id": self.session_id, "path": self.csv_path, "samples": self.samples_recorded}
            if np is None:
                return
            x = np.asarray(chunk, float) * self.scale
            t = np.asarray(stamps, float)
            y = self.filter(x)
            q = self.qfilter(x)
            if self.ts is None:
                self.ts, self.raw, self.filt, self.qfilt = t, x, y, q
            else:
                self.ts = np.concatenate([self.ts, t])
                self.raw = np.concatenate([self.raw, x])
                self.filt = np.concatenate([self.filt, y])
                self.qfilt = np.concatenate([self.qfilt, q])
            keep = int(RING_SECONDS * self.srate)
            if len(self.ts) > 2 * keep:
                self.ts, self.raw, self.filt, self.qfilt = self.ts[-keep:], self.raw[-keep:], self.filt[-keep:], self.qfilt[-keep:]

    # ---------- filter chain ----------
    def _build_chain(self, spec):
        """A Chain for the current stream. If it can't be built here (e.g. a reference channel this headset lacks),
        fall back to the default chain and report why, so scoring never runs on half-configured filters."""
        try:
            ch = F.Chain(spec, self.srate, self.labels)
            self.filter_error = None
            return ch
        except F.FilterError as exc:
            self.filter_error = f"{exc}. Running the built-in default chain until this is fixed."
            return F.Chain(F.DEFAULT_SPEC, self.srate, self.labels)

    @property
    def running_signature(self):
        """Signature of the chain the samples are actually filtered with (the default one if the chosen chain failed)."""
        return self.filter.signature if self.filter is not None else self.filter_sig

    @property
    def reject_uv(self):
        return self.filter.reject_uv if self.filter is not None else self.filter_spec["reject_uv"]

    def set_filters(self, spec, source="Agent page"):
        """Switch the running chain now. The raw ring buffer is re-filtered from its start, so every epoch is cut from
        consistently filtered data. Raises FilterError if the chain can't run on this stream (nothing changes then)."""
        spec = F.normalize(spec)
        with self.lock:
            if self.labels and np is not None:
                F.Chain(spec, self.srate, self.labels)        # validate against this headset first
            old_sig = self.running_signature
            self.filter_spec, self.filter_sig, self.filter_source = spec, F.signature(spec), source
            self.filter_changed_at = time.strftime("%H:%M:%S")
            if self.labels and np is not None:
                self.filter = self._build_chain(spec)
                if self.raw is not None and len(self.raw):
                    self.filt = self.filter(self.raw)
            changed = old_sig != self.running_signature
        if changed and self.scorer:
            self.scorer.on_filters_changed()
        return spec

    def raw_window(self, seconds):
        """(ts, raw µV, labels, srate) for the last `seconds`, for the Agent page's filter preview."""
        with self.lock:
            if self.ts is None:
                return None, None, list(self.labels), self.srate
            n = int(seconds * self.srate)
            return self.ts[-n:].copy(), self.raw[-n:].copy(), list(self.labels), self.srate

    def filter_status(self):
        return {"name": self.filter_spec["name"], "spec": self.filter_spec, "signature": self.running_signature,
                "chosen_signature": self.filter_sig, "source": self.filter_source, "error": self.filter_error,
                "summary": F.summary(self.filter.spec if self.filter is not None else self.filter_spec),
                "reject_uv": self.reject_uv, "changed_at": self.filter_changed_at, "srate": self.srate or None}

    def snapshot(self, seconds):
        """Copy of the last `seconds` of (ts, filtered) data."""
        with self.lock:
            if self.ts is None:
                return None, None
            n = int(seconds * self.srate)
            return self.ts[-n:].copy(), self.filt[-n:].copy()

    def window(self, t_from):
        with self.lock:
            if self.ts is None:
                return None, None
            i = np.searchsorted(self.ts, t_from)
            return self.ts[i:].copy(), self.filt[i:].copy()

    def last_ts(self):
        with self.lock:
            return None if self.ts is None or not len(self.ts) else float(self.ts[-1])

    def _update_quality(self):
        with self.lock:
            if self.ts is None or len(self.ts) < self.srate * 2:
                return
            n = int(2 * self.srate)
            filt = self.qfilt[-n:]
            raw = self.raw[-int(4 * self.srate):]
            labels = list(self.labels)
            srate = self.srate
        std = filt.std(axis=0)
        q = []
        for i, l in enumerate(labels):
            s = float(std[i])
            status = "flat" if s < P.FLAT_STD_UV else "noisy" if s > P.NOISY_STD_UV else "good"
            q.append({"label": l, "std_uv": round(s, 1), "status": status})
        # band power per channel (Welch-like: mean periodogram of 1 s Hann windows over the last 4 s of raw data)
        bands = {}
        seg = int(srate)
        if len(raw) >= seg:
            r = raw - raw.mean(axis=0)
            win = np.hanning(seg)[:, None]
            specs = [np.abs(np.fft.rfft(r[k:k + seg] * win, axis=0)) ** 2 for k in range(0, len(r) - seg + 1, seg // 2)]
            psd = np.mean(specs, axis=0)
            f = np.fft.rfftfreq(seg, 1 / srate)
            total = psd[(f >= 1) & (f <= 30)].sum(axis=0) + 1e-12
            for name, (lo, hi) in BANDS.items():
                bands[name] = [round(float(v), 4) for v in psd[(f >= lo) & (f < hi)].sum(axis=0) / total]
        self.info["quality"] = q
        self.info["bands"] = bands
        self.info["good_channels"] = sum(1 for x in q if x["status"] == "good")

    def trace(self, seconds=5.0, hz=50):
        if np is None:
            return {"error": "numpy not installed"}
        ts, x = self.snapshot(seconds)
        if ts is None or not len(ts):
            return {"labels": self.labels, "data": [], "hz": hz}
        step = max(1, int(round(self.srate / hz)))
        x = x[: len(x) // step * step].reshape(-1, step, x.shape[1]).mean(axis=1)
        return {"labels": self.labels, "hz": self.srate / step, "t_end": float(ts[-1]),
                "data": [[round(float(v), 2) for v in x[:, c]] for c in range(x.shape[1])],
                "quality": self.info.get("quality"), "bands": self.info.get("bands"),
                "simulated": self.info.get("simulated"), "evidence": self.evidence}

    def status(self):
        out = dict(self.info)
        if NUMPY_ERROR:
            out["analysis_error"] = f"numpy/scipy missing ({NUMPY_ERROR}); recording only. pip install numpy scipy"
        if self.scorer:
            out["scorer"] = self.scorer.status()
        out["filters"] = self.filter_status()
        return out


class P300Scorer:
    """Turns flash markers + EEG into per-option P300 evidence. Three ways to score, chosen per prediction:

    * model     - a trained LinearP300: --p300-model, else the file in models/active/, or one picked on the Setup
                  page while running (set_model). Trained offline with pipeline/train_p300.py or calibrated live.
    * adaptive  - a LinearP300 retrained in the bridge after every finished valid trial of the current session
    * heuristic - untrained: mean 250-500 ms amplitude over parietal channels (works only for a strong P300)
    mode 'auto' = model if given, else adaptive once it has enough data, else heuristic.
    """
    MIN_TRIALS = 5
    MIN_TARGET_EPOCHS = 30

    def __init__(self, engine, model_path, mode, log_dir):
        self.engine = engine
        self.mode = mode
        self.log_dir = log_dir
        self.lock = threading.RLock()
        self.trials = {}            # (session_id, trial_id) -> {"flashes": [(lsl_ts, option_id, is_target)], ...}
        self.pool = []              # finished valid trials: {"key","X","y","options","true"}
        self.adaptive = None
        self.adaptive_session = None
        self.adaptive_participant = None
        self.model = None
        self.model_path = model_path
        self.model_error = None
        self.model_source = "command line (--p300-model)" if model_path else "none"
        self.model_info = None
        self.live_cbs = set()
        self.filters_reset_note = None
        self._live_keys = {}
        if model_path:
            try:
                self.set_model(model_path, self.model_source, quiet=True)
            except ValueError:
                pass          # kept in model_error and shown on the Setup page

    def set_model(self, path, source="Setup page", quiet=False):
        """Load a trained model and score with it from the next prediction on. path None = no trained model
        (adapt to the session, else heuristic). On a load error the previous model stays and the error is raised."""
        if path is None:
            with self.lock:
                self.model, self.model_path, self.model_error, self.model_info = None, None, None, None
                self.model_source = source
            return
        try:
            m = P.LinearP300.load(path)
        except Exception as exc:
            msg = f"could not load {path}: {exc}"
            with self.lock:
                if self.model is None:
                    self.model_path, self.model_error, self.model_source = path, msg, source
            if not quiet:
                print("  P300 model:", msg)
            raise ValueError(msg)
        err = None
        labels = list(self.engine.labels or [])
        if labels:
            try:
                m.reorder_for(labels)
            except ValueError as exc:
                err = str(exc)
        meta = m.meta or {}
        with self.lock:
            self.model, self.model_path, self.model_error, self.model_source = m, path, err, source
            self.model_info = {"file": os.path.basename(path), "participants": meta.get("participants") or (
                [meta["participant_id"]] if meta.get("participant_id") else []), "n_epochs": meta.get("n_epochs"),
                "cv_answer_accuracy": meta.get("cv_answer_accuracy"), "simulated": meta.get("simulated"),
                "channels": list(m.labels), "filters": m.filters, "filter_signature": m.filter_signature,
                "filter_summary": F.summary(m.filters)}
        if not quiet:
            print(f"  P300 model: {os.path.basename(path)} ({source})" + (f"  WARNING: {err}" if err else ""))

    def filters_match(self):
        """Is the trained model's filter chain the one running? (No model: True.)"""
        return self.model is None or self.model.filter_signature == self.engine.running_signature

    def on_filters_changed(self):
        """The running chain changed, so features made before no longer match. The live calibration is rebuilt from the
        trials whose EEG is still in the ring buffer (it was re-filtered); older trials drop out."""
        with self.lock:
            had = len(self.pool)
            self.pool, self.adaptive = [], None
            for key, t in self.trials.items():
                if t.get("pooled") and key[0] == self.adaptive_session:
                    t["pooled"] = False        # tick() re-cuts them from the re-filtered buffer if it still covers them
            self.filters_reset_note = (f"Filters changed at {time.strftime('%H:%M:%S')}: live calibration rebuilt from the "
                                       f"last {RING_SECONDS} s of EEG ({had} trial(s) were in it before).") if had else None

    def on_stream(self, labels):
        if self.model is not None:
            try:
                self.model.reorder_for(labels)
                self.model_error = None
            except ValueError as exc:
                self.model_error = str(exc)

    # ---- markers (called from the bridge for every marker the app sends) ----
    def on_marker(self, msg, lsl_ts):
        ev = msg.get("event")
        key = (msg.get("session_id"), msg.get("trial_id"))
        with self.lock:
            if ev == "session_start" and msg.get("session_id") != self.adaptive_session:
                self.pool, self.adaptive, self.adaptive_session = [], None, msg.get("session_id")
                self.adaptive_participant = (msg.get("extra") or {}).get("participant_id")
            if ev == "trial_start":
                self.trials[key] = {"flashes": [], "done": False, "invalid": False, "options": None}
                if len(self.trials) > 300:
                    for k in list(self.trials)[:100]:
                        self.trials.pop(k, None)
            elif ev == "flash_on" and lsl_ts is not None:
                self.trials.setdefault(key, {"flashes": [], "done": False, "invalid": False, "options": None})
                self.trials[key]["flashes"].append((lsl_ts, msg.get("option_id"), bool(msg.get("is_target"))))
            elif ev == "trial_end" and key in self.trials:
                self.trials[key]["done"] = True        # labels may be used for training once this trial is predicted
                self.trials[key]["done_at"] = time.time()
            elif ev in ("trial_invalid", "trial_aborted", "trial_redo") and key in self.trials:
                self.trials[key]["invalid"] = True
                self.pool = [p for p in self.pool if p["key"] != key]

    def tick(self):
        """Called from the acquisition loop: move finished trials whose EEG is complete into the training pool."""
        last = self.engine.last_ts()
        if last is None:
            return
        retrain = False
        with self.lock:
            for key, t in list(self.trials.items()):
                if not t["done"] or t["invalid"] or t.get("pooled") or not t["flashes"]:
                    continue
                # never let a trial's labels into the model before that trial's own prediction has been made
                if not t.get("predicted") and time.time() - t.get("done_at", 0) < 4.0:
                    continue
                if last < max(f[0] for f in t["flashes"]) + P.TMAX + 0.05:
                    continue
                t["pooled"] = True
                X, y, opts = self._features(t["flashes"])
                if X is not None and y.sum() > 0:
                    true = next((f[1] for f in t["flashes"] if f[2]), None)
                    self.pool.append({"key": key, "X": X, "y": y, "options": opts, "true": true})
                    retrain = True
        if retrain:
            self._retrain()

    def _features(self, flashes):
        onsets = [f[0] for f in flashes]
        ts, x = self.engine.window(min(onsets) + P.BASELINE[0] - 0.05)
        if ts is None or not len(ts):
            return None, None, None
        eps = P.epochs(ts, x, onsets, self.engine.reject_uv)
        rows = [(e.ravel(), int(f[2]), f[1]) for e, f in zip(eps, flashes) if e is not None]
        if not rows:
            return None, None, None
        return np.array([r[0] for r in rows]), np.array([r[1] for r in rows]), [r[2] for r in rows]

    def _retrain(self):
        with self.lock:
            pool = list(self.pool)
        n_target = sum(int(p["y"].sum()) for p in pool)
        if len(pool) < self.MIN_TRIALS or n_target < self.MIN_TARGET_EPOCHS:
            return
        X = np.vstack([p["X"] for p in pool])
        y = np.concatenate([p["y"] for p in pool])
        labels = list(self.engine.labels)
        m = P.LinearP300.fit(X, y, labels, filters=self.engine.filter.spec if self.engine.filter else None,
                             meta={"trained_on": "live session", "session_id": self.adaptive_session,
                                                  "participant_id": self.adaptive_participant,
                                                  "simulated": bool(self.engine.info.get("simulated")),
                                                  "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                   "n_trials": len(pool), "n_epochs": int(len(y))})
        ev = []
        for p in pool:
            sums, _, _ = P.option_evidence(p["options"], m.score(p["X"]), sorted(set(p["options"])))
            ev.append((sums, p["true"]))
        m.temperature = P.fit_temperature(ev)
        with self.lock:
            self.adaptive = m
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(self.adaptive_session))
            m.save(os.path.join(self.log_dir, f"p300_adaptive_{safe}.npz"))
        except Exception:
            pass

    def erp(self):
        """Average response (40 ms bins) to target vs non-target flashes over this session's finished trials."""
        with self.lock:
            pool = list(self.pool)
        labels = list(self.engine.labels)
        if not pool or not labels:
            return {"labels": labels, "n_target": 0, "n_nontarget": 0}
        X = np.vstack([p["X"] for p in pool])
        y = np.concatenate([p["y"] for p in pool])
        nb = P.n_bins()
        X = X.reshape(len(X), len(labels), nb)
        times = [round(P.TMIN + (b + 0.5) * P.BIN_MS / 1000, 3) for b in range(nb)]
        out = {"labels": labels, "times": times, "n_target": int(y.sum()), "n_nontarget": int((1 - y).sum()),
               "trials": len(pool), "session_id": self.adaptive_session}
        if y.sum():
            out["target"] = np.round(X[y == 1].mean(axis=0), 2).tolist()
        if (1 - y).sum():
            out["nontarget"] = np.round(X[y == 0].mean(axis=0), 2).tolist()
        if y.sum() and (1 - y).sum():
            b0 = int((P.HEURISTIC_WINDOW[0] - P.TMIN) * 1000 / P.BIN_MS)
            b1 = int((P.HEURISTIC_WINDOW[1] - P.TMIN) * 1000 / P.BIN_MS)
            d = X[y == 1][:, :, b0:b1].mean(axis=(0, 2)) - X[y == 0][:, :, b0:b1].mean(axis=(0, 2))
            out["p300_map"] = [round(float(v), 2) for v in d]   # target minus non-target, 250-450 ms, per channel
        return out

    def _pick(self):
        labels = list(self.engine.labels)
        if self.mode in ("auto", "model") and self.model is not None and not self.model_error and self.filters_match():
            if not labels:          # no EEG stream yet: nothing to score, but status must not crash
                return "model", self.model, None
            try:
                return "model", self.model, self.model.reorder_for(labels)
            except ValueError as exc:
                self.model_error = str(exc)
        if self.mode in ("auto", "adaptive") and self.adaptive is not None:
            return "adaptive", self.adaptive, list(range(len(labels)))
        return "heuristic", None, None

    def status(self):
        with self.lock:
            kind, _, _ = self._pick()
            mismatch = None if self.filters_match() else (
                "paused: it was trained with different filters ("
                + "; ".join(F.summary(self.model.filters)) + ") than the running chain ("
                + "; ".join(F.summary(self.engine.filter.spec if self.engine.filter else self.engine.filter_spec))
                + "). Switch back to its chain, or calibrate a new model with these filters.")
            return {"mode": self.mode, "active": kind, "model_file": self.model_path, "model_error": self.model_error,
                    "model_filter_mismatch": mismatch, "filters_reset_note": self.filters_reset_note,
                    "model_source": self.model_source, "model_info": self.model_info,
                    "adaptive_trials": len(self.pool),
                    "adaptive_ready": self.adaptive is not None,
                    "adaptive_needs": f"{self.MIN_TRIALS} valid trials and {self.MIN_TARGET_EPOCHS} target epochs"}

    # ---- scoring ----
    def score_trial(self, session_id, trial_id, option_ids, partial=False):
        """Return a prediction dict for one trial. Uses only onsets + option ids, never is_target of this trial."""
        with self.lock:
            t = self.trials.get((session_id, trial_id))
            flashes = [(f[0], f[1]) for f in t["flashes"]] if t else []   # <- labels dropped here
            kind, model, order = self._pick()
        if not flashes:
            return {"error": "no flash markers reached the bridge for this trial (was the bridge connected?)"}
        onsets = [f[0] for f in flashes]
        ts, x = self.engine.window(min(onsets) + P.BASELINE[0] - 0.05)
        if ts is None or not len(ts):
            return {"error": "no EEG data for this trial"}
        eps = P.epochs(ts, x, onsets, self.engine.reject_uv)
        labels = list(self.engine.labels)
        scores = []
        for e in eps:
            if e is None:
                scores.append(None)
            elif kind == "heuristic":
                scores.append(P.heuristic_score(e, labels))
            else:
                scores.append(float(model.score(e[order].ravel())[0]))
        n_ok = sum(s is not None for s in scores)
        if not partial:
            with self.lock:
                if t is not None:
                    t["predicted"] = True
        sums, means, counts = P.option_evidence([f[1] for f in flashes], scores, option_ids)
        if kind == "heuristic":
            probs = P.softmax(means, 0.5)          # 2 uV difference ~ e-fold; uncalibrated
            evidence = means
        else:
            probs = P.softmax(sums, model.temperature)
            evidence = sums
        best = max(probs, key=probs.get) if n_ok else None
        return {
            "best_option_id": best, "confidence": probs.get(best) if best else None,
            "p300": {k: round(v, 4) for k, v in evidence.items()},
            "probabilities": probs if n_ok else {}, "probabilities_calibrated": False,
            "scorer": kind, "n_epochs": len(scores), "n_epochs_used": n_ok, "n_epochs_rejected": len(scores) - n_ok,
            "epochs_per_option": counts, "partial": partial,
        }

    def wait_and_score(self, session_id, trial_id, option_ids, deadline_s=5.0):
        """Wait until the EEG covers the last epoch (or the deadline), then score."""
        t_end = time.time() + deadline_s
        while time.time() < t_end:
            with self.lock:
                t = self.trials.get((session_id, trial_id))
                last_onset = max((f[0] for f in t["flashes"]), default=None) if t else None
            last = self.engine.last_ts()
            if last_onset is not None and last is not None and last >= last_onset + P.TMAX + 0.02:
                break
            time.sleep(0.05)
        if os.environ.get("CORTEXT_DEBUG"):
            print(f"[debug] {time.time():.2f} waited {deadline_s - (t_end - time.time()):.2f}s last_onset={last_onset} last_eeg={last} "
                  f"lag={None if last is None or last_onset is None else round(last - last_onset, 3)}", flush=True)
        return self.score_trial(session_id, trial_id, option_ids)

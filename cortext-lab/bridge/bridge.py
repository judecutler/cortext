#!/usr/bin/env python3
"""CorText Lab bridge: serves the web app and relays flash markers to LSL / CSV.

    python bridge/bridge.py            # serves http://localhost:8000 and ws://localhost:8000/ws

* Browser -> WebSocket (/ws) -> this script -> LSL string-marker outlet (if pylsl is installed)
* Every marker is ALSO appended to a CSV in bridge/logs/ (a backup you can always use;
  disable with --no-csv). If pylsl or liblsl is missing the bridge keeps working in CSV-only mode.
* Clock sync: the page runs ping-pong rounds and reports (bridge_time - page_time). Markers carry the
  page's performance.now() timestamp, and the bridge converts it to an LSL timestamp, so network
  / event-loop delay does not shift the marker in the EEG time base.
* Storage (see datasets/cortext-p300/cortext_store.py): the bridge lists, uploads to Hugging Face and really deletes
  sessions on request from the web app (/api/sessions, /api/session/delete, /api/storage*).
* EEG (see eeg_engine.py): finds the headset's LSL stream (eeg/stream_eeg.py), records it per session to
  logs/eeg_<session>.csv, reports signal quality / band power, and serves a live P300 scorer on ws://.../predict
  that the app uses when Predictor = "Live EEG".
* Trained models (see models.py): runs the model in cortext-lab/models/active/ (or --p300-model FILE); the Agent page
  lists models/library/ and this laptop's calibrations and switches the running model live (/api/models*).
* Filter chains (see filters.py): the layers between the headset and the model. Runs the chain in
  cortext-lab/filters/active/ (or --filters FILE, else the built-in 0.5-20 Hz default); the Agent page edits, previews
  and switches it live (/api/filters*). A model is only used while the chain it was trained with runs.

Standard library only; `pip install pylsl` is optional (free, no account).
"""
import argparse
import csv
import json
import os
import socket
import sys
import threading
import time
from functools import partial
from pathlib import Path
import shutil
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsproto  # noqa: E402
import eeg_engine  # noqa: E402
import voice  # noqa: E402  (TTS + LLM narration for the demo; all optional)
import models  # noqa: E402  (trained P300 models: models/library + models/active)
import filters  # noqa: E402  (filter chains: filters/library + filters/active)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets", "cortext-p300"))
import cortext_store as store  # noqa: E402  (stdlib only; huggingface_hub is imported lazily)

try:
    import build_session  # noqa: E402  (needs numpy + scipy)
    BUILD_ERROR = None
except Exception as exc:
    build_session = None
    BUILD_ERROR = repr(exc)

try:  # optional dependency
    import pylsl  # type: ignore
    LSL_IMPORT_ERROR = None
except Exception as exc:  # ImportError, or OSError when liblsl is missing
    pylsl = None
    LSL_IMPORT_ERROR = repr(exc)

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(os.path.dirname(HERE), "web")
LOG_DIR = os.environ.get("CORTEXT_BRIDGE_LOG_DIR") or os.path.join(HERE, "logs")   # tests point this at a temp folder

CSV_COLUMNS = [
    "recv_wall_iso", "recv_perf_ms", "page_t_ms", "bridge_t_ms", "est_transit_ms", "lsl_timestamp",
    "session_id", "seq", "event", "trial_id", "trial_index", "option_id", "option_position",
    "category", "is_target", "repeat_index", "flash_index", "true_answer_id", "frame_ms", "wall_ms",
    "extra",
]


def perf_ms() -> float:
    return time.perf_counter() * 1000.0


class State:
    """Process-wide state shared by all connections."""

    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.outlet = None
        self.outlet_name = None
        self.markers_received = 0
        self.last_seq = None
        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None
        self.clock_offset_ms = None  # bridge_time - page_time
        self.clock_rtt_ms = None
        self.eeg = None  # EEGEngine, created in main() when pylsl is available
        self.clients = set()
        self.datasets = {}  # session_id -> last build result / error
        self.build_lock = threading.Lock()
        self.storage_lock = threading.RLock()   # one storage operation (upload / download / delete) at a time
        self.sessions_seen = {}  # session_id -> {"last": wall time of its last marker, "ended": bool}
        self.raw_rec = None      # the free raw EEG recording in progress (raw_start / raw_stop)
        if pylsl and not args.no_lsl:
            info = pylsl.StreamInfo(args.lsl_name, "Markers", 1, 0, "string", f"cortext-lab-{socket.gethostname()}")
            self.outlet = pylsl.StreamOutlet(info)
            self.outlet_name = args.lsl_name

    # ---- CSV ----
    def open_csv(self, session_id):
        if self.args.no_csv:
            return None
        os.makedirs(LOG_DIR, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(session_id or "nosession"))
        path = os.path.join(LOG_DIR, f"markers_{safe}_{stamp}.csv")
        with self.lock:
            if self.csv_file:
                self.csv_file.close()
            self.csv_file = open(path, "w", newline="", encoding="utf-8")
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            self.csv_writer.writeheader()
            self.csv_file.flush()
            self.csv_path = path
        return path

    def handle_marker(self, msg, offset=None):
        """offset: this connection's page->bridge clock offset (each browser tab has its own clock origin)."""
        recv = perf_ms()
        page_t = msg.get("t_ms")
        if page_t is not None and offset is not None:
            bridge_t = page_t + offset
        else:
            bridge_t = recv  # no sync yet: best effort
        transit = recv - bridge_t
        lsl_ts = None
        if pylsl is not None:  # the marker's time in the LSL clock (also what the EEG timestamps use)
            lsl_ts = pylsl.local_clock() - transit / 1000.0
        if self.eeg is not None and self.eeg.scorer is not None:
            self.eeg.scorer.on_marker(msg, lsl_ts)
        sid = msg.get("session_id")
        if sid and msg.get("event") != "timing_flash":
            self.sessions_seen[sid] = {"last": time.time(), "ended": msg.get("event") == "session_end"}
        if msg.get("event") == "session_end" and msg.get("session_id"):
            self.on_session_end(msg.get("session_id"))
        if self.outlet is not None:
            payload = format_marker(msg, self.args.format)
            self.outlet.push_sample([payload], lsl_ts)
        row = {
            "recv_wall_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "recv_perf_ms": round(recv, 3),
            "page_t_ms": page_t,
            "bridge_t_ms": round(bridge_t, 3),
            "est_transit_ms": round(transit, 3),
            "lsl_timestamp": None if lsl_ts is None else round(lsl_ts, 6),
        }
        row.update({k: msg.get(k) for k in CSV_COLUMNS if k in msg})
        extra = msg.get("extra")
        row["extra"] = json.dumps(extra) if isinstance(extra, (dict, list)) else extra
        with self.lock:
            self.markers_received += 1
            self.last_seq = msg.get("seq")
            if self.csv_writer:
                self.csv_writer.writerow(row)
                self.csv_file.flush()

    # ---- sessions on this machine: inventory and deletion ----
    def session_is_running(self, session_id):
        seen = self.sessions_seen.get(session_id)
        return bool(seen and not seen["ended"] and time.time() - seen["last"] < 20)

    def release_session(self, session_id):
        """Close this session's open log files (Windows refuses to delete a file that is still open)."""
        safe = store.safe_id(session_id)
        with self.lock:
            if self.csv_file and self.csv_path and os.path.basename(self.csv_path).startswith(f"markers_{safe}_"):
                try:
                    self.csv_file.close()
                except OSError:
                    pass
                self.csv_file = self.csv_writer = self.csv_path = None
        if self.eeg is not None:
            self.eeg.stop_session(session_id)

    def inventory(self):
        """Everything this machine holds, one row per session: dataset folder and/or raw logs (the browser adds its own copy)."""
        rows, logs = {}, store.scan_logs(LOG_DIR)
        for r in store.scan_local():
            rows[store.safe_id(r["session_id"])] = {**r, "has_dataset": True}
        for safe, info in logs.items():
            row = rows.setdefault(safe, {"session_id": safe, "has_dataset": False, "simulated": None, "uploaded": False})
            row.update(has_logs=True, log_files=info["files"], log_mb=round(info["bytes"] / 1e6, 2))
        for safe, row in rows.items():
            row.setdefault("has_logs", False)
            row["running"] = self.session_is_running(row["session_id"])
            row["timing_test"] = safe.startswith("timing-test")
        return sorted(rows.values(), key=lambda r: str(r.get("recorded") or r["session_id"]), reverse=True)

    def delete_session(self, session_id, remote=False, purge_remote=True):
        if self.session_is_running(session_id):
            return {"ok": False, "error": "This session is still running. End it first.", "status": 409}
        with self.storage_lock, self.build_lock:
            self.release_session(session_id)
            report = store.delete_session(session_id, log_dir=LOG_DIR, remote=remote, purge_remote=purge_remote)
        self.datasets.pop(session_id, None)
        self.sessions_seen.pop(session_id, None)
        return report

    # ---- dataset (pipeline/build_session.py) ----
    def build_dataset(self, session_id, session=None, trials=None):
        if build_session is None:
            r = {"ok": False, "error": f"dataset builder unavailable: {BUILD_ERROR} (pip install numpy scipy)"}
        else:
            with self.build_lock:
                try:
                    stream = None
                    if self.eeg and self.eeg.info.get("found"):
                        stream = {k: v for k, v in self.eeg.info.items() if k not in ("quality", "bands", "recording", "scorer")}
                    res = build_session.build(session_id, trials=trials, session=session, log_dir=LOG_DIR,
                                              stream_info=stream if (session or {}).get("eeg") is None else None)
                    r = {"ok": True, **res, "with_app_trials": bool(trials), "built_at": time.strftime("%H:%M:%S")}
                except Exception as exc:
                    r = {"ok": False, "error": str(exc)}
        self.datasets[session_id] = r
        return r

    def on_session_end(self, session_id):
        def later():
            time.sleep(2.5)  # let the last EEG samples arrive
            if not self.datasets.get(session_id, {}).get("with_app_trials"):
                self.build_dataset(session_id)
        threading.Thread(target=later, daemon=True).start()

    # ---- free raw EEG recording (no stimuli, no session): Experiment -> Data storage -> "Record raw EEG" ----
    def raw_status(self):
        r = self.raw_rec
        if not r:
            return {"recording": False}
        eng = self.eeg
        app_took_over = bool(eng and eng.session_id != r["id"])
        return {"recording": True, "id": r["id"], "started": r["started_iso"], "seconds": round(time.time() - r["t0"], 1),
                "samples": eng.samples_recorded if (eng and not app_took_over) else None, "interrupted": app_took_over}

    def raw_start(self, participant="", note=""):
        eng = self.eeg
        if eng is None:
            raise store.StorageError("The bridge has no EEG engine (pip install pylsl), so it cannot record.")
        if not eng.info.get("found"):
            raise store.StorageError("No headset stream is reaching the bridge. Start eeg/stream_eeg.py (Unicorn) or eeg/stream_muse.py (Muse) first.")
        if self.raw_rec:
            raise store.StorageError("A raw recording is already running. Stop it first.")
        if eng.session_id:
            raise store.StorageError(f"Session {eng.session_id} is recording; finish it before starting a raw recording.")
        if participant and not all(c.isalnum() or c in "-_" for c in participant):
            raise store.StorageError("Participant must be a code such as P01, never a name.")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        rid = f"rawrec-{stamp}"
        eng.start_session(rid)
        self.raw_rec = {"id": rid, "stamp": stamp, "t0": time.time(), "started_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "participant": participant, "note": note, "offsets": [time.time() - pylsl.local_clock()],
                        "device": eng.info.get("device"), "simulated": bool(eng.info.get("simulated")),
                        "srate": eng.info.get("nominal_srate"), "labels": list(eng.labels)}
        return self.raw_status()

    def raw_stop(self, upload=True):
        r = self.raw_rec
        if not r:
            raise store.StorageError("No raw recording is running.")
        eng = self.eeg
        r["offsets"].append(time.time() - pylsl.local_clock())
        if eng.session_id == r["id"]:
            eng.stop_session(r["id"])
        files = eng._find_files(r["id"])
        self.raw_rec = None
        if not files:
            raise store.StorageError("Nothing was recorded (the headset stream may have dropped).")
        out_dir = store.data_root(create=True) / store.RAW_EEG_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        offset = sum(r["offsets"]) / len(r["offsets"])   # wall clock - LSL clock (drifts < 1 ms over a recording)
        name = f"{r['stamp']}_{r['participant'] or 'unknown'}{'_SIM' if r['simulated'] else ''}_raw.csv.gz"
        out = out_dir / name
        total, labels, first = 0, [], None
        import gzip as _gz
        parts = []
        for i, f in enumerate(files):   # usually one file; more if the headset reconnected mid-recording
            part = out if len(files) == 1 else out_dir / f".part{i}_{name}"
            n, labels, f0 = store.export_raw_recording(f, part, unix_offset=offset)
            total += n
            first = first if first is not None else f0
            parts.append(part)
        if len(parts) > 1:   # concatenate the gzip members (a valid .gz), keeping one header
            with _gz.open(out, "wt", encoding="utf-8", newline="") as dst:
                for i, part in enumerate(parts):
                    with _gz.open(part, "rt", encoding="utf-8") as src:
                        head = src.readline()
                        if i == 0:
                            dst.write(head)
                        shutil.copyfileobj(src, dst)
                    os.remove(part)
        for f in files:   # the CSV lives on as the .csv.gz; remove the logs copy so it isn't listed as an orphan session
            for g in (f, f[:-4] + ".json"):
                try:
                    os.remove(g)
                except OSError:
                    pass
        meta = {"kind": "free raw recording (no stimuli, no markers)", "device": r["device"], "simulated": r["simulated"],
                "srate_nominal": r["srate"], "channels": labels, "samples": total,
                "duration_s": round(time.time() - r["t0"], 1), "started": r["started_iso"],
                "unix_start": first, "clock": "unix_time = lsl_time + (wall clock - LSL clock) measured at start and stop",
                "units": "microvolts as sent by the headset; raw (DC offset and mains hum included)"}
        res = {"ok": True, "local_path": store.display_path(out), "samples": total, "duration_s": meta["duration_s"], "uploaded": False}
        if upload:
            if r["simulated"]:
                res["upload_error"] = "Simulated EEG is not uploaded (kept on this computer only)."
            else:
                try:
                    with self.storage_lock:
                        up = store.push_raw_eeg(out, note=r["note"], participant=r["participant"], extra=meta, name=name,
                                                stamp=r["stamp"])
                    res.update(uploaded=True, url=up["url"], path_in_repo=up["path_in_repo"])
                except store.StorageError as exc:
                    res["upload_error"] = str(exc)
        with open(str(out) + ".json", "w", encoding="utf-8") as fh:
            json.dump({**meta, "participant_id": r["participant"] or None, "note": r["note"],
                       "uploaded": res.get("path_in_repo")}, fh, indent=1)
        return res

    def raw_files(self):
        d = store.data_root() / store.RAW_EEG_DIR
        try:
            names = sorted((p for p in d.iterdir() if p.is_file() and not p.name.endswith(".json") and not p.name.startswith(".")),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            return []
        out = []
        for p in names[:50]:
            info = {}
            try:
                info = json.loads((Path(str(p) + ".json")).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            out.append({"name": p.name, "size_mb": round(p.stat().st_size / 1e6, 2), "uploaded": info.get("uploaded"),
                        "duration_s": info.get("duration_s"), "simulated": info.get("simulated", False)})
        return out

    def raw_push_local(self, name):
        d = store.data_root() / store.RAW_EEG_DIR
        p = d / os.path.basename(str(name))
        if not p.is_file() or p.name.endswith(".json"):
            raise store.StorageError(f"{name} is not in {store.display_path(d)}.")
        meta = {}
        try:
            meta = json.loads(Path(str(p) + ".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        if meta.get("simulated"):
            raise store.StorageError("Simulated EEG is not uploaded.")
        stamp = p.name.split("_", 1)[0] if p.name[:8].isdigit() else None
        with self.storage_lock:
            up = store.push_raw_eeg(p, note=meta.get("note", ""), participant=meta.get("participant_id") or "",
                                    extra={k: v for k, v in meta.items() if k not in ("note", "participant_id", "uploaded")},
                                    name=p.name.split("_", 1)[1] if stamp else p.name, stamp=stamp)
        meta["uploaded"] = up["path_in_repo"]
        with open(str(p) + ".json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=1)
        return {"ok": True, **up}

    def raw_upload_bytes(self, filename, data, note="", participant=""):
        """A file picked in the browser (recorded with another program): keep a copy here, then upload it untouched."""
        d = store.data_root(create=True) / store.RAW_EEG_DIR
        d.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        local = d / store.raw_eeg_name(filename, stamp)
        if local.suffix.lower() not in store.RAW_EEG_EXTENSIONS:
            raise store.StorageError(f"{filename}: not an EEG file type this folder takes ({', '.join(sorted(store.RAW_EEG_EXTENSIONS))}).")
        local.write_bytes(data)
        meta = {"kind": "uploaded file (recorded elsewhere)", "original_name": filename, "participant_id": participant or None,
                "note": note, "uploaded": None}
        try:
            with self.storage_lock:
                up = store.push_raw_eeg(local, note=note, participant=participant, name=filename, stamp=stamp)
            meta["uploaded"] = up["path_in_repo"]
            return {"ok": True, **up, "local_path": store.display_path(local)}
        finally:
            with open(str(local) + ".json", "w", encoding="utf-8") as fh:
                json.dump(meta, fh, indent=1)

    def status(self):
        consumers = None
        if self.outlet is not None:
            try:
                consumers = bool(self.outlet.have_consumers())
            except Exception:
                consumers = None
        return {
            "type": "stats",
            "bridge_time_ms": perf_ms(),
            "lsl_available": self.outlet is not None,
            "lsl_error": None if (pylsl or self.args.no_lsl) else LSL_IMPORT_ERROR,
            "lsl_name": self.outlet_name,
            "lsl_consumers": consumers,
            "csv_path": self.csv_path,
            "markers_received": self.markers_received,
            "last_seq": self.last_seq,
            "clock_offset_ms": self.clock_offset_ms,
            "clock_rtt_ms": self.clock_rtt_ms,
            "eeg": self.eeg.status() if self.eeg else {"found": False, "error": "pylsl not installed: pip install pylsl"},
            "raw_recording": self.raw_status(),
        }


def format_marker(msg, fmt):
    if fmt == "compact":
        keys = ["event", "session_id", "trial_id", "option_id", "category", "is_target", "repeat_index", "flash_index"]
        return "|".join("" if msg.get(k) is None else str(msg.get(k)) for k in keys)
    keep = {k: msg.get(k) for k in CSV_COLUMNS if k in msg and k not in ("extra",)}
    keep["extra"] = msg.get("extra")
    return json.dumps(keep, separators=(",", ":"))


class Handler(SimpleHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    state: State = None  # set in main()

    def log_message(self, fmt, *args):
        if self.state.args.verbose:
            super().log_message(fmt, *args)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        path = urlparse(self.path).path
        q = parse_qs(urlparse(self.path).query)
        if self.headers.get("Upgrade", "").lower() == "websocket":
            if path == "/ws":
                return self.serve_ws(self.marker_messages)
            if path == "/predict":
                return self.serve_ws(self.predict_messages)
        if path == "/api/status":
            return self.send_json(self.state.status())
        if path == "/api/eeg/trace":
            eng = self.state.eeg
            secs = min(30.0, float(q.get("seconds", ["5"])[0]))
            return self.send_json(eng.trace(secs) if eng else {"error": "pylsl not installed"})
        if path == "/api/eeg/erp":
            eng = self.state.eeg
            return self.send_json(eng.scorer.erp() if (eng and eng.scorer) else {"labels": []})
        if path == "/api/dataset":
            return self.send_json(self.state.datasets.get(q.get("session_id", [""])[0]) or {"ok": None})
        if path == "/api/eeg/session.csv":
            return self.send_session_eeg(q.get("session_id", [""])[0])
        if path == "/api/sessions":
            return self.send_json({"data_root": store.display_path(store.data_root()), "sessions": self.state.inventory()})
        if path == "/api/storage/raw":
            return self.send_json({"ok": True, "folder": store.RAW_EEG_DIR, "recording": self.state.raw_status(),
                                   "files": self.state.raw_files()})
        if path == "/api/storage":
            return self.send_json(self.storage_info(check=q.get("check", ["0"])[0] == "1"))
        if path == "/api/voice/status":
            return self.send_json({**voice.status(check=q.get("check", ["0"])[0] == "1"), "default_persona": voice.DEFAULT_PERSONA, "rules": voice.RULES})
        if path == "/api/tts":
            return self.send_tts(q)
        if path == "/api/models":
            return self.send_json(self.models_info())
        if path == "/api/filters":
            return self.send_json(self.filters_info())
        if path in ("/setup", "/agent", "/experiment"):   # old bookmarks: the app routes by #hash
            self.send_response(302)
            self.send_header("Location", "/#" + path)
            self.end_headers()
            return None
        return super().do_GET()

    # ---- trained P300 models (Setup page: list, pick the one to run, add files) ----
    def _scorer(self):
        eng = self.state.eeg
        return eng.scorer if eng else None

    def models_info(self):
        eng = self.state.eeg
        labels = list(eng.labels) if eng and eng.labels else None
        info = models.listing(LOG_DIR, labels, self._running_filter_sig())
        sc = self._scorer()
        info["running"] = sc.status() if sc else None
        info["scoring_available"] = sc is not None
        info["live_channels"] = labels
        return info

    def models_post(self, path, body):
        sc = self._scorer()
        if path == "/api/models/select":
            mid = body.get("id")
            if mid:
                models.path_for(mid, LOG_DIR)                   # unknown id: fail before touching any filters
            follow = None
            if mid and body.get("follow_filters", True):
                # a model only works with the filters it was trained with: switch to its chain first
                d = models.describe(models.path_for(mid, LOG_DIR))
                if d.get("filters") and d.get("filter_signature") != self._running_filter_sig():
                    try:
                        follow = self._apply_filters(d["filters"], f"chain of model {d['file']}")
                    except filters.FilterError as exc:
                        return {"ok": False, "error": f"this model's filters can't run on this headset: {exc}"}
            new = models.set_active(mid, LOG_DIR)
            note = f"Filters switched to the chain this model was trained with: {'; '.join(follow['summary'])}." if follow else None
            if sc is not None:
                try:
                    sc.set_model(new, "Setup page" if new else "none (picked on the Setup page)")
                except ValueError as exc:
                    return {"ok": False, "error": str(exc)}
            else:
                note = "saved as the active model; the bridge cannot score here (needs pylsl + numpy), it will be used once it can"
            return {"ok": True, "active": os.path.basename(new) if new else None, "note": note,
                    "filters_switched": bool(follow), **self.models_info()}
        if path == "/api/models/upload":
            import base64
            try:
                data = base64.b64decode(body.get("data") or "", validate=True)
            except ValueError:
                return {"ok": False, "error": "bad file data"}
            mid = models.add_to_library(data, body.get("filename"))
            return {"ok": True, "id": mid, **self.models_info()}
        if path == "/api/models/save-calibration":
            eng = self.state.eeg
            mid = models.save_calibration(body.get("id"), LOG_DIR, list(eng.labels) if eng else None)
            return {"ok": True, "id": mid, **self.models_info()}
        return None

    # ---- filter chains (Agent page: see, edit, preview and switch the layers between headset and model) ----
    def _running_filter_sig(self):
        eng = self.state.eeg
        if eng is not None:
            return eng.running_signature
        spec, _, _ = filters.active_chain()
        return filters.signature(spec)

    def _apply_filters(self, spec, source):
        """Make `spec` the running chain: validate on the live stream, write filters/active/, switch the engine."""
        eng = self.state.eeg
        spec = filters.normalize(spec)
        if eng is not None:
            eng.set_filters(spec, source)       # raises FilterError if it can't run here; nothing changes then
        filters.set_active(spec)
        return {"spec": spec, "summary": filters.summary(spec)}

    def filters_info(self):
        eng = self.state.eeg
        info = filters.listing(self._running_filter_sig())
        info["running"] = eng.filter_status() if eng else {
            **filters.describe(filters.active_chain()[0]), "source": "filters/active (bridge has no EEG engine: pip install pylsl)",
            "error": None}
        info["live"] = bool(eng and eng.info.get("found"))
        info["labels"] = list(eng.labels) if eng and eng.labels else []
        info["epoching"] = {"tmin_s": eeg_engine.P.TMIN, "tmax_s": eeg_engine.P.TMAX, "baseline_s": list(eeg_engine.P.BASELINE),
                            "bin_ms": eeg_engine.P.BIN_MS, "heuristic_window_s": list(eeg_engine.P.HEURISTIC_WINDOW),
                            "parietal": eeg_engine.P.PARIETAL} if eeg_engine.P else None
        if build_session is not None:
            pr = build_session.PARAMS
            info["offline_cleaning"] = {"zero_phase": True, "bandpass_hz": pr.get("bandpass_hz"), "bandpass_order": pr.get("bandpass_order"),
                                        "notch_q": pr.get("notch_q"), "artifact_uv": pr.get("artifact_uv"), "params": pr}
        sc = self._scorer()
        info["model"] = sc.status() if sc else None
        return info

    def filters_post(self, path, body):
        if path == "/api/filters/apply":
            r = self._apply_filters(body.get("spec"), body.get("source") or "Agent page")
            return {"ok": True, **r, **self.filters_info()}
        if path == "/api/filters/select":
            r = self._apply_filters(filters.spec_for(body.get("id")), f"Agent page ({body.get('id')})")
            return {"ok": True, **r, **self.filters_info()}
        if path == "/api/filters/save":
            cid = filters.save_to_library(body.get("spec"), body.get("name"), overwrite=bool(body.get("overwrite")))
            return {"ok": True, "id": cid, **self.filters_info()}
        if path == "/api/filters/delete":
            spec = filters.delete_from_library(body.get("id"))
            return {"ok": True, "deleted": spec, **self.filters_info()}
        if path == "/api/filters/preview":
            eng = self.state.eeg
            if eng is None or eeg_engine.np is None:
                return {"ok": False, "error": "the bridge cannot filter here (needs pylsl, numpy and scipy)"}
            ts, raw, labels, srate = eng.raw_window(min(30.0, float(body.get("warm_s", 20))))
            if ts is None or len(ts) < srate * 2:
                return {"ok": False, "error": "no EEG yet: connect the headset (or run eeg/fake_eeg.py) to preview"}
            return filters.preview(body.get("spec"), ts, raw, labels, srate, body.get("channel"),
                                   show_s=min(10.0, float(body.get("seconds", 4))))
        return None

    def send_tts(self, q):
        try:
            data, ctype, eng = voice.synthesize(q.get("text", [""])[0], voice=q.get("voice", [None])[0],
                                                rate=q.get("rate", ["1"])[0], engine=q.get("engine", [None])[0])
        except voice.VoiceError as exc:
            return self.send_json({"ok": False, "error": str(exc)}, 503)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-TTS-Engine", eng)
        self.end_headers()
        self.wfile.write(data)

    def storage_info(self, check=False):
        rows = store.scan_local()
        info = {"data_root": store.display_path(store.data_root()), "hf": store.hf_status(check=check),
                "local_sessions": sum(not r["simulated"] for r in rows), "simulated_sessions": sum(r["simulated"] for r in rows),
                "size_mb": round(sum(r["size_mb"] or 0 for r in rows), 1), "pending_upload": store.pending_uploads()}
        if info["hf"].get("remote_sessions") is not None:
            local = {store.safe_id(r["session_id"]) for r in rows}
            info["remote_only"] = [x for x in info["hf"]["remote_sessions"] if x not in local]
        return info

    def same_origin(self):
        """Refuse cross-site POSTs (a random web page must not be able to delete or upload sessions)."""
        origin = self.headers.get("Origin")
        if not origin:
            return True
        host = self.headers.get("Host", "")
        return urlparse(origin).netloc == host

    def do_POST(self):
        path = urlparse(self.path).path
        if not self.same_origin():
            return self.send_json({"ok": False, "error": "cross-site request refused"}, 403)
        if path == "/api/storage/raw-upload":   # body = the file's bytes; name/note/participant in the query string
            q = parse_qs(urlparse(self.path).query)
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return self.send_json({"ok": False, "error": "empty file"}, 400)
            if n > store.RAW_EEG_MAX_MB * 1_000_000:
                self.close_connection = True
                return self.send_json({"ok": False, "error": f"file is larger than {store.RAW_EEG_MAX_MB} MB"}, 413)
            data = self.rfile.read(n)
            try:
                return self.send_json(self.state.raw_upload_bytes(q.get("name", ["recording"])[0], data,
                                                                  note=q.get("note", [""])[0], participant=q.get("participant", [""])[0]))
            except store.StorageError as exc:
                return self.send_json({"ok": False, "error": str(exc)})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode() or "{}") if n else {}
        except ValueError:
            return self.send_json({"ok": False, "error": "bad JSON"}, 400)
        sid = body.get("session_id")
        state = self.state
        if path.startswith("/api/filters/"):
            try:
                r = self.filters_post(path, body)
            except filters.FilterError as exc:
                return self.send_json({"ok": False, "error": str(exc)})
            except OSError as exc:
                return self.send_json({"ok": False, "error": f"could not write the filters folder: {exc}"})
            if r is not None:
                return self.send_json(r)
        if path.startswith("/api/models/"):
            try:
                r = self.models_post(path, body)
            except (models.ModelError, filters.FilterError) as exc:
                return self.send_json({"ok": False, "error": str(exc)})
            except OSError as exc:
                return self.send_json({"ok": False, "error": f"could not write the models folder: {exc}"})
            if r is not None:
                return self.send_json(r)
        if path == "/api/narrate":
            try:
                return self.send_json(voice.narrate(body.get("provider"), body.get("facts") or {}, timeout=float(body.get("timeout", 6)), context=body.get("context")))
            except voice.VoiceError as exc:
                return self.send_json({"ok": False, "error": str(exc)})
        try:
            if path == "/api/dataset/build" and sid:
                return self.send_json(state.build_dataset(sid, body.get("session"), body.get("trials")))
            if path == "/api/dataset/push" and sid:
                r = state.datasets.get(sid)
                if not r or not r.get("ok"):
                    r = state.build_dataset(sid, body.get("session"), body.get("trials"))
                if not r.get("ok"):
                    return self.send_json(r)
                with state.storage_lock:
                    up = store.push_session(sid, allow_simulated=bool(body.get("allow_simulated")))
                return self.send_json({"ok": True, "url": up["url"], "path_in_repo": up["path_in_repo"], "relative_path": r.get("relative_path"),
                                       "voltages": up.get("voltages")})
            if path == "/api/session/delete" and sid:
                rep = state.delete_session(sid, remote=bool(body.get("remote")), purge_remote=body.get("purge_remote", True))
                return self.send_json(rep, rep.pop("status", 200))
            if path == "/api/session/bucket" and sid:
                with state.storage_lock:
                    return self.send_json({"ok": True, "bucket": store.set_bucket(sid, body.get("bucket", ""))})
            if path == "/api/storage/push-pending":
                with state.storage_lock:
                    res = store.push_pending()
                return self.send_json({"ok": all(x["ok"] for x in res), "results": res})
            if path == "/api/storage/pull":
                with state.storage_lock:
                    res = store.pull(body.get("sessions") or None, profile=body.get("profile", "epochs"))
                return self.send_json({"ok": True, **res})
            if path == "/api/storage/raw/start":
                return self.send_json({"ok": True, **state.raw_start(str(body.get("participant") or "").strip(), str(body.get("note") or ""))})
            if path == "/api/storage/raw/stop":
                return self.send_json(state.raw_stop(upload=bool(body.get("upload", True))))
            if path == "/api/storage/raw/push":
                return self.send_json(state.raw_push_local(body.get("name", "")))
            if path == "/api/storage/init":
                with state.storage_lock:
                    return self.send_json({"ok": True, **store.init_remote()})
        except store.StorageError as exc:
            return self.send_json({"ok": False, "error": str(exc)})
        return self.send_json({"ok": False, "error": "unknown endpoint"}, 404)

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_session_eeg(self, session_id):
        eng = self.state.eeg
        files = eng.session_files(session_id) if (eng and session_id) else []
        if not files:
            return self.send_json({"error": "no EEG recorded for this session"}, 404)
        # several files = the stream dropped and came back; concatenate under one header
        parts = []
        for i, f in enumerate(files):
            with open(f, "r", encoding="utf-8") as fh:
                lines = fh.read().splitlines(True)
            parts.append("".join(lines if i == 0 else lines[1:]))
        body = "".join(parts).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-EEG-Files", str(len(files)))
        self.end_headers()
        self.wfile.write(body)

    # ---- websocket ----
    def serve_ws(self, on_message):
        key = self.headers.get("Sec-WebSocket-Key")
        if not key:
            self.send_error(400)
            return
        self.send_response(101, "Switching Protocols")
        self.send_header("Upgrade", "websocket")
        self.send_header("Connection", "Upgrade")
        self.send_header("Sec-WebSocket-Accept", wsproto.accept_key(key))
        SimpleHTTPRequestHandler.end_headers(self)  # skip our no-store override for the 101 response
        self.close_connection = True
        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        send_lock = threading.Lock()

        def send_raw(opcode, payload):
            with send_lock:
                self.wfile.write(wsproto.build_frame(opcode, payload))
                self.wfile.flush()

        def send_json(obj):
            send_raw(0x1, json.dumps(obj, separators=(",", ":")).encode())

        conn = {"send": send_json, "stop": threading.Event()}
        on_message(conn, None)  # connection opened
        try:
            while True:
                text = wsproto.read_message(self.rfile, send_raw)
                if text is None:
                    break
                try:
                    msg = json.loads(text)
                except ValueError:
                    continue
                if on_message(conn, msg) is False:
                    break
        except (ConnectionError, OSError):
            pass
        finally:
            conn["stop"].set()
            self.state.clients.discard(send_json)

    # ---- /ws: markers, clock sync, session hello ----
    def marker_messages(self, conn, msg):
        state = self.state
        send_json = conn["send"]
        if msg is None:
            state.clients.add(send_json)

            def stats_loop():
                while not conn["stop"].wait(1.0):
                    try:
                        send_json(state.status())
                    except Exception:
                        return
            threading.Thread(target=stats_loop, daemon=True).start()
            return
        kind = msg.get("type")
        if kind == "marker":
            state.handle_marker(msg, conn.get("offset"))
        elif kind == "sync":
            send_json({"type": "sync_reply", "id": msg.get("id"), "t_c": msg.get("t_c"), "t_b": perf_ms()})
        elif kind == "sync_result":
            # Per connection: a second tab (e.g. the neural monitor) has a different clock origin and must not
            # overwrite the session tab's offset. The global copy (used for page_t_ms in EEG files) follows the
            # connection that is running a session.
            conn["offset"] = msg.get("offset_ms")
            if conn.get("session_id") or state.clock_offset_ms is None:
                state.clock_offset_ms = msg.get("offset_ms")
                state.clock_rtt_ms = msg.get("rtt_ms")
        elif kind == "hello":
            path = state.open_csv(msg.get("session_id")) if msg.get("session_id") else state.csv_path
            if msg.get("session_id"):
                conn["session_id"] = msg.get("session_id")
                if conn.get("offset") is not None:
                    state.clock_offset_ms = conn["offset"]
            if msg.get("session_id") and state.eeg:
                state.eeg.start_session(msg.get("session_id"))
            send_json({"type": "hello_ack", "csv_path": path, "lsl_available": state.outlet is not None,
                       "lsl_name": state.outlet_name, "bridge_time_ms": perf_ms()})
        elif kind == "bye":
            return False

    # ---- /predict: the app's predictor protocol, answered by the live EEG P300 scorer ----
    def predict_messages(self, conn, msg):
        if msg is None:
            conn["trial"] = None
            return
        eng = self.state.eeg
        scorer = eng.scorer if eng else None
        kind = msg.get("type")
        send = conn["send"]
        if kind == "trial_start":
            t = {"session_id": msg.get("session_id"), "trial_id": msg.get("trial_id"),
                 "options": [o.get("id") for o in msg.get("options", [])], "active": True}
            conn["trial"] = t
            if scorer:
                threading.Thread(target=self._live_loop, args=(conn, t, scorer), daemon=True).start()
        elif kind == "trial_end":
            if os.environ.get("CORTEXT_DEBUG"):
                print(f"[debug] {time.time():.2f} trial_end received", flush=True)
            t = conn.get("trial")
            if not t or t["trial_id"] != msg.get("trial_id"):
                t = {"session_id": msg.get("session_id"), "trial_id": msg.get("trial_id"), "options": []}
            t["active"] = False

            def finish():
                base = {"type": "prediction", "trial_id": t["trial_id"], "category_scores": {},
                        "eeg_stream": (eng.info.get("name") if eng else None),
                        "simulated": bool(eng and eng.info.get("simulated"))}
                if scorer is None or not eng.info.get("found"):
                    why = "no EEG stream" if (eng and scorer) else (eeg_engine.NUMPY_ERROR or "pylsl not installed")
                    send({**base, "best_option_id": None, "confidence": None, "p300": {}, "error": why})
                    return
                r = scorer.wait_and_score(t["session_id"], t["trial_id"], t["options"])
                if "error" in r:
                    send({**base, "best_option_id": None, "confidence": None, "p300": {}, "error": r["error"]})
                else:
                    if os.environ.get("CORTEXT_DEBUG"):
                        print(f"[debug] {time.time():.2f} prediction scored, sending", flush=True)
                    send({**base, **r})
                    eng.evidence = {"trial_id": t["trial_id"], "final": True, **r}
            threading.Thread(target=finish, daemon=True).start()
        # 'flash' messages are ignored: the bridge already has every flash, with an LSL timestamp, from /ws markers.

    def _live_loop(self, conn, t, scorer):
        while t.get("active") and not conn["stop"].wait(0.3):
            try:
                r = scorer.score_trial(t["session_id"], t["trial_id"], t["options"], partial=True)
                if "error" not in r and r.get("n_epochs_used"):
                    conn["send"]({"type": "live", "trial_id": t["trial_id"], "p300": r["p300"], "category_scores": {}})
                    self.state.eeg.evidence = {"trial_id": t["trial_id"], "final": False, **r}
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; use 0.0.0.0 to serve other machines)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--lsl-name", default="CorTextMarkers", help="LSL outlet name")
    ap.add_argument("--format", choices=["json", "compact"], default="json", help="LSL marker string format")
    ap.add_argument("--no-lsl", action="store_true", help="never create an LSL outlet")
    ap.add_argument("--no-csv", action="store_true", help="do not write the CSV backup")
    ap.add_argument("--eeg-stream", default=None, help="LSL EEG stream name to use (default: the first stream of type EEG, preferring a real one over the simulator)")
    ap.add_argument("--p300-model", default=None,
                    help="a .npz to score with (default: the file in cortext-lab/models/active/, if any). "
                         "'none' = ignore models/active/ and adapt to the session. You can also switch on the Setup page.")
    ap.add_argument("--filters", default=None,
                    help="a filter-chain .json to run (default: the file in cortext-lab/filters/active/, else the built-in "
                         "0.5-20 Hz band-pass). 'default' = ignore filters/active/. You can also switch on the Agent page.")
    ap.add_argument("--scorer", choices=["auto", "model", "adaptive", "heuristic"], default="auto",
                    help="auto: trained model if given, else adapt to this session after 5 trials, else a heuristic")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    state = State(args)
    Handler.state = state
    handler = partial(Handler, directory=WEB_DIR)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    if pylsl is not None:
        model_path, model_source, model_warn = models.resolve_startup(args.p300_model)
        try:
            fspec, fsource, fwarn = filters.resolve_startup(args.filters)
        except filters.FilterError as exc:
            sys.exit(f"--filters: {exc}")
        state.eeg = eeg_engine.EEGEngine(state, pylsl, args.eeg_stream, model_path, args.scorer, LOG_DIR,
                                         filter_spec=fspec, filter_source=fsource)
        print(f"  Filters: {fspec['name']}: {'; '.join(filters.summary(fspec))}  [{fsource}]"
              + (f"\n  WARNING: {fwarn}" if fwarn else ""))
        if state.eeg.scorer is not None:
            state.eeg.scorer.model_source = model_source
        print(f"  P300 model: {os.path.basename(model_path) if model_path else 'none (adapts to the session after 5 trials)'}"
              f"  [{model_source}]" + (f"\n  WARNING: {model_warn}" if model_warn else ""))
        if state.eeg.scorer is not None and state.eeg.scorer.model_error:
            print(f"  WARNING: {state.eeg.scorer.model_error}")
        threading.Thread(target=state.eeg.run, daemon=True).start()

    print(f"CorText Lab bridge  http://{args.host}:{args.port}   (ws path /ws)")
    if state.outlet:
        print(f"  LSL marker outlet: '{args.lsl_name}' (type Markers, format {args.format})")
    elif args.no_lsl:
        print("  LSL disabled (--no-lsl). Markers go to CSV only.")
    else:
        print(f"  LSL NOT available -> CSV fallback only ({LSL_IMPORT_ERROR}). Install with: pip install pylsl")
    print(f"  CSV backups: {'disabled' if args.no_csv else LOG_DIR}")
    if state.eeg is None:
        print("  EEG: needs pylsl (pip install pylsl) to receive the headset stream")
    else:
        print("  EEG: waiting for an LSL EEG stream (run eeg/stream_eeg.py or eeg/stream_muse.py, or eeg/fake_eeg.py [--muse] to simulate)"
              + ("" if eeg_engine.NUMPY_ERROR is None else "\n       numpy/scipy missing -> recording only, no scoring/brain map"))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""CorText Lab bridge: serves the web app and relays flash markers to LSL / CSV.

    python bridge/bridge.py            # serves http://localhost:8000 and ws://localhost:8000/ws

* Browser -> WebSocket (/ws) -> this script -> LSL string-marker outlet (if pylsl is installed)
* Every marker is ALSO appended to a CSV in bridge/logs/ (a backup you can always use;
  disable with --no-csv). If pylsl or liblsl is missing the bridge keeps working in CSV-only mode.
* Clock sync: the page runs ping-pong rounds and reports (bridge_time - page_time). Markers carry the
  page's performance.now() timestamp, and the bridge converts it to an LSL timestamp, so network
  / event-loop delay does not shift the marker in the EEG time base.
* Optional: watches for an LSL stream of type EEG and reports whether samples are flowing.

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
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wsproto  # noqa: E402

try:  # optional dependency
    import pylsl  # type: ignore
    LSL_IMPORT_ERROR = None
except Exception as exc:  # ImportError, or OSError when liblsl is missing
    pylsl = None
    LSL_IMPORT_ERROR = repr(exc)

HERE = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(os.path.dirname(HERE), "web")
LOG_DIR = os.path.join(HERE, "logs")

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
        self.eeg = {"found": False}
        self.clients = set()
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

    def handle_marker(self, msg):
        recv = perf_ms()
        page_t = msg.get("t_ms")
        offset = self.clock_offset_ms
        if page_t is not None and offset is not None:
            bridge_t = page_t + offset
        else:
            bridge_t = recv  # no sync yet: best effort
        transit = recv - bridge_t
        lsl_ts = None
        if self.outlet is not None:
            lsl_ts = pylsl.local_clock() - transit / 1000.0
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
            "eeg": self.eeg,
        }


def format_marker(msg, fmt):
    if fmt == "compact":
        keys = ["event", "session_id", "trial_id", "option_id", "category", "is_target", "repeat_index", "flash_index"]
        return "|".join("" if msg.get(k) is None else str(msg.get(k)) for k in keys)
    keep = {k: msg.get(k) for k in CSV_COLUMNS if k in msg and k not in ("extra",)}
    keep["extra"] = msg.get("extra")
    return json.dumps(keep, separators=(",", ":"))


def eeg_monitor(state: State):
    """Background thread: report whether an LSL EEG stream is present and delivering samples."""
    if pylsl is None:
        return
    inlet = None
    count = 0
    t_window = time.time()
    last_sample = None
    while True:
        try:
            if inlet is None:
                found = pylsl.resolve_byprop("type", "EEG", timeout=1.0)
                if found:
                    inlet = pylsl.StreamInlet(found[0], max_buflen=2)
                    state.eeg = {"found": True, "name": found[0].name(), "channels": found[0].channel_count(),
                                 "nominal_srate": found[0].nominal_srate(), "measured_srate": None, "age_ms": None}
                    count, t_window = 0, time.time()
                else:
                    state.eeg = {"found": False}
                    time.sleep(1.0)
                continue
            chunk, stamps = inlet.pull_chunk(timeout=0.0, max_samples=4096)
            now = time.time()
            if stamps:
                count += len(stamps)
                last_sample = now
            if now - t_window >= 1.0:
                state.eeg["measured_srate"] = round(count / (now - t_window), 1)
                state.eeg["age_ms"] = None if last_sample is None else round((now - last_sample) * 1000)
                count, t_window = 0, now
            time.sleep(0.05)
        except Exception as exc:  # stream lost etc.
            state.eeg = {"found": False, "error": repr(exc)}
            inlet = None
            time.sleep(1.0)


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
        if path == "/ws" and self.headers.get("Upgrade", "").lower() == "websocket":
            return self.serve_ws()
        if path == "/api/status":
            body = json.dumps(self.state.status()).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        return super().do_GET()

    # ---- websocket ----
    def serve_ws(self):
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

        state = self.state
        state.clients.add(send_json)
        stop = threading.Event()

        def stats_loop():
            while not stop.wait(1.0):
                try:
                    send_json(state.status())
                except Exception:
                    return

        threading.Thread(target=stats_loop, daemon=True).start()
        try:
            while True:
                text = wsproto.read_message(self.rfile, send_raw)
                if text is None:
                    break
                try:
                    msg = json.loads(text)
                except ValueError:
                    continue
                kind = msg.get("type")
                if kind == "marker":
                    state.handle_marker(msg)
                elif kind == "sync":
                    send_json({"type": "sync_reply", "id": msg.get("id"), "t_c": msg.get("t_c"), "t_b": perf_ms()})
                elif kind == "sync_result":
                    state.clock_offset_ms = msg.get("offset_ms")
                    state.clock_rtt_ms = msg.get("rtt_ms")
                elif kind == "hello":
                    path = state.open_csv(msg.get("session_id")) if msg.get("session_id") else state.csv_path
                    send_json({"type": "hello_ack", "csv_path": path, "lsl_available": state.outlet is not None,
                               "lsl_name": state.outlet_name, "bridge_time_ms": perf_ms()})
                elif kind == "bye":
                    break
        except (ConnectionError, OSError):
            pass
        finally:
            stop.set()
            state.clients.discard(send_json)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; use 0.0.0.0 to serve other machines)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--lsl-name", default="CorTextMarkers", help="LSL outlet name")
    ap.add_argument("--format", choices=["json", "compact"], default="json", help="LSL marker string format")
    ap.add_argument("--no-lsl", action="store_true", help="never create an LSL outlet")
    ap.add_argument("--no-csv", action="store_true", help="do not write the CSV backup")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    state = State(args)
    Handler.state = state
    handler = partial(Handler, directory=WEB_DIR)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.daemon_threads = True
    threading.Thread(target=eeg_monitor, args=(state,), daemon=True).start()

    print(f"CorText Lab bridge  http://{args.host}:{args.port}   (ws path /ws)")
    if state.outlet:
        print(f"  LSL marker outlet: '{args.lsl_name}' (type Markers, format {args.format})")
    elif args.no_lsl:
        print("  LSL disabled (--no-lsl). Markers go to CSV only.")
    else:
        print(f"  LSL NOT available -> CSV fallback only ({LSL_IMPORT_ERROR}). Install with: pip install pylsl")
    print(f"  CSV backups: {'disabled' if args.no_csv else LOG_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()

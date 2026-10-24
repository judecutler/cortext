#!/usr/bin/env python3
"""Example predictor server that speaks the CorText Lab prediction protocol.

    python tools/mock_predictor_server.py --port 8765

Use it as a template for the real P300 model. It scores options with RANDOM numbers and never sees labels:
the web app deliberately does NOT send is_target / true_answer to a ws predictor (no label leakage).

Protocol (JSON text frames)
  app -> server  {"type":"trial_start","session_id","trial_id","options":[{"id","category","label","position"}]}
  app -> server  {"type":"flash","trial_id","option_id","flash_index","repeat_index","t_ms","wall_ms"}
  app -> server  {"type":"trial_end","trial_id"}
  server -> app  {"type":"live","trial_id","p300":{option_id: score},"category_scores":{category: score}}   (any time)
  server -> app  {"type":"prediction","trial_id","best_option_id","confidence","p300":{...},"category_scores":{...}}
"""
import argparse
import json
import os
import random
import socket
import socketserver
import sys
import threading

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bridge"))
import wsproto  # noqa: E402


class H(socketserver.StreamRequestHandler):
    def handle(self):
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = self.request.recv(4096)
            if not chunk:
                return
            data += chunk
        headers = dict(l.split(": ", 1) for l in data.decode().split("\r\n")[1:] if ": " in l)
        key = headers.get("Sec-WebSocket-Key")
        self.request.sendall(("HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                              f"Sec-WebSocket-Accept: {wsproto.accept_key(key)}\r\n\r\n").encode())
        lock = threading.Lock()

        def raw(op, payload):
            with lock:
                self.request.sendall(wsproto.build_frame(op, payload))

        def send(obj):
            raw(0x1, json.dumps(obj).encode())

        options, scores, counts = [], {}, {}
        while True:
            try:
                text = wsproto.read_message(self.rfile, raw)
            except (ConnectionError, OSError):
                return
            if text is None:
                return
            msg = json.loads(text)
            if msg["type"] == "trial_start":
                options = msg["options"]
                scores = {o["id"]: 0.0 for o in options}
                counts = {o["id"]: 0 for o in options}
            elif msg["type"] == "flash" and msg["option_id"] in scores:
                oid = msg["option_id"]
                counts[oid] += 1
                scores[oid] += random.gauss(0, 1)  # <- replace with a real P300 score for this epoch
                send({"type": "live", "trial_id": msg["trial_id"],
                      "p300": {k: scores[k] / max(1, counts[k]) for k in scores}})
            elif msg["type"] == "trial_end":
                avg = {k: scores[k] / max(1, counts[k]) for k in scores}
                best = max(avg, key=avg.get) if avg else None
                send({"type": "prediction", "trial_id": msg["trial_id"], "best_option_id": best,
                      "confidence": 1.0 / max(1, len(avg)), "p300": avg, "category_scores": {}})


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    print(f"mock predictor on ws://localhost:{a.port}  (random scores; for protocol testing only)")
    Server(("127.0.0.1", a.port), H).serve_forever()

#!/usr/bin/env python3
"""Integration test for bridge/bridge.py (stdlib only). Run: python tests/test_bridge.py"""
import base64
import csv
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 8765 + (os.getpid() % 200)


def ws_connect(port):
    s = socket.create_connection(("127.0.0.1", port), timeout=5)
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall((f"GET /ws HTTP/1.1\r\nHost: localhost:{port}\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += s.recv(1024)
    head, rest = buf.split(b"\r\n\r\n", 1)
    assert b"101" in head.split(b"\r\n")[0], head
    sys.path.insert(0, os.path.join(ROOT, "bridge"))
    import wsproto
    assert wsproto.accept_key(key).encode() in head
    return s, rest


def send_text(s, obj):
    payload = json.dumps(obj).encode()
    mask = os.urandom(4)
    n = len(payload)
    head = bytes([0x81])
    if n < 126:
        head += bytes([0x80 | n])
    else:
        head += bytes([0x80 | 126]) + struct.pack(">H", n)
    s.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))


class Reader:
    def __init__(self, s, pre=b""):
        self.s, self.buf = s, pre

    def _need(self, n):
        while len(self.buf) < n:
            chunk = self.s.recv(65536)
            if not chunk:
                raise ConnectionError("closed")
            self.buf += chunk
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def message(self):
        b1, b2 = self._need(2)
        n = b2 & 0x7F
        if n == 126:
            (n,) = struct.unpack(">H", self._need(2))
        return json.loads(self._need(n).decode())

    def until(self, kind, limit=20):
        for _ in range(limit):
            m = self.message()
            if m.get("type") == kind:
                return m
        raise AssertionError(f"no {kind} message")


def main():
    logdir = os.path.join(ROOT, "bridge", "logs")
    before = set(os.listdir(logdir)) if os.path.isdir(logdir) else set()
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT), "--no-lsl"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/status", timeout=1).read()
                break
            except Exception:
                time.sleep(0.1)
        # static + status
        cfg = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/config/stimuli.json"))
        assert len(cfg["items"]) == 12
        status = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/status"))
        assert status["lsl_available"] is False

        s, pre = ws_connect(PORT)
        r = Reader(s, pre)
        send_text(s, {"type": "hello", "session_id": "test-sess", "participant_id": "P0"})
        ack = r.until("hello_ack")
        assert ack["csv_path"] and ack["lsl_available"] is False

        # clock sync: pretend page clock started 1000 ms before bridge clock
        t_page = 50.0
        send_text(s, {"type": "sync", "id": 1, "t_c": t_page})
        rep = r.until("sync_reply")
        assert rep["id"] == 1 and rep["t_c"] == t_page
        offset = rep["t_b"] - t_page
        send_text(s, {"type": "sync_result", "offset_ms": offset, "rtt_ms": 1.0})

        now_page = rep["t_b"] - offset - 5  # a marker stamped 5 ms BEFORE the sync reply, so its transit time is positive
        for i in range(3):
            send_text(s, {"type": "marker", "seq": i, "session_id": "test-sess", "event": "flash_on", "trial_id": 1,
                          "option_id": "dog", "category": "animal", "is_target": True, "repeat_index": i,
                          "flash_index": i, "t_ms": now_page + i, "extra": {"k": 1}})
        time.sleep(0.4)
        st = r.until("stats")
        for _ in range(5):
            if st["markers_received"] >= 3:
                break
            st = r.until("stats")
        assert st["markers_received"] == 3, st
        assert abs(st["clock_offset_ms"] - offset) < 1e-6

        with open(ack["csv_path"], newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3 and rows[0]["event"] == "flash_on" and rows[2]["repeat_index"] == "2", rows
        assert float(rows[0]["est_transit_ms"]) >= 0
        s.close()
        os.remove(ack["csv_path"])
        print("bridge OK: static serving, WS handshake, sync, markers -> CSV (LSL disabled)")
    finally:
        proc.terminate()
        try:
            proc.wait(3)
        except Exception:
            proc.kill()
        if os.path.isdir(logdir):
            for f in set(os.listdir(logdir)) - before:
                try:
                    os.remove(os.path.join(logdir, f))
                except OSError:   # Windows keeps a just-stopped process's files locked for a moment
                    pass


if __name__ == "__main__":
    main()

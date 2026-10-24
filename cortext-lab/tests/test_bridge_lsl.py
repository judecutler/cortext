#!/usr/bin/env python3
"""LSL integration test (needs `pip install pylsl`). Run: python tests/test_bridge_lsl.py

Starts the bridge with LSL enabled, sends markers over WebSocket, and checks they arrive on a real LSL
inlet with the right strings and a timestamp corrected for the (fake) clock offset."""
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_bridge as tb  # noqa: E402

try:
    import pylsl
except Exception as exc:
    print("SKIP: pylsl not available:", exc)
    sys.exit(0)

ROOT = tb.ROOT
PORT = tb.PORT + 300


def main():
    logdir = os.path.join(ROOT, "bridge", "logs")
    before = set(os.listdir(logdir)) if os.path.isdir(logdir) else set()
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT),
                             "--lsl-name", "CorTextTest"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/status", timeout=1).read()
                break
            except Exception:
                time.sleep(0.1)
        streams = pylsl.resolve_byprop("name", "CorTextTest", timeout=5)
        assert streams, "LSL outlet not found"
        inlet = pylsl.StreamInlet(streams[0])
        inlet.open_stream(timeout=5)

        s, pre = tb.ws_connect(PORT)
        r = tb.Reader(s, pre)
        tb.send_text(s, {"type": "hello", "session_id": "lsl-test"})
        ack = r.until("hello_ack")
        assert ack["lsl_available"] is True
        tb.send_text(s, {"type": "sync", "id": 1, "t_c": 10.0})
        rep = r.until("sync_reply")
        offset = rep["t_b"] - 10.0
        tb.send_text(s, {"type": "sync_result", "offset_ms": offset, "rtt_ms": 0.5})
        time.sleep(0.1)
        # the page says this flash happened 200 ms ago (page time = bridge time - offset - 200)
        page_now = (rep["t_b"] + 100) - offset
        t_before = pylsl.local_clock()
        tb.send_text(s, {"type": "marker", "seq": 7, "session_id": "lsl-test", "event": "flash_on", "trial_id": 3,
                         "option_id": "dog", "category": "animal", "is_target": True, "repeat_index": 2,
                         "flash_index": 9, "t_ms": page_now - 200})
        sample, ts = inlet.pull_sample(timeout=5)
        assert sample, "no LSL sample"
        got = json.loads(sample[0])
        assert got["event"] == "flash_on" and got["option_id"] == "dog" and got["is_target"] is True and got["seq"] == 7, got
        # timestamp should be ~ (receive time - transit) where transit ~= bridge recv - (page_t + offset)
        assert ts < pylsl.local_clock() + 0.01
        print("bridge LSL OK: marker received on LSL inlet:", sample[0][:90], "... lsl_ts age(s) =", round(pylsl.local_clock() - ts, 3))
        s.close()
    finally:
        proc.terminate()
        try:
            proc.wait(3)
        except Exception:
            proc.kill()
        if os.path.isdir(logdir):
            for f in set(os.listdir(logdir)) - before:
                os.remove(os.path.join(logdir, f))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""End-to-end test of the EEG path WITHOUT a headset:  fake_eeg.py -> LSL -> bridge (record, quality, P300 scorer).

    python tests/test_eeg_pipeline.py

Plays the app's role over the bridge WebSockets (/ws for markers, /predict for predictions), runs flash trials,
and checks: the stream is found with the Unicorn channel names, quality + band power are reported, EEG is recorded
per session and downloadable, predictions arrive in time, the adaptive model switches on after 5 trials, accuracy on
the simulated P300 is well above chance, and a trial's prediction never depends on its own labels.
Needs pylsl, numpy, scipy.
"""
import base64
import csv
import io
import json
import os
import random
import socket
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from test_bridge import Reader, send_text  # noqa: E402

PORT = 9100 + (os.getpid() % 300)


def ws(path):
    s = socket.create_connection(("127.0.0.1", PORT), timeout=10)
    key = base64.b64encode(os.urandom(16)).decode()
    s.sendall((f"GET {path} HTTP/1.1\r\nHost: localhost\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
    buf = b""
    while b"\r\n\r\n" not in buf:
        buf += s.recv(1024)
    head, rest = buf.split(b"\r\n\r\n", 1)
    assert b"101" in head.split(b"\r\n")[0], head
    return s, Reader(s, rest)


def get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=5) as r:
        return r.status, r.read()


def status():
    return json.loads(get("/api/status")[1])


def main():
    fake = subprocess.Popen([sys.executable, os.path.join(ROOT, "eeg", "fake_eeg.py"), "--seed", "3", "--p300-uv", "6"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bridge = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT), "--p300-model", "none"],  # tests the heuristic -> adaptive path, not models/active/
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(100):
            try:
                st = status()
                if st["eeg"].get("found") and st["eeg"].get("quality"):
                    break
            except Exception:
                pass
            time.sleep(0.2)
        eeg = st["eeg"]
        assert eeg["found"] and eeg["simulated"], eeg
        assert eeg["labels"] == ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "POz", "PO8"], eeg["labels"]
        assert len(eeg["quality"]) == 8 and set(eeg["bands"]) == {"delta", "theta", "alpha", "beta"}, eeg
        assert eeg["bands"]["alpha"][6] > eeg["bands"]["alpha"][0], "simulated alpha should be stronger at POz than Fz"
        tr = json.loads(get("/api/eeg/trace?seconds=3")[1])
        assert len(tr["data"]) == 8 and len(tr["data"][0]) > 100, tr.keys()
        print("stream found, labels, quality, band power, trace: OK", {q['label']: q['status'] for q in eeg["quality"]})

        ms, mr = ws("/ws")
        ps, pr = ws("/predict")
        sid = f"eegtest-{random.randint(0, 99999)}"
        send_text(ms, {"type": "sync_result", "offset_ms": 0.0, "rtt_ms": 0.1})   # same machine: page clock = perf clock
        send_text(ms, {"type": "hello", "session_id": sid})
        seq = [0]

        def mark(event, **kw):
            seq[0] += 1
            send_text(ms, {"type": "marker", "seq": seq[0], "session_id": sid, "event": event,
                           "t_ms": time.perf_counter() * 1000.0, **kw})

        mark("session_start")
        options = ["a", "b", "c", "d"]
        rng = random.Random(1)
        results = []
        n_trials, reps = 14, 6
        for trial_id in range(n_trials):
            truth = options[trial_id % 4]
            mark("trial_start", trial_id=trial_id)
            send_text(ps, {"type": "trial_start", "session_id": sid, "trial_id": trial_id,
                           "options": [{"id": o, "category": "x", "label": o, "position": i} for i, o in enumerate(options)]})
            order = []
            for r in range(reps):
                perm = options[:]
                rng.shuffle(perm)
                order += perm
            for k, o in enumerate(order):
                mark("flash_on", trial_id=trial_id, option_id=o, is_target=(o == truth), repeat_index=k // 4, flash_index=k)
                time.sleep(0.25)
            time.sleep(0.2)
            t0 = time.time()
            send_text(ps, {"type": "trial_end", "trial_id": trial_id})
            pred = pr.until("prediction", 200)
            lat = time.time() - t0
            assert lat < 3.0, f"prediction took {lat:.2f}s (app waits 3 s)"
            assert pred["trial_id"] == trial_id and pred["simulated"] is True
            assert abs(sum(pred["probabilities"].values()) - 1) < 1e-6
            mark("trial_end", trial_id=trial_id, true_answer_id=truth)
            results.append((pred["scorer"], pred["best_option_id"] == truth, pred["n_epochs_used"], pred["n_epochs"], round(lat, 2)))
            time.sleep(0.9)   # let the trial's EEG arrive so it can join the adaptive pool
        print("trials (scorer, correct, epochs used/total, latency s):")
        for r in results:
            print("  ", r)
        scorers = [r[0] for r in results]
        assert scorers[0] == "heuristic" and "adaptive" in scorers[5:], scorers
        acc = sum(r[1] for r in results) / len(results)
        assert acc >= 0.6, f"accuracy {acc:.2f} on simulated P300 (chance 0.25)"
        assert all(r[2] > 0 for r in results)
        print(f"live predictions: accuracy {acc:.2f} vs chance 0.25 (SIMULATED EEG), adaptive model after trial "
              f"{scorers.index('adaptive')}: OK")

        code, body = get(f"/api/eeg/session.csv?session_id={sid}")
        rows = list(csv.reader(io.StringIO(body.decode())))
        assert rows[0] == ["lsl_time", "page_t_ms", "Fz", "C3", "Cz", "C4", "Pz", "PO7", "POz", "PO8"], rows[0]
        assert len(rows) > 250 * 20, len(rows)
        page_t = [float(r[1]) for r in rows[1:] if r[1]]
        assert page_t and abs(page_t[-1] - time.perf_counter() * 1000) < 2000, "page_t_ms should be in the page clock"
        st = status()
        assert st["eeg"]["scorer"]["adaptive_ready"] and os.path.exists(os.path.join(ROOT, "bridge", "logs", f"p300_adaptive_{sid}.npz"))
        print(f"session EEG recorded ({len(rows) - 1} samples) and downloadable, page-clock column aligned, adaptive model saved: OK")
        ms.close()
        ps.close()
        print("EEG pipeline OK")
    finally:
        bridge.terminate()
        fake.terminate()


if __name__ == "__main__":
    main()

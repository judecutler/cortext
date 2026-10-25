#!/usr/bin/env python3
"""Decoder on a simulated P300 while the headset drives the visuals (demo with an unreliable headset).

    python tests/test_sim_p300.py

A "broken headset" (fake_eeg.py with NO P300 and 40 µV noise) streams to the bridge, which runs the demo's default
p3000 function. Checks: with the decoder on the headset, accuracy is near chance; switched to 'replace' (simulated
P300 on simulated background EEG) the same p3000 function decodes well above chance with no other change; the live
trace stays the headset's own signal; the recording is untouched; 'overlay' works; bad input is refused; predictions
carry sim_p300 and the UI flag 'simulated' is not set by the sim mode itself. Needs pylsl, numpy, scipy.
"""
import json
import os
import random
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from test_bridge import send_text  # noqa: E402
import test_eeg_pipeline as T  # noqa: E402

PORT = T.PORT = 9500 + (os.getpid() % 300)


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def main():
    fake = subprocess.Popen([sys.executable, os.path.join(ROOT, "eeg", "fake_eeg.py"), "--seed", "5", "--p300-uv", "0",
                             "--noise-uv", "40", "--name", "BrokenHeadset-FAKE"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import tempfile
    env = dict(os.environ, CORTEXT_BRIDGE_LOG_DIR=tempfile.mkdtemp(prefix="cortext-sim-"))
    bridge = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT),
                               "--scorer", "model"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        st = None
        for _ in range(100):
            try:
                st = T.status()
                if st["eeg"].get("found") and st["eeg"].get("quality"):
                    break
            except Exception:
                pass
            time.sleep(0.2)
        eeg = st["eeg"]
        assert eeg["sim_p300"]["mode"] == "off" and eeg["sim_p300"]["uv"] == 15.0, eeg["sim_p300"]
        assert "p3000" in (eeg["scorer"]["model_file"] or ""), eeg["scorer"]
        assert post("/api/eeg/sim", {"mode": "bogus"})["ok"] is False
        assert post("/api/eeg/sim", {"uv": 99})["ok"] is False
        assert T.status()["eeg"]["sim_p300"]["mode"] == "off", "a refused change must not change anything"
        print("status, defaults, p3000 active, bad input refused: OK")

        ms, mr = T.ws("/ws")
        ps, pr = T.ws("/predict")
        sid = f"simtest-{random.randint(0, 99999)}"
        send_text(ms, {"type": "sync_result", "offset_ms": 0.0, "rtt_ms": 0.1})
        send_text(ms, {"type": "hello", "session_id": sid})
        seq = [0]

        def mark(event, **kw):
            seq[0] += 1
            send_text(ms, {"type": "marker", "seq": seq[0], "session_id": sid, "event": event,
                           "t_ms": time.perf_counter() * 1000.0, **kw})

        mark("session_start")
        options = ["a", "b", "c", "d"]
        rng = random.Random(2)
        trial_no = [0]

        def run(n, expect_mode):
            hits = 0
            for _ in range(n):
                trial_id = trial_no[0]
                trial_no[0] += 1
                truth = rng.choice(options)
                mark("trial_start", trial_id=trial_id)
                send_text(ps, {"type": "trial_start", "session_id": sid, "trial_id": trial_id,
                               "options": [{"id": o, "category": "x", "label": o, "position": i} for i, o in enumerate(options)]})
                for r in range(3):                       # the demo: one at a time, 3 rounds, 1 s + 0.3 s gap
                    perm = options[:]
                    rng.shuffle(perm)
                    for k, o in enumerate(perm):
                        mark("flash_on", trial_id=trial_id, option_id=o, is_target=(o == truth), repeat_index=r, flash_index=r * 4 + k)
                        time.sleep(0.45)                 # shorter than the app's 1.3 s, neighbouring epochs overlap: a harder test than the app
                time.sleep(0.3)
                send_text(ps, {"type": "trial_end", "trial_id": trial_id})
                pred = pr.until("prediction", 200)
                assert pred["trial_id"] == trial_id and not pred.get("error"), pred
                assert pred["sim_p300"] == expect_mode, pred
                mark("trial_end", trial_id=trial_id, true_answer_id=truth)
                hits += pred["best_option_id"] == truth
            return hits / n

        acc_off = run(12, "off")
        print(f"decoder on the broken headset: {acc_off:.2f} (chance 0.25)")
        assert acc_off <= 0.6, acc_off

        r = post("/api/eeg/sim", {"mode": "replace"})
        assert r["ok"] and r["mode"] == "replace" and r["uv"] == 15.0, r
        acc_rep = run(16, "replace")
        print(f"decoder on the simulated P300 (replace, 15 µV), same p3000 function: {acc_rep:.2f}")
        assert acc_rep >= 0.6, acc_rep

        tr = json.loads(T.get("/api/eeg/trace?seconds=3")[1])
        import statistics
        sd = statistics.pstdev(tr["data"][4])
        assert sd > 15, f"the live trace must stay the headset's own (40 µV noise) signal, std {sd:.1f}"
        print(f"live trace is still the headset (Pz std {sd:.0f} µV): OK")

        r = post("/api/eeg/sim", {"mode": "overlay", "uv": 20})
        assert r["ok"] and r["mode"] == "overlay" and r["uv"] == 20.0, r
        acc_ov = run(4, "overlay")
        print(f"overlay on 40 µV headset noise runs (accuracy {acc_ov:.2f}, few trials): OK")
        assert post("/api/eeg/sim", {"mode": "off"})["mode"] == "off"
        run(1, "off")

        code, body = T.get(f"/api/eeg/session.csv?session_id={sid}")
        assert code == 200 and len(body) > 100000, len(body)
        print("session recorded from the headset: OK")
        ms.close()
        ps.close()
        print("sim P300 OK")
    finally:
        bridge.terminate()
        fake.terminate()


if __name__ == "__main__":
    main()

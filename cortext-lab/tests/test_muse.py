#!/usr/bin/env python3
"""Muse headset path, without a Muse:  fake_eeg.py --muse -> LSL -> bridge -> Muse p3000 function -> predictions.

    python tests/test_muse.py

Part 1 (offline, simulated Muse EEG with blinks): the Muse p3000 function picks the attended picture, sits at chance
with no P300, and blinks on AF7/AF8 no longer throw epochs away.
Part 2 (live, two runs): a simulated Muse with the unconnected "Right AUX" channel, like muselsl streams it, and then
one stamped with wall-clock time (muselsl without --lsltime). Checks: AUX dropped, labels TP9 AF7 AF8 TP10, 256 Hz,
device "Muse", the 8-channel p3000 function that is active swaps itself for the Muse one and its "Muse P300" filter
chain, every trial is scored by it from trial 1, accuracy well above chance, the session CSV has the 4 channels, and
wall-clock stamps are corrected. Uses temporary models/filters folders. Simulated accuracy is wiring evidence only.
Needs pylsl, numpy, scipy.
"""
import csv
import io
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "bridge"))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import filters as F  # noqa: E402
import make_p3000_function as F3  # noqa: E402
import p300 as P  # noqa: E402
from test_bridge import send_text  # noqa: E402
import test_eeg_pipeline as T  # noqa: E402  (ws helper)

os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")
SR = 256.0
P3 = np.array([1.0, 0.3, 0.3, 1.0])
BLINK = np.array([0.1, 1.0, 1.0, 0.1])


def offline():
    m = F3.build_muse()
    chain_spec = F.read_file(F3.MUSE_CHAIN)
    labels = list(P.MUSE_MONTAGE)
    assert m.labels == labels and m.filter_signature == F.signature(chain_spec)
    for p300_uv, seed, want in ((6.0, 1, "hit"), (0.0, 2, "chance")):
        rng = np.random.default_rng(seed)
        trials, t = [], 1.0
        for _ in range(60):
            true = rng.choice(list("ABCD"))
            fl = []
            for _ in range(3):
                for o in rng.permutation(list("ABCD")):
                    fl.append((t, o))
                    t += 1.3
            trials.append((fl, true))
            t += 2.0
        n = int((t + 2) * SR)
        ts = np.arange(n) / SR
        x = rng.normal(0, 8, (n, 4)) + 800.0 + 4 * np.sin(2 * np.pi * 60 * ts)[:, None]
        k = np.arange(int(0.8 * SR)) / SR
        bump = np.exp(-0.5 * ((k - 0.35) / 0.07) ** 2)
        for fl, true in trials:
            for on, o in fl:
                i = int(on * SR)
                if o == true:
                    x[i:i + len(k)] += p300_uv * bump[:, None] * P3
        blinks = rng.choice(n - 100, size=len(trials) * 3, replace=False)      # ~ one blink every 5 s
        for b in blinks:
            x[b:b + 75] += 180 * np.hanning(75)[:, None] * BLINK
        y = F.Chain(chain_spec, SR, labels)(x)
        check = P.reject_picks(labels)
        assert check == [0, 3], check
        hits, used, used_all = 0, 0, 0
        for fl, true in trials:
            eps = P.epochs(ts, y, [f[0] for f in fl], chain_spec["reject_uv"], check)
            used += sum(e is not None for e in eps)
            used_all += sum(e is not None for e in P.epochs(ts, y, [f[0] for f in fl], chain_spec["reject_uv"]))
            scores = [None if e is None else float(m.score(e.ravel())[0]) for e in eps]
            probs, _ = P.noise_normalised_probs([f[1] for f in fl], scores, list("ABCD"))
            hits += max(probs, key=probs.get) == true
        acc, total = hits / len(trials), 12 * len(trials)
        print(f"offline Muse, {p300_uv:g} uV P300: accuracy {acc:.0%} (chance 25%); epochs kept {used}/{total} "
              f"(would be {used_all}/{total} if blinks on AF7/AF8 rejected them)")
        assert used > used_all and used >= 0.95 * total
        assert (acc >= 0.6) if want == "hit" else (acc <= 0.45), acc


def live(wall_clock, models_dir, filters_dir):
    T.PORT = 9500 + (os.getpid() % 200) + (7 if wall_clock else 0)
    env = dict(os.environ, CORTEXT_MODELS_DIR=models_dir, CORTEXT_FILTERS_DIR=filters_dir)
    fake = subprocess.Popen([sys.executable, os.path.join(ROOT, "eeg", "fake_eeg.py"), "--muse", "--seed", "5", "--p300-uv", "10"]
                            + (["--wall-clock"] if wall_clock else []), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    bridge = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(T.PORT), "--no-csv"],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        st = None
        for _ in range(300):
            try:
                st = T.status()
                if st["eeg"].get("found") and st["eeg"].get("quality") and st["eeg"].get("scorer"):
                    break
            except Exception:
                pass
            time.sleep(0.2)
        eeg = st["eeg"]
        assert eeg["labels"] == ["TP9", "AF7", "AF8", "TP10"], eeg["labels"]
        assert eeg["channels"] == 4 and eeg["dropped_channels"] == ["Right AUX"], eeg
        assert eeg["nominal_srate"] == 256.0 and len(eeg["quality"]) == 4, eeg
        assert eeg["simulated"]
        if wall_clock:
            assert eeg.get("clock_note"), "wall-clock stamps should have been detected"
        assert abs(eeg.get("delay_ms", 1e9)) < 2000, eeg.get("delay_ms")
        sc = eeg["scorer"]
        assert sc["active"] == "model" and "p3000_function_muse" in (sc["model_file"] or ""), sc
        assert not sc["model_error"] and not sc["model_filter_mismatch"], sc
        print(f"[{'wall-clock' if wall_clock else 'muselsl-like'}] stream: {eeg['device']}, {eeg['labels']}, AUX dropped, "
              f"{eeg['measured_srate']} Hz, delay {eeg.get('delay_ms')} ms, model {os.path.basename(sc['model_file'])}"
              f" ({sc['model_source']})")

        ms, mr = T.ws("/ws")
        ps, pr = T.ws("/predict")
        sid = f"musetest-{random.randint(0, 99999)}"
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
        results = []
        for trial_id in range(16):
            truth = rng.choice(options)
            mark("trial_start", trial_id=trial_id)
            send_text(ps, {"type": "trial_start", "session_id": sid, "trial_id": trial_id,
                           "options": [{"id": o, "category": "x", "label": o, "position": i} for i, o in enumerate(options)]})
            order = []
            for _ in range(4):
                perm = options[:]
                rng.shuffle(perm)
                order += perm
            for k, o in enumerate(order):
                mark("flash_on", trial_id=trial_id, option_id=o, is_target=(o == truth), repeat_index=k // 4, flash_index=k)
                time.sleep(0.35)
            time.sleep(0.2)
            send_text(ps, {"type": "trial_end", "trial_id": trial_id})
            pred = pr.until("prediction", 200)
            mark("trial_end", trial_id=trial_id, true_answer_id=truth)
            results.append((pred["scorer"], pred["best_option_id"] == truth, pred["n_epochs_used"], pred["n_epochs"],
                            round(pred.get("confidence") or 0, 2)))
            time.sleep(0.3)
        print("   trials (scorer, correct, epochs used/total, confidence):", results)
        time.sleep(4.5)                      # let the last trials join the pool so the ERP covers them
        erp = json.loads(T.get("/api/eeg/erp")[1])
        if erp.get("target") and erp.get("nontarget"):
            tt = np.array(erp["times"])
            win = (tt >= 0.24) & (tt < 0.48)
            for ch in ("TP9", "TP10", "AF7"):
                i = erp["labels"].index(ch)
                d = np.array(erp["target"][i]) - np.array(erp["nontarget"][i])
                print(f"   ERP {ch}: target - nontarget, 240-480 ms = {d[win].mean():+.1f} uV; by bin {np.round(d, 1).tolist()}")
            i = erp["labels"].index("TP9")
            assert (np.array(erp["target"][i]) - np.array(erp["nontarget"][i]))[win].mean() > 2.0, "no P300 at TP9"
        assert all(r[0] == "model" for r in results), results
        acc = sum(r[1] for r in results) / len(results)
        assert acc >= 0.45, f"accuracy {acc:.2f} (chance 0.25; 8+/16 has p < 0.03 under chance)"
        assert all(r[2] >= 0.75 * r[3] for r in results), "too many epochs rejected"
        assert max(r[4] for r in results) <= P.P3000_CAP + 1e-6
        mark("session_end")
        code, body = T.get(f"/api/eeg/session.csv?session_id={sid}")
        rows = list(csv.reader(io.StringIO(body.decode())))
        assert rows[0] == ["lsl_time", "page_t_ms", "TP9", "AF7", "AF8", "TP10"], rows[0]
        assert len(rows) > 256 * 20 and len(rows[1]) == 6
        print(f"   accuracy {acc:.0%} vs chance 25% (SIMULATED), session CSV has 4 channels ({len(rows) - 1} samples): OK")
        ms.close()
        ps.close()
    finally:
        bridge.terminate()
        fake.terminate()
        try:
            out = bridge.communicate(timeout=5)[0]
        except Exception:
            out = ""
        if os.environ.get("CORTEXT_DEBUG"):
            print(out)


def main():
    offline()
    tmp = tempfile.mkdtemp()
    try:
        models_dir, filters_dir = os.path.join(tmp, "models"), os.path.join(tmp, "filters")
        os.makedirs(os.path.join(models_dir, "library"))
        os.makedirs(os.path.join(models_dir, "active"))
        shutil.copytree(os.path.join(ROOT, "filters", "library"), os.path.join(filters_dir, "library"))
        os.makedirs(os.path.join(filters_dir, "active"))
        lib = os.path.join(models_dir, "library")
        F3.build().save(os.path.join(lib, F3.NAME))
        F3.build_muse().save(os.path.join(lib, F3.MUSE_NAME))
        shutil.copy(os.path.join(lib, F3.NAME), os.path.join(models_dir, "active", F3.NAME))   # like the demo laptop
        for wall in (False, True):
            live(wall, models_dir, filters_dir)
        print("Muse pipeline OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

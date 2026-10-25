#!/usr/bin/env python3
"""Raw EEG to raw_eeg/ (Experiment -> Lab & data -> "Raw EEG -> Hugging Face"), through the real bridge and a simulated headset.

    python tests/test_raw_eeg.py

Checks: a free recording starts and stops over the API, writes every sample with unix_time + lsl_time + one uV column
per electrode, is refused while nothing streams or twice, is not listed as a session, and (being SIMULATED) is kept
locally and not uploaded; a file picked in the browser is stored untouched and the upload error (no token here) is
reported, not crashed. Hugging Face itself is covered by test_store.py's fake API. Needs pylsl, numpy, scipy.
"""
import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PORT = 9450 + (os.getpid() % 300)


def call(path, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, method="POST" if data is not None else "GET",
                                 headers={"Content-Type": "application/octet-stream" if raw is not None else "application/json"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read())


def main():
    tmp = tempfile.mkdtemp(prefix="cortext-raw-")
    env = {**os.environ, "CORTEXT_DATA_DIR": os.path.join(tmp, "data"), "CORTEXT_HF_REPO": "acme/cortext-test",
           "HF_TOKEN": "", "HUGGING_FACE_HUB_TOKEN": "", "HF_HOME": os.path.join(tmp, "hf")}
    bridge = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT)], env=env,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    fake = None
    try:
        for _ in range(100):
            try:
                call("/api/status")
                break
            except Exception:
                time.sleep(0.1)
        r = call("/api/storage/raw/start", {"participant": "P01"})
        if not r.get("ok"):
            assert "No headset stream" in r["error"], r
        fake = subprocess.Popen([sys.executable, os.path.join(ROOT, "eeg", "fake_eeg.py"), "--seed", "5"], env=env,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(150):
            if call("/api/status")["eeg"].get("found"):
                break
            time.sleep(0.2)
        assert not call("/api/storage/raw/start", {"participant": "Jude Cutler"})["ok"]      # names refused
        r = call("/api/storage/raw/start", {"participant": "P01", "note": "eyes open"})
        assert r["ok"] and r["recording"] and r["id"].startswith("rawrec-"), r
        assert not call("/api/storage/raw/start", {})["ok"]                                  # only one at a time
        time.sleep(3.0)
        st = call("/api/storage/raw")
        assert st["recording"]["recording"] and st["recording"]["samples"] > 400, st
        assert not any(s["session_id"].startswith("rawrec-") for s in call("/api/sessions")["sessions"])
        t_stop = time.time()
        r = call("/api/storage/raw/stop", {"upload": True})
        assert r["ok"] and not r["uploaded"] and "Simulated" in r["upload_error"], r
        path = os.path.join(tmp, "data", "raw_eeg", os.path.basename(r["local_path"]))
        assert path.endswith("_P01_SIM_raw.csv.gz") and os.path.exists(path + ".json"), r
        rows = gzip.open(path, "rt").read().splitlines()
        head = rows[0].split(",")
        assert head == ["unix_time", "lsl_time", "Fz_uV", "C3_uV", "Cz_uV", "C4_uV", "Pz_uV", "PO7_uV", "POz_uV", "PO8_uV"], head
        assert len(rows) - 1 == r["samples"] and r["samples"] > 600, r
        last = [float(x) for x in rows[-1].split(",")]
        assert abs(last[0] - t_stop) < 2.0, (last[0], t_stop)                                 # unix_time is real wall time
        ts = [float(x.split(",")[1]) for x in rows[1:]]
        assert all(b > a for a, b in zip(ts, ts[1:])), "LSL times must increase"
        meta = json.load(open(path + ".json"))
        assert meta["simulated"] and meta["participant_id"] == "P01" and meta["note"] == "eyes open", meta
        assert not any(f.startswith("eeg_rawrec-") for f in os.listdir(os.path.join(ROOT, "bridge", "logs"))), "logs copy left behind"
        assert call("/api/storage/raw")["files"][0]["simulated"] is True
        assert not call("/api/storage/raw/push", {"name": os.path.basename(path)})["ok"]      # simulated never uploads
        assert not call("/api/storage/raw/stop", {})["ok"]
        print(f"free recording: ok ({r['samples']} samples in {r['duration_s']} s, unix_time + lsl_time + 8 uV columns, simulated kept local)")

        blob = b"time,Fz,Pz\n0.000,1.0,2.0\n0.004,1.1,2.1\n"
        r = call("/api/storage/raw-upload?name=unicorn%20recorder.csv&note=x&participant=P02", raw=blob)
        assert not r["ok"] and "token" in r["error"].lower(), r                               # no token in this test
        kept = [f for f in os.listdir(os.path.join(tmp, "data", "raw_eeg")) if f.endswith("_unicorn_recorder.csv")]
        assert kept and open(os.path.join(tmp, "data", "raw_eeg", kept[0]), "rb").read() == blob, kept
        r = call("/api/storage/raw-upload?name=virus.exe", raw=b"MZ")
        assert not r["ok"] and "not an EEG file type" in r["error"], r
        print("file upload: ok (kept untouched, missing token reported, wrong types refused)")
        print("raw EEG OK")
    finally:
        for p in (bridge, fake):
            if p:
                p.terminate()
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()

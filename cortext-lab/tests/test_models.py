#!/usr/bin/env python3
"""Trained-model library + active folder: startup precedence, /api/models listing, live switching, uploads, saving a
calibration to the library, path safety. Uses a temporary models folder (CORTEXT_MODELS_DIR). Run: python tests/test_models.py"""
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bridge"))
import p300 as P  # noqa: E402

PORT = 9300 + (os.getpid() % 150)
BASE = f"http://127.0.0.1:{PORT}"
LABELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"]


def make_model(path, participants, simulated=False, labels=LABELS, live=False):
    rng = np.random.default_rng(len(path))
    d = len(labels) * P.n_bins()
    X = rng.normal(size=(80, d))
    y = np.array([1] * 20 + [0] * 60)
    X[y == 1] += 0.5
    meta = ({"trained_on": "live session", "session_id": "S-live", "participant_id": participants[0], "simulated": simulated,
             "n_trials": 10, "n_epochs": 80} if live else
            {"trained_on": ["S1", "S2"], "participants": participants, "simulated": simulated, "n_epochs": 80, "n_trials": 20,
             "cv_answer_accuracy": 0.7, "shuffled_accuracy": 0.26, "chance": 0.25})
    m = P.LinearP300.fit(X, y, labels, meta=meta)
    m.save(path)
    return path


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.loads(r.read().decode())


def post(path, obj, origin=None):
    h = {"Content-Type": "application/json"}
    if origin:
        h["Origin"] = origin
    req = urllib.request.Request(BASE + path, data=json.dumps(obj).encode(), headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def start_bridge(env, *extra):
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT), "--no-lsl", "--no-csv", *extra],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(100):
        try:
            get("/api/status")
            return proc
        except Exception:
            time.sleep(0.1)
    proc.terminate()
    raise SystemExit("bridge did not start:\n" + proc.stdout.read())


def stop(proc):
    proc.terminate()
    out = proc.communicate(timeout=5)[0]
    return out


def main():
    import faulthandler; faulthandler.dump_traceback_later(150, exit=True)   # never hang CI
    tmp = tempfile.mkdtemp(prefix="cortext-models-")
    lib, act = os.path.join(tmp, "library"), os.path.join(tmp, "active")
    os.makedirs(lib)
    os.makedirs(act)
    env = {**os.environ, "CORTEXT_MODELS_DIR": tmp}
    log_dir = os.path.join(ROOT, "bridge", "logs")
    os.makedirs(log_dir, exist_ok=True)
    calib = os.path.join(log_dir, f"p300_adaptive_TESTMODELS{os.getpid()}.npz")
    try:
        make_model(os.path.join(lib, "P01_20261025-0900.npz"), ["P01"])
        make_model(os.path.join(lib, "P02_20261025-0930_SIM.npz"), ["P02"], simulated=True)
        make_model(os.path.join(lib, "P03_other_headset.npz"), ["P03"], labels=["T7", "T8", "Cz", "Pz"])
        make_model(calib, ["P09"], live=True)
        shutil.copyfile(os.path.join(lib, "P01_20261025-0900.npz"), os.path.join(act, "P01_20261025-0900.npz"))

        # 1) startup: the file in active/ is loaded without any flag
        proc = start_bridge(env)
        info = get("/api/models")
        run = info["running"]
        assert run["model_file"].endswith("P01_20261025-0900.npz") and run["model_source"] == "models/active", run
        assert run["model_info"]["participants"] == ["P01"], run
        names = {m["file"]: m for m in info["library"]}
        assert set(names) == {"P01_20261025-0900.npz", "P02_20261025-0930_SIM.npz", "P03_other_headset.npz"}, names
        assert names["P01_20261025-0900.npz"]["is_active"] and not names["P02_20261025-0930_SIM.npz"]["is_active"]
        assert names["P02_20261025-0930_SIM.npz"]["simulated"] is True
        assert names["P01_20261025-0900.npz"]["cv_answer_accuracy"] == 0.7 and names["P01_20261025-0900.npz"]["n_epochs"] == 80
        cal = [m for m in info["calibration"] if m["file"] == os.path.basename(calib)]
        assert cal and cal[0]["kind"] == "calibration" and cal[0]["participants"] == ["P09"], info["calibration"]

        # 2) switch live from the website: the scorer swaps and active/ holds exactly that file.
        # A model someone put ONLY in active/ is kept (moved into the library), never deleted.
        make_model(os.path.join(act, "ONLY_IN_ACTIVE.npz"), ["P07"])
        code, r = post("/api/models/select", {"id": "library/P02_20261025-0930_SIM.npz"})
        assert code == 200 and r["ok"] and r["running"]["model_file"].endswith("P02_20261025-0930_SIM.npz"), r
        assert r["running"]["model_source"] == "Setup page"
        assert os.listdir(act) == ["P02_20261025-0930_SIM.npz"], os.listdir(act)
        assert os.path.exists(os.path.join(lib, "ONLY_IN_ACTIVE.npz")), "a model only in active/ must be kept in the library"
        assert not [n for n in os.listdir(lib) if n.startswith("P01_20261025-0900-")], "identical copy already in library: no duplicate"
        st = get("/api/status")["eeg"]["scorer"]
        assert st["model_file"].endswith("P02_20261025-0930_SIM.npz"), st

        # a calibration from this laptop can run directly, and be saved into the library
        code, r = post("/api/models/select", {"id": f"calibration/{os.path.basename(calib)}"})
        assert r["ok"] and r["running"]["model_info"]["participants"] == ["P09"], r
        code, r = post("/api/models/save-calibration", {"id": f"calibration/{os.path.basename(calib)}"})
        assert r["ok"] and r["id"].startswith("library/P09_") and r["id"].endswith("_calibration.npz"), r
        assert os.path.exists(os.path.join(lib, r["id"].split("/", 1)[1]))

        # no trained model: active/ emptied, the scorer adapts to the session instead
        code, r = post("/api/models/select", {"id": None})
        assert r["ok"] and r["running"]["model_file"] is None and r["running"]["active"] != "model", r
        assert [n for n in os.listdir(act) if n.endswith(".npz")] == []

        # 3) uploads: a valid model lands in the library (never overwriting), junk is refused
        data = open(os.path.join(lib, "P01_20261025-0900.npz"), "rb").read()
        code, r = post("/api/models/upload", {"filename": "P01_20261025-0900.npz", "data": base64.b64encode(data).decode()})
        assert r["ok"] and r["id"] == "library/P01_20261025-0900-2.npz", r
        code, r = post("/api/models/upload", {"filename": "evil.npz", "data": base64.b64encode(b"not a model").decode()})
        assert not r["ok"] and "not a usable" in r["error"], r
        assert not any(n.startswith("evil") or n.startswith("tmp") for n in os.listdir(lib)), os.listdir(lib)
        code, r = post("/api/models/upload", {"filename": "../../x.npz", "data": base64.b64encode(data).decode()})
        assert r["ok"] and r["id"] == "library/x.npz", r          # path stripped to a plain name inside the library

        # 4) safety: no path tricks, no cross-site switching
        for bad in ["library/../../bridge/bridge.py", "../x.npz", "library/", "secrets/x.npz", "library/missing.npz"]:
            code, r = post("/api/models/select", {"id": bad})
            assert not r["ok"], (bad, r)
        code, r = post("/api/models/select", {"id": "library/P01_20261025-0900.npz"}, origin="http://evil.example")
        assert code == 403, (code, r)
        out = stop(proc)

        # 5) precedence: --p300-model none ignores active/; a flag beats active/
        shutil.copyfile(os.path.join(lib, "P01_20261025-0900.npz"), os.path.join(act, "P01_20261025-0900.npz"))
        proc = start_bridge(env, "--p300-model", "none")
        run = get("/api/models")["running"]
        assert run["model_file"] is None and "none" in run["model_source"], run
        stop(proc)
        proc = start_bridge(env, "--p300-model", os.path.join(lib, "P02_20261025-0930_SIM.npz"))
        run = get("/api/models")["running"]
        assert run["model_file"].endswith("P02_20261025-0930_SIM.npz") and "command line" in run["model_source"], run
        stop(proc)

        # 6) two files in active/: newest wins, with a warning
        time.sleep(0.05)
        shutil.copyfile(os.path.join(lib, "P02_20261025-0930_SIM.npz"), os.path.join(act, "P02_20261025-0930_SIM.npz"))
        proc = start_bridge(env)
        info = get("/api/models")
        assert info["running"]["model_file"].endswith("P02_20261025-0930_SIM.npz") and "Keep only one" in info["active_warning"], info
        out = stop(proc)
        assert "WARNING" in out and "Keep only one" in out, out
        print("models OK: active/ loaded at startup, live switch (library, calibration, none), save calibration, "
              "switching never deletes a model, uploads (validated, no overwrite, name sanitised), path safety + CSRF, flag precedence, duplicate warning")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for f in os.listdir(log_dir):
            if f.startswith(f"p300_adaptive_TESTMODELS{os.getpid()}"):
                os.remove(os.path.join(log_dir, f))


if __name__ == "__main__":
    main()

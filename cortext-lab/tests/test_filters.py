#!/usr/bin/env python3
"""Filter chains (bridge/filters.py): the default chain is bit-identical to the filter every model was trained with,
chunked == one pass, layers do what they say, validation, signatures, model <-> chain compatibility, the library /
active folders, the engine's live switch (re-filtered buffer, model paused on a mismatch, calibration rebuilt) and
the /api/filters endpoints. Uses temporary filters + models folders. Run: python tests/test_filters.py"""
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
sys.path.insert(0, os.path.join(ROOT, "tests"))
TMP = tempfile.mkdtemp(prefix="cortext-filters-")
os.environ["CORTEXT_FILTERS_DIR"] = os.path.join(TMP, "filters")
os.environ["CORTEXT_MODELS_DIR"] = os.path.join(TMP, "models")
import filters as F  # noqa: E402
import p300 as P  # noqa: E402
import eeg_engine  # noqa: E402

LABELS = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "POz", "PO8"]
SR = 250.0
NOTCH = {"name": "Notch + band", "layers": [{"type": "notch", "freq": 60}, {"type": "bandpass", "low": 0.5, "high": 20, "order": 2}]}
PORT = 9450 + (os.getpid() % 150)
BASE = f"http://127.0.0.1:{PORT}"


def signal(seconds=20, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = rng.normal(0, 5, (len(t), len(LABELS))) + 3000.0                 # DC offset like a real headset
    x += 40 * np.sin(2 * np.pi * 60 * t)[:, None]                        # mains hum
    x += 10 * np.sin(2 * np.pi * 10 * t)[:, None] * np.linspace(0.5, 1.5, len(LABELS))   # alpha
    x += np.cumsum(rng.normal(0, 0.3, (len(t), len(LABELS))), axis=0)    # drift
    return t, x


def test_default_is_the_old_filter():
    _, x = signal()
    old = P.CausalFilter(SR, len(LABELS))
    new = F.Chain(F.DEFAULT_SPEC, SR, LABELS)
    a = np.vstack([old(x[i:i + 37]) for i in range(0, len(x), 37)])
    b = np.vstack([new(x[i:i + 53]) for i in range(0, len(x), 53)])
    assert np.allclose(a, b, atol=1e-9), np.abs(a - b).max()
    print("default chain == the filter every existing model was trained with: ok")


def test_chunked_equals_one_pass():
    _, x = signal()
    spec = {"layers": [{"type": "notch", "freq": 60, "harmonics": 2}, {"type": "reref", "channels": []},
                       {"type": "highpass", "cutoff": 0.3}, {"type": "lowpass", "cutoff": 30, "order": 4}]}
    one = F.Chain(spec, SR, LABELS)(x)
    c = F.Chain(spec, SR, LABELS)
    chunked = np.vstack([c(x[i:i + 17]) for i in range(0, len(x), 17)])
    assert np.allclose(one, chunked, atol=1e-8)
    print("multi-layer chain: chunked == one pass (live == offline): ok")


def power_at(y, f0):
    seg = y[-int(8 * SR):, 4]
    seg = seg - seg.mean()
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / SR)
    return spec[np.argmin(abs(f - f0))]


def test_layers_do_what_they_say():
    _, x = signal()
    notch = F.Chain({"layers": [{"type": "notch", "freq": 60}]}, SR, LABELS)(x)
    assert 10 * np.log10(power_at(x, 60) / power_at(notch, 60)) > 30
    assert abs(10 * np.log10(power_at(x, 10) / power_at(notch, 10))) < 0.5      # alpha untouched
    car = F.Chain({"layers": [{"type": "reref", "channels": []}]}, SR, LABELS)(x)
    assert np.allclose(car.mean(axis=1), 0, atol=1e-6)
    ref = F.Chain({"layers": [{"type": "reref", "channels": ["Cz"]}]}, SR, LABELS)(x)
    assert np.allclose(ref[:, 2], 0)
    hp = F.Chain({"layers": [{"type": "highpass", "cutoff": 0.5}]}, SR, LABELS)(x)
    assert abs(hp[-int(5 * SR):].mean()) < 5         # DC offset of 3000 µV gone
    print("notch removes 60 Hz (>30 dB) and leaves alpha, re-reference, high-pass: ok")


def test_validation_and_signature():
    bad = [({"layers": [{"type": "magic"}]}, "unknown type"),
           ({"layers": [{"type": "bandpass", "low": 30, "high": 20}]}, "below high"),
           ({"layers": [{"type": "lowpass", "cutoff": 900}]}, "between"),
           ({"layers": "x"}, "layers"), ({"layers": [], "reject_uv": 1}, "artifact threshold")]
    for spec, want in bad:
        try:
            F.normalize(spec)
        except F.FilterError as exc:
            assert want in str(exc), (want, exc)
        else:
            raise AssertionError(f"accepted {spec}")
    for spec, want in [({"layers": [{"type": "lowpass", "cutoff": 124}]}, "half the sampling rate"),
                       ({"layers": [{"type": "reref", "channels": ["Oz"]}]}, "no Oz")]:
        try:
            F.Chain(spec, SR, LABELS)
        except F.FilterError as exc:
            assert want in str(exc), exc
        else:
            raise AssertionError(spec)
    a = F.normalize(NOTCH)
    b = {**a, "name": "renamed", "reject_uv": 150, "layers": a["layers"] + [{"type": "lowpass", "cutoff": 5, "enabled": False}]}
    assert F.same_chain(a, b), "name, reject threshold and disabled layers must not change the signature"
    assert not F.same_chain(a, F.DEFAULT_SPEC)
    assert F.summary(a) == ["Notch 60 Hz", "Band-pass 0.5-20 Hz (order 2)"], F.summary(a)
    print("validation, signatures, summaries: ok")


def make_model(path, chain=None, participant="P01"):
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, len(LABELS) * P.n_bins()))
    y = np.array([1] * 15 + [0] * 45)
    X[y == 1] += 0.4
    m = P.LinearP300.fit(X, y, LABELS, meta={"participants": [participant], "trained_on": ["S1"]}, filters=chain)
    m.save(path)
    return path


def test_models_carry_their_chain():
    p1 = make_model(os.path.join(TMP, "notch.npz"), NOTCH)
    m = P.LinearP300.load(p1)
    assert F.same_chain(m.filters, NOTCH) and m.filters["name"] == "Notch + band"
    # a file saved before chains existed (no filter_chain key) loads as the default chain
    legacy = os.path.join(TMP, "legacy.npz")
    z = dict(np.load(make_model(os.path.join(TMP, "tmp.npz")), allow_pickle=False))
    z.pop("filter_chain")
    np.savez(legacy, **z)
    assert F.same_chain(P.LinearP300.load(legacy).filters, F.DEFAULT_SPEC)
    z["band"] = np.array([1.0, 30.0])
    np.savez(legacy, **z)
    try:
        P.LinearP300.load(legacy)
    except ValueError as exc:
        assert "different filter settings" in str(exc)
    else:
        raise AssertionError("legacy model with another band must not load")
    print("models store their chain; legacy files = default chain: ok")


def test_folders():
    shutil.rmtree(F.filters_dir(), ignore_errors=True)
    spec, path, warn = F.active_chain()
    assert path is None and F.same_chain(spec, F.DEFAULT_SPEC) and not warn
    a = F.save_to_library(NOTCH)
    b = F.save_to_library(NOTCH)
    assert a == "library/notch-band.json" and b == "library/notch-band-2.json", (a, b)
    assert F.library_match({**NOTCH, "name": "other"}) in (a, b)
    F.set_active(F.spec_for(a))
    spec, path, _ = F.active_chain()
    assert path.endswith("notch-band.json") and F.same_chain(spec, NOTCH)
    F.set_active(F.DEFAULT_SPEC)                    # back to default: the folder is emptied
    assert F.active_chain()[1] is None
    gone = F.delete_from_library(b)
    assert F.same_chain(gone, NOTCH) and not os.path.exists(os.path.join(F.library_dir(), "notch-band-2.json"))
    for bad in ("library/../x.json", "active/x.json", "nope"):
        try:
            F.delete_from_library(bad)
        except F.FilterError:
            pass
        else:
            raise AssertionError(bad)
    os.makedirs(F.active_dir(), exist_ok=True)
    for n in ("a.json", "b.json"):
        with open(os.path.join(F.active_dir(), n), "w") as fh:
            json.dump(F.DEFAULT_SPEC, fh)
    assert "2 files" in F.active_chain()[2]
    shutil.rmtree(F.active_dir())
    print("library save (no overwrite), match, active, delete, path safety: ok")


class _Clock:
    @staticmethod
    def local_clock():
        return time.perf_counter()


class _State:
    clock_offset_ms = None


def engine_with_data(spec=None):
    eng = eeg_engine.EEGEngine(_State(), _Clock(), filter_spec=spec, log_dir=TMP)
    eng.labels, eng.srate = list(LABELS), SR
    eng.qfilter = P.CausalFilter(SR, len(LABELS))
    eng.filter = eng._build_chain(eng.filter_spec)
    t, x = signal(30)
    for i in range(0, len(x), 25):
        eng._ingest(x[i:i + 25].tolist(), (t[i:i + 25] + 1000).tolist())
    return eng, x


def test_engine_switch():
    eng, x = engine_with_data()
    assert len(eng.filt) == len(x) and np.allclose(eng.filt, F.Chain(F.DEFAULT_SPEC, SR, LABELS)(x))
    q_before = eng.qfilt.copy()
    model = make_model(os.path.join(TMP, "default.npz"))
    eng.scorer.set_model(model, quiet=True)
    assert eng.scorer.filters_match() and eng.scorer._pick()[0] == "model"
    eng.set_filters(NOTCH)
    assert np.allclose(eng.filt, F.Chain(NOTCH, SR, LABELS)(eng.raw)), "buffer must be re-filtered with the new chain"
    assert np.array_equal(eng.qfilt, q_before), "electrode quality keeps its fixed reference filter"
    st = eng.scorer.status()
    assert not eng.scorer.filters_match() and st["active"] != "model" and "different filters" in st["model_filter_mismatch"]
    assert eng.filter_status()["summary"] == ["Notch 60 Hz", "Band-pass 0.5-20 Hz (order 2)"]
    eng.set_filters({**F.DEFAULT_SPEC, "name": "same layers, new name"})
    assert eng.scorer.filters_match() and eng.scorer._pick()[0] == "model"
    try:
        eng.set_filters({"layers": [{"type": "reref", "channels": ["Oz"]}]})
    except F.FilterError:
        pass
    else:
        raise AssertionError("a chain that can't run on this headset must be refused")
    assert eng.scorer.filters_match(), "a refused chain must leave the running one in place"
    # a chain chosen before the headset connected that can't run on it: default runs, error reported
    eng2, _ = engine_with_data({"name": "bad", "layers": [{"type": "reref", "channels": ["Oz"]}]})
    assert eng2.filter_error and F.same_chain(eng2.filter.spec, F.DEFAULT_SPEC)
    print("engine: live switch re-filters the buffer, pauses a mismatched model, refuses chains it can't run: ok")


def test_preview():
    eng, _ = engine_with_data()
    ts, raw, labels, sr = eng.raw_window(20)
    r = F.preview(NOTCH, ts, raw, labels, sr, "Pz")
    assert r["ok"] and r["channel"] == "Pz" and len(r["stages"]) == 3
    assert [s["label"] for s in r["stages"]] == ["Raw input", "Mains notch", "Band-pass"]
    f = np.array(r["stages"][0]["psd_f"])
    i60 = np.argmin(abs(f - 60))
    assert r["stages"][0]["psd_db"][i60] - r["stages"][1]["psd_db"][i60] > 20
    assert r["reject_estimate"]["windows"] > 5
    print("preview: per-stage traces and spectra, notch visible at 60 Hz: ok")


def http(path, body=None):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def start_bridge(*extra):
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT), "--no-lsl", "--no-csv", *extra],
                            env=dict(os.environ), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for _ in range(100):
        try:
            http("/api/filters")
            return proc
        except Exception:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError(proc.stdout.read())


def test_http():
    shutil.rmtree(F.filters_dir(), ignore_errors=True)
    shutil.rmtree(os.environ["CORTEXT_MODELS_DIR"], ignore_errors=True)
    nid = F.save_to_library(NOTCH)
    lib = os.path.join(os.environ["CORTEXT_MODELS_DIR"], "library")
    os.makedirs(lib)
    make_model(os.path.join(lib, "P01_notch.npz"), NOTCH)
    make_model(os.path.join(lib, "P01_default.npz"))
    proc = start_bridge()
    try:
        info = http("/api/filters")
        assert info["running"]["name"] == "P300 default" and "bandpass" in info["layer_types"]
        assert info["epoching"]["bin_ms"] == P.BIN_MS and info["offline_cleaning"]["zero_phase"]
        assert [c["id"] for c in info["library"]] == [nid]
        r = http("/api/filters/select", {"id": nid})
        assert r["ok"] and r["running"]["summary"][0] == "Notch 60 Hz" and r["active"]["file"] == "notch-band.json"
        ms = http("/api/models")
        fit = {m["file"]: m["fits_filters"] for m in ms["library"]}
        assert fit == {"P01_notch.npz": True, "P01_default.npz": False}, fit
        # picking a model switches the filters to the chain it was trained with
        r = http("/api/models/select", {"id": "library/P01_default.npz"})
        assert r["ok"] and r["filters_switched"] and "Band-pass" in r["note"], r
        assert http("/api/filters")["running"]["summary"] == ["Band-pass 0.5-20 Hz (order 2)"]
        r = http("/api/filters/apply", {"spec": {"name": "bad", "layers": [{"type": "bandpass", "low": 9, "high": 2}]}})
        assert not r["ok"] and "below high" in r["error"]
        r = http("/api/filters/save", {"spec": NOTCH, "name": "Second"})
        assert r["ok"] and r["id"] == "library/second.json"
        r = http("/api/filters/delete", {"id": "library/second.json"})
        assert r["ok"] and r["deleted"]["name"] == "Second"
        r = http("/api/filters/preview", {"spec": NOTCH})
        assert not r["ok"] and "EEG" in r["error"]
    finally:
        proc.kill()
        proc.wait()
    proc = start_bridge("--filters", os.path.join(F.library_dir(), "notch-band.json"))
    try:
        assert http("/api/filters")["running"]["source"].startswith("command line")
    finally:
        proc.kill()
        proc.wait()
    print("bridge /api/filters: list, select, model follows its chain, validation, save, delete, --filters: ok")


if __name__ == "__main__":
    try:
        test_default_is_the_old_filter()
        test_chunked_equals_one_pass()
        test_layers_do_what_they_say()
        test_validation_and_signature()
        test_models_carry_their_chain()
        test_folders()
        test_engine_switch()
        test_preview()
        test_http()
        print("ALL FILTER TESTS PASSED")
    finally:
        shutil.rmtree(TMP, ignore_errors=True)

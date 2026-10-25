"""Filter chains: the signal-cleaning layers between the headset and the P300 model, and which chain the bridge runs.

A chain is a small JSON file, an ordered list of layers applied to every incoming EEG sample:

    {"name": "P300 default", "description": "...", "reject_uv": 100,
     "layers": [{"type": "notch", "freq": 60, "q": 30, "harmonics": 1, "enabled": true},
                {"type": "bandpass", "low": 0.5, "high": 20, "order": 2, "enabled": true}]}

    cortext-lab/filters/library/   every chain the team has shared (in git)
    cortext-lab/filters/active/    the ONE chain the bridge runs (empty = the built-in default below)

Precedence at startup: --filters FILE > the file in filters/active/ > the built-in default. The Agent page edits and
switches chains while the bridge runs.

Rules that keep live and offline identical (same reason as p300.py):
  * every layer is CAUSAL and keeps its state between chunks, so a live stream and one offline pass give the same output;
  * a model is trained on features made by one chain and stores that chain. It is only used while the running chain
    has the same signature (the enabled layers and their settings); otherwise the scorer pauses it.
  * reject_uv (epoch artifact threshold) decides which epochs are scored, not what the features are, so it is not part
    of the signature.
Set CORTEXT_FILTERS_DIR to use another filters folder (tests do).
"""
import copy
import json
import os
import re
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(os.path.dirname(HERE), "filters")
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,120}\.json$")
MAX_LAYERS = 12

DEFAULT_SPEC = {
    "name": "P300 default",
    "description": "Causal 0.5-20 Hz band-pass, order 2: the filter every model so far was trained with. "
                   "Order 2 / 0.5 Hz keeps the P300 shape (1 Hz / order 4 flipped it).",
    "reject_uv": 100.0,
    "layers": [{"type": "bandpass", "low": 0.5, "high": 20.0, "order": 2, "enabled": True}],
}

# What each layer does and which settings it takes. The Agent page builds its editor from this.
LAYER_TYPES = {
    "bandpass": {
        "label": "Band-pass", "kind": "iir",
        "help": "Keeps one frequency band. Removes electrode drift below 'low' and muscle / mains noise above 'high'.",
        "params": {"low": {"label": "Low cut (Hz)", "min": 0.05, "max": 100, "step": 0.05, "default": 0.5},
                   "high": {"label": "High cut (Hz)", "min": 1, "max": 200, "step": 0.5, "default": 20.0},
                   "order": {"label": "Order", "min": 1, "max": 8, "step": 1, "default": 2, "int": True}}},
    "highpass": {
        "label": "High-pass", "kind": "iir",
        "help": "Removes slow drift (sweat, electrode settling) below the cut-off.",
        "params": {"cutoff": {"label": "Cut-off (Hz)", "min": 0.05, "max": 50, "step": 0.05, "default": 0.5},
                   "order": {"label": "Order", "min": 1, "max": 8, "step": 1, "default": 2, "int": True}}},
    "lowpass": {
        "label": "Low-pass", "kind": "iir",
        "help": "Removes fast activity (muscle, mains) above the cut-off.",
        "params": {"cutoff": {"label": "Cut-off (Hz)", "min": 1, "max": 200, "step": 0.5, "default": 20.0},
                   "order": {"label": "Order", "min": 1, "max": 8, "step": 1, "default": 2, "int": True}}},
    "notch": {
        "label": "Mains notch", "kind": "iir",
        "help": "Cuts the power-line hum (60 Hz in the US, 50 Hz in Europe) and optionally its harmonics.",
        "params": {"freq": {"label": "Mains (Hz)", "min": 40, "max": 70, "step": 10, "default": 60.0, "choices": [50, 60]},
                   "q": {"label": "Sharpness (Q)", "min": 5, "max": 100, "step": 1, "default": 30.0},
                   "harmonics": {"label": "Harmonics", "min": 1, "max": 4, "step": 1, "default": 1, "int": True}}},
    "reref": {
        "label": "Re-reference", "kind": "spatial",
        "help": "Subtracts a reference from every channel: the average of all channels (common average) or of the "
                "channels listed. Removes noise shared by every electrode.",
        "params": {"channels": {"label": "Reference channels", "default": [], "list": True,
                                "hint": "empty = common average of all channels"}}},
}


class FilterError(Exception):
    pass


# ---------------------------------------------------------------- spec validation / identity
def _num(v, p, where):
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise FilterError(f"{where}: {p['label']} must be a number")
    if x != x or x < p["min"] or x > p["max"]:
        raise FilterError(f"{where}: {p['label']} must be between {p['min']} and {p['max']}")
    return int(round(x)) if p.get("int") else round(x, 6)


def normalize(spec):
    """Validated copy of a chain spec with every default filled in. Raises FilterError."""
    if not isinstance(spec, dict):
        raise FilterError("a filter chain must be a JSON object")
    layers = spec.get("layers")
    if not isinstance(layers, list):
        raise FilterError("a filter chain needs a 'layers' list")
    if len(layers) > MAX_LAYERS:
        raise FilterError(f"at most {MAX_LAYERS} layers")
    out_layers = []
    for i, layer in enumerate(layers):
        where = f"layer {i + 1}"
        if not isinstance(layer, dict) or layer.get("type") not in LAYER_TYPES:
            raise FilterError(f"{where}: unknown type {layer.get('type') if isinstance(layer, dict) else layer!r} "
                              f"(choose {', '.join(LAYER_TYPES)})")
        t = LAYER_TYPES[layer["type"]]
        clean = {"type": layer["type"], "enabled": layer.get("enabled", True) is not False}
        for k, p in t["params"].items():
            v = layer.get(k, copy.deepcopy(p["default"]))
            if p.get("list"):
                if isinstance(v, str):
                    v = [s.strip() for s in v.split(",") if s.strip()]
                if not isinstance(v, list) or not all(isinstance(s, str) and s.strip() for s in v):
                    raise FilterError(f"{where}: {p['label']} must be a list of channel names")
                clean[k] = [s.strip() for s in v]
            else:
                clean[k] = _num(v, p, f"{where} ({t['label']})")
        if layer["type"] == "bandpass" and clean["low"] >= clean["high"]:
            raise FilterError(f"{where} (Band-pass): low cut must be below high cut")
        if layer.get("note"):
            clean["note"] = str(layer["note"])[:200]
        out_layers.append(clean)
    try:
        reject = float(spec.get("reject_uv", DEFAULT_SPEC["reject_uv"]))
    except (TypeError, ValueError):
        raise FilterError("reject_uv must be a number")
    if not 10 <= reject <= 2000:
        raise FilterError("artifact threshold (reject_uv) must be between 10 and 2000 µV")
    name = str(spec.get("name") or "Untitled chain").strip()[:80] or "Untitled chain"
    return {"name": name, "description": str(spec.get("description") or "")[:600], "reject_uv": round(reject, 3),
            "layers": out_layers}


def signature(spec):
    """What a model depends on: the enabled layers, in order, with their settings. Names and reject_uv don't count."""
    s = normalize(spec)
    core = [{k: v for k, v in layer.items() if k not in ("enabled", "note")} for layer in s["layers"] if layer["enabled"]]
    return json.dumps(core, sort_keys=True, separators=(",", ":"))


def same_chain(a, b):
    return signature(a) == signature(b)


def summary(spec):
    """One short line per enabled layer, e.g. ['Mains notch 60 Hz', 'Band-pass 0.5-20 Hz (order 2)']."""
    out = []
    for layer in normalize(spec)["layers"]:
        if not layer["enabled"]:
            continue
        t = layer["type"]
        if t == "bandpass":
            out.append(f"Band-pass {layer['low']:g}-{layer['high']:g} Hz (order {layer['order']})")
        elif t == "highpass":
            out.append(f"High-pass {layer['cutoff']:g} Hz (order {layer['order']})")
        elif t == "lowpass":
            out.append(f"Low-pass {layer['cutoff']:g} Hz (order {layer['order']})")
        elif t == "notch":
            h = layer["harmonics"]
            out.append(f"Notch {layer['freq']:g} Hz" + (f" + {h - 1} harmonic{'s' if h > 2 else ''}" if h > 1 else ""))
        elif t == "reref":
            out.append("Common average reference" if not layer["channels"] else f"Reference {'+'.join(layer['channels'])}")
    return out or ["(no filtering: raw signal)"]


# ---------------------------------------------------------------- the running filter
class _IIR:
    def __init__(self, sos):
        self.sos = sos
        self.zi = None

    def __call__(self, x):
        from scipy.signal import sosfilt, sosfilt_zi
        if self.zi is None:  # start from the first sample's value, avoiding a huge step response from the DC offset
            self.zi = sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


class _Reref:
    def __init__(self, idx):
        self.idx = idx

    def __call__(self, x):
        return x - x[:, self.idx].mean(axis=1, keepdims=True)


def _stage(layer, srate, labels):
    import numpy as np
    from scipy.signal import butter, iirnotch, tf2sos
    nyq = srate / 2.0
    t = layer["type"]

    def below_nyq(f, what):
        if f >= nyq * 0.98:
            raise FilterError(f"{LAYER_TYPES[t]['label']}: {what} {f:g} Hz is too close to half the sampling rate "
                              f"({nyq:g} Hz at {srate:g} Hz)")

    if t == "bandpass":
        below_nyq(layer["high"], "high cut")
        return _IIR(butter(layer["order"], [layer["low"], layer["high"]], btype="bandpass", fs=srate, output="sos"))
    if t == "highpass":
        below_nyq(layer["cutoff"], "cut-off")
        return _IIR(butter(layer["order"], layer["cutoff"], btype="highpass", fs=srate, output="sos"))
    if t == "lowpass":
        below_nyq(layer["cutoff"], "cut-off")
        return _IIR(butter(layer["order"], layer["cutoff"], btype="lowpass", fs=srate, output="sos"))
    if t == "notch":
        secs = []
        for k in range(1, layer["harmonics"] + 1):
            f0 = layer["freq"] * k
            if f0 >= nyq * 0.98:
                break           # harmonics above Nyquist don't exist in the signal
            b, a = iirnotch(f0, layer["q"], fs=srate)
            secs.append(tf2sos(b, a))
        if not secs:
            raise FilterError(f"Mains notch: {layer['freq']:g} Hz is above half the sampling rate ({nyq:g} Hz)")
        return _IIR(np.vstack(secs))
    if t == "reref":
        if layer["channels"]:
            missing = [c for c in layer["channels"] if c not in labels]
            if missing:
                raise FilterError(f"Re-reference: this headset has no {', '.join(missing)} (it has {', '.join(labels)})")
            return _Reref([labels.index(c) for c in layer["channels"]])
        return _Reref(list(range(len(labels))))
    raise FilterError(f"unknown layer type {t}")


class Chain:
    """Callable causal filter built from a spec, for one stream (sampling rate + channel labels).
    Keeps state between chunks. stages() also gives every intermediate output (used by the preview)."""

    def __init__(self, spec, srate, labels):
        self.spec = normalize(spec)
        self.srate = float(srate)
        self.labels = list(labels)
        self.signature = signature(self.spec)
        self.reject_uv = self.spec["reject_uv"]
        self._stages = [(layer, _stage(layer, self.srate, self.labels)) for layer in self.spec["layers"] if layer["enabled"]]

    def __call__(self, x):
        import numpy as np
        y = np.asarray(x, dtype=float)
        if y.size == 0:
            return y
        for _, st in self._stages:
            y = st(y)
        return y

    def stages(self, x):
        """[(layer or None for the input, output array)] after each enabled layer, from a fresh state."""
        import numpy as np
        y = np.asarray(x, dtype=float)
        out = [(None, y)]
        for layer, st in self._stages:
            y = st(y)
            out.append((layer, y))
        return out


# ---------------------------------------------------------------- folders: library + active
def filters_dir():
    return os.environ.get("CORTEXT_FILTERS_DIR") or DEFAULT_DIR


def library_dir():
    return os.path.join(filters_dir(), "library")


def active_dir():
    return os.path.join(filters_dir(), "active")


def _json_files(folder):
    try:
        names = [n for n in os.listdir(folder) if n.lower().endswith(".json") and not n.startswith(".")]
    except FileNotFoundError:
        return []
    return sorted(names, key=lambda n: os.path.getmtime(os.path.join(folder, n)), reverse=True)


def read_file(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return normalize(json.load(fh))
    except FilterError:
        raise
    except (OSError, ValueError) as exc:
        raise FilterError(f"could not read {os.path.basename(path)}: {exc}")


def active_chain():
    """(spec, path or None, warning). No file in active/ = the built-in default."""
    names = _json_files(active_dir())
    if not names:
        return normalize(DEFAULT_SPEC), None, None
    warn = f"{len(names)} files in filters/active/; using the newest ({names[0]}). Keep only one there." if len(names) > 1 else None
    path = os.path.join(active_dir(), names[0])
    try:
        return read_file(path), path, warn
    except FilterError as exc:
        return normalize(DEFAULT_SPEC), path, f"{exc}; running the built-in default instead"


def resolve_startup(arg):
    """Turn the --filters argument into (spec, source label, warning)."""
    if arg and arg.strip().lower() == "default":
        return normalize(DEFAULT_SPEC), "built-in default (--filters default)", None
    if arg:
        return read_file(arg), f"command line (--filters {os.path.basename(arg)})", None
    spec, path, warn = active_chain()
    return spec, ("filters/active" if path and not warn else "built-in default"), warn


def safe_name(name):
    base = re.sub(r"[^A-Za-z0-9._\-]+", "-", str(name or "chain").strip().lower()).strip("-._") or "chain"
    base = base[:100]
    return base if base.endswith(".json") else base + ".json"


def _write(folder, name, spec):
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".json", dir=folder)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2)
        fh.write("\n")
    dst = os.path.join(folder, name)
    os.replace(tmp, dst)
    return dst


def describe(spec, running_sig=None, **extra):
    s = normalize(spec)
    sig = signature(s)
    d = {"name": s["name"], "description": s["description"], "spec": s, "signature": sig, "summary": summary(s),
         "n_layers": sum(layer["enabled"] for layer in s["layers"]), **extra}
    if running_sig is not None:
        d["is_running"] = sig == running_sig
    return d


def listing(running_sig=None):
    lib = []
    for n in _json_files(library_dir()):
        p = os.path.join(library_dir(), n)
        try:
            lib.append(describe(read_file(p), running_sig, id=f"library/{n}", file=n, group="library",
                                modified=time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))))
        except FilterError as exc:
            lib.append({"id": f"library/{n}", "file": n, "group": "library", "name": n, "error": str(exc)})
    spec, path, warn = active_chain()
    return {"filters_dir": filters_dir(), "library": lib,
            "builtin": describe(DEFAULT_SPEC, running_sig, id="builtin/default", file=None, group="builtin"),
            "active": describe(spec, running_sig, id=f"active/{os.path.basename(path)}" if path else "builtin/default",
                               file=os.path.basename(path) if path else None),
            "active_warning": warn, "layer_types": LAYER_TYPES}


def spec_for(chain_id):
    group, _, name = str(chain_id or "").partition("/")
    if group == "builtin":
        return normalize(DEFAULT_SPEC)
    folder = {"library": library_dir(), "active": active_dir()}.get(group)
    if folder is None or not _NAME.match(name):
        raise FilterError(f"not a filter chain id: {chain_id!r}")
    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        raise FilterError(f"{chain_id} does not exist (refresh the list)")
    return read_file(path)


def set_active(spec):
    """Make this chain the one in filters/active/ (removing any other). The built-in default empties the folder.
    Returns the normalized spec."""
    s = normalize(spec)
    folder = active_dir()
    keep = None
    if not same_chain(s, DEFAULT_SPEC) or s["name"] != DEFAULT_SPEC["name"]:
        keep = safe_name(s["name"])
        _write(folder, keep, s)
    for n in _json_files(folder):
        if n != keep:
            os.remove(os.path.join(folder, n))
    return s


def save_to_library(spec, name=None, overwrite=False):
    """Write a chain into filters/library/. Without overwrite an existing file name gets -2, -3…  Returns the id."""
    s = normalize(spec)
    if name:
        s["name"] = str(name).strip()[:80] or s["name"]
    fname = safe_name(s["name"])
    folder = library_dir()
    if not overwrite:
        stem, i = fname[:-5], 2
        while os.path.exists(os.path.join(folder, fname)):
            fname = f"{stem}-{i}.json"
            i += 1
    _write(folder, fname, s)
    return f"library/{fname}"


def delete_from_library(chain_id):
    group, _, name = str(chain_id or "").partition("/")
    if group != "library" or not _NAME.match(name):
        raise FilterError("only chains in the library can be deleted")
    path = os.path.join(library_dir(), name)
    if not os.path.isfile(path):
        raise FilterError(f"{chain_id} does not exist")
    spec = read_file(path)
    os.remove(path)
    return spec


def library_match(spec):
    """Id of a library chain with the same signature, or None."""
    sig = signature(spec)
    for n in _json_files(library_dir()):
        try:
            if signature(read_file(os.path.join(library_dir(), n))) == sig:
                return f"library/{n}"
        except FilterError:
            pass
    return None


# ---------------------------------------------------------------- preview (Agent page)
def preview(spec, ts, raw, labels, srate, channel=None, show_s=4.0, out_hz=125.0):
    """Run `spec` on a stretch of raw EEG from a fresh state and describe the signal after every stage.

    The whole stretch (up to ~20 s) is filtered so slow high-pass transients settle; only the last `show_s` seconds
    are returned. Per stage: a trace of one channel (the raw input is de-meaned for display), each channel's std (µV)
    and the channel's spectrum. For the last stage also an estimate of how many P300 epochs reject_uv would reject.
    """
    import numpy as np
    from scipy.signal import welch
    chain = Chain(spec, srate, labels)
    ch = labels.index(channel) if channel in labels else (labels.index("Pz") if "Pz" in labels else 0)
    n_show = min(len(ts), int(show_s * srate))
    step = max(1, int(round(srate / out_hz)))
    nper = int(min(n_show, srate * 2))
    stages = []
    for layer, y in chain.stages(raw):
        tail = y[-n_show:]
        if layer is None:
            tail = tail - tail.mean(axis=0)
        f, pxx = welch(tail[:, ch], fs=srate, nperseg=nper) if nper >= 16 else (np.array([]), np.array([]))
        keep = (f >= 0.5) & (f <= min(80.0, srate / 2))
        stages.append({
            "label": "Raw input" if layer is None else LAYER_TYPES[layer["type"]]["label"],
            "summary": "headset signal, mean removed for display" if layer is None else summary({"layers": [layer]})[0],
            "trace": [round(float(v), 2) for v in tail[::step, ch]],
            "std_uv": [round(float(v), 2) for v in (tail - tail.mean(axis=0)).std(axis=0)],
            "psd_f": [round(float(v), 2) for v in f[keep]],
            "psd_db": [round(float(10 * np.log10(v + 1e-12)), 2) for v in pxx[keep]],
        })
    # how often would an epoch exceed the artifact threshold? (windows like P300 epochs, every 0.25 s)
    y = chain.stages(raw)[-1][1][-n_show:]
    win, hop, b0 = int(0.9 * srate), int(0.25 * srate), int(0.1 * srate)
    n_win = n_bad = 0
    for s0 in range(0, max(0, len(y) - win), hop):
        seg = y[s0:s0 + win]
        seg = seg - seg[:b0].mean(axis=0)
        n_win += 1
        n_bad += bool(np.abs(seg).max() > chain.reject_uv)
    return {"ok": True, "channel": labels[ch], "labels": labels, "srate": srate, "trace_hz": srate / step,
            "seconds": n_show / srate, "filtered_seconds": round(len(ts) / srate, 1), "stages": stages,
            "reject_uv": chain.reject_uv, "reject_estimate": {"windows": n_win, "rejected": n_bad},
            "signature": chain.signature}

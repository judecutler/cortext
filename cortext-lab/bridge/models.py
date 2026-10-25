"""Trained P300 models: where they live and which one the bridge runs.

    cortext-lab/models/library/   every trained model the team has shared (in git). train_p300.py saves here.
    cortext-lab/models/active/    the ONE model the bridge runs. Put a file here (or pick one on the Setup page)
                                  and the bridge uses it at startup; no --p300-model flag needed.
    cortext-lab/bridge/logs/p300_adaptive_*.npz   models calibrated live on this laptop (not in git)

Precedence at startup: --p300-model FILE > the file in models/active/ > none (adapt live, then heuristic).
--p300-model none ignores models/active/. The Setup page can switch models while the bridge runs.
Set CORTEXT_MODELS_DIR to use another models folder (tests do).
"""
import filecmp
import os
import re
import shutil
import tempfile
import time

import filters as F

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(os.path.dirname(HERE), "models")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,120}\.npz$")


class ModelError(Exception):
    pass


def models_dir():
    return os.environ.get("CORTEXT_MODELS_DIR") or DEFAULT_DIR


def library_dir():
    return os.path.join(models_dir(), "library")


def active_dir():
    return os.path.join(models_dir(), "active")


def _npz(folder):
    try:
        names = [n for n in os.listdir(folder) if n.lower().endswith(".npz") and not n.startswith(".")]
    except FileNotFoundError:
        return []
    return sorted(names, key=lambda n: os.path.getmtime(os.path.join(folder, n)), reverse=True)


def active_model():
    """(path, warning) of the model in models/active/, or (None, None). Several files: newest wins, with a warning."""
    names = _npz(active_dir())
    if not names:
        return None, None
    warn = None
    if len(names) > 1:
        warn = f"{len(names)} files in models/active/; using the newest ({names[0]}). Keep only one there."
    return os.path.join(active_dir(), names[0]), warn


def resolve_startup(arg):
    """Turn the --p300-model argument into (path or None, source label, warning)."""
    if arg and arg.strip().lower() == "none":
        return None, "none (--p300-model none)", None
    if arg:
        return arg, "command line (--p300-model)", None
    path, warn = active_model()
    if path:
        return path, "models/active", warn
    return None, "none", None


def describe(path, labels=None, filter_sig=None):
    """Metadata for one model file. labels = the live EEG channels, to say whether the model fits this headset;
    filter_sig = the running filter chain's signature, to say whether its filters match."""
    st = os.stat(path)
    info = {"file": os.path.basename(path), "size_kb": round(st.st_size / 1024, 1),
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)), "mtime": st.st_mtime}
    try:
        import p300 as P
        m = P.LinearP300.load(path)
    except Exception as exc:          # wrong settings, corrupt file, numpy missing
        info["error"] = str(exc)
        return info
    meta = m.meta or {}
    trained_on = meta.get("trained_on")
    sessions = [trained_on] if isinstance(trained_on, str) and trained_on != "live session" else (
        list(trained_on) if isinstance(trained_on, (list, tuple)) else [])
    if meta.get("session_id"):
        sessions = [meta["session_id"]]
    parts = meta.get("participants") or ([meta["participant_id"]] if meta.get("participant_id") else [])
    info.update({
        "channels": m.labels, "participants": [str(p) for p in parts if p], "sessions": sessions,
        "n_epochs": meta.get("n_epochs"), "n_trials": meta.get("n_trials"),
        "cv_answer_accuracy": meta.get("cv_answer_accuracy"), "cv_auc": meta.get("cv_auc"),
        "shuffled_accuracy": meta.get("shuffled_accuracy"), "chance": meta.get("chance"),
        "trained_at": meta.get("trained_at"), "simulated": meta.get("simulated"),
        "kind": "calibration" if trained_on == "live session" else "offline", "temperature": round(float(m.temperature), 2),
        "filters": m.filters, "filter_name": m.filters["name"], "filter_summary": F.summary(m.filters),
        "filter_signature": m.filter_signature,
    })
    if filter_sig is not None:
        info["fits_filters"] = m.filter_signature == filter_sig
    if labels:
        missing = [c for c in m.labels if c not in labels]
        info["fits_headset"] = not missing
        if missing:
            info["missing_channels"] = missing
    return info


def _entries(group, folder, labels, active_name=None, filter_sig=None):
    out = []
    for n in _npz(folder):
        d = describe(os.path.join(folder, n), labels, filter_sig)
        d["id"] = f"{group}/{n}"
        d["group"] = group
        if active_name is not None:
            d["is_active"] = n == active_name
        out.append(d)
    return out


def listing(log_dir, labels=None, filter_sig=None):
    path, warn = active_model()
    active_name = os.path.basename(path) if path else None
    return {
        "models_dir": models_dir(),
        "library": _entries("library", library_dir(), labels, active_name, filter_sig),
        "calibration": [d for d in _entries("calibration", log_dir, labels, None, filter_sig) if d["file"].startswith("p300_adaptive_")],
        "active": describe(path, labels, filter_sig) | {"id": f"active/{active_name}"} if path else None,
        "active_warning": warn,
    }


def path_for(model_id, log_dir):
    """Map an id from listing() ("library/x.npz", "calibration/x.npz", "active/x.npz") to a file path, safely."""
    group, _, name = str(model_id or "").partition("/")
    folder = {"library": library_dir(), "calibration": log_dir, "active": active_dir()}.get(group)
    if folder is None or not _NAME.match(name):
        raise ModelError(f"not a model id: {model_id!r}")
    path = os.path.join(folder, name)
    if not os.path.isfile(path):
        raise ModelError(f"{model_id} does not exist (refresh the list)")
    return path


def _validate(path):
    import p300 as P
    try:
        P.LinearP300.load(path)
    except Exception as exc:
        raise ModelError(f"not a usable CorText P300 model: {exc}")


def _copy_into(src, folder, name):
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=folder)
    os.close(fd)
    shutil.copyfile(src, tmp)
    dst = os.path.join(folder, name)
    os.replace(tmp, dst)
    return dst


def _retire(name):
    """Remove a file from active/, first copying it into the library unless an identical copy is already there,
    so switching models never loses one that was only put in active/."""
    src = os.path.join(active_dir(), name)
    lib = library_dir()
    if not any(filecmp.cmp(src, os.path.join(lib, n), shallow=False) for n in _npz(lib)):
        _copy_into(src, lib, _free_name(lib, name))
    os.remove(src)


def activate_file(path):
    """Make an arbitrary model file the active one (used by train_p300.py --activate). Returns the new path."""
    _validate(path)
    folder = active_dir()
    name = safe_name(os.path.basename(path))
    dst = _copy_into(path, folder, name)
    for n in _npz(folder):
        if n != name:
            _retire(n)
    return dst


def set_active(model_id, log_dir):
    """Make model_id the one in models/active/ (removing any other there). None = empty the folder. Returns the path."""
    folder = active_dir()
    if model_id is None:
        for n in _npz(folder):
            _retire(n)
        return None
    src = path_for(model_id, log_dir)
    _validate(src)
    name = os.path.basename(src)
    dst = src if os.path.dirname(os.path.abspath(src)) == os.path.abspath(folder) else _copy_into(src, folder, name)
    for n in _npz(folder):
        if n != name:
            _retire(n)
    return dst


def _free_name(folder, name):
    stem, ext = os.path.splitext(name)
    cand, i = name, 2
    while os.path.exists(os.path.join(folder, cand)):
        cand = f"{stem}-{i}{ext}"
        i += 1
    return cand


def safe_name(name):
    base = os.path.basename(str(name or "model.npz"))
    if not base.lower().endswith(".npz"):
        base += ".npz"
    base = re.sub(r"[^A-Za-z0-9._\-]", "_", base).lstrip("._-") or "model.npz"
    return base[-120:]


def add_to_library(data, filename):
    """Save uploaded bytes into models/library/ after checking it loads. Never overwrites: adds -2, -3…"""
    if len(data) > MAX_UPLOAD_BYTES:
        raise ModelError("file too large for a P300 model (over 10 MB)")
    folder = library_dir()
    os.makedirs(folder, exist_ok=True)
    fd, tmp = tempfile.mkstemp(suffix=".npz", dir=folder)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    try:
        _validate(tmp)
    except ModelError:
        os.remove(tmp)
        raise
    name = _free_name(folder, safe_name(filename))
    os.replace(tmp, os.path.join(folder, name))
    return f"library/{name}"


def save_calibration(model_id, log_dir, labels=None):
    """Copy a model calibrated on this laptop into the shared library, named <participant>_<date>_calibration.npz."""
    src = path_for(model_id, log_dir)
    d = describe(src, labels)
    if d.get("error"):
        raise ModelError(d["error"])
    who = "-".join(d.get("participants") or []) or "unknown"
    when = time.strftime("%Y%m%d-%H%M", time.localtime(d["mtime"]))
    name = _free_name(library_dir(), safe_name(f"{who}_{when}_calibration{'_SIM' if d.get('simulated') else ''}.npz"))
    _copy_into(src, library_dir(), name)
    return f"library/{name}"

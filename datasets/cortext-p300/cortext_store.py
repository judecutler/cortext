#!/usr/bin/env python3
"""Where CorText recordings live, and how they move between machines.

The decision (details in docs/DATA_STORAGE.md):

    GitHub                code only. No recordings, ever.
    Hugging Face dataset  the recordings: a private dataset repo, one folder per session.
    data/cortext-p300/    this machine's local copy of those folders (git-ignored).

Layout, identical locally and on Hugging Face:

    sessions/<session_id>/      real recordings   (meta.json, trials.csv, events.csv, epochs.npz, eeg_clean.npz, eeg_raw.npz)
    simulated/<session_id>/     fake_eeg.py runs  (local only; never uploaded unless you force it)
    voltages/<id>_raw.csv.gz    plain voltage table per session (uploaded with each session; "upload_voltages" in dataset.json)
    index.csv                   derived summary of the local sessions; rebuilt whenever sessions change
    sync_state.json             this machine's record of what it uploaded / downloaded

Every session is its own folder, so two laptops can upload at the same time without ever touching the same file.
There is deliberately no shared index file on the server: each machine builds it from the meta.json files it has.

Command line (from the repo root):

    python datasets/cortext-p300/cortext_store.py status           # where the data is, what is uploaded, is HF reachable
    python datasets/cortext-p300/cortext_store.py init             # one-off: create the private HF repo + dataset card
    python datasets/cortext-p300/cortext_store.py push [ID ...]    # upload sessions (default: every session not yet uploaded)
    python datasets/cortext-p300/cortext_store.py pull [--all]     # download sessions (default: epochs only, the small files)
    python datasets/cortext-p300/cortext_store.py voltages [ID ...] [--signal raw|clean] [--no-upload]
                                                                # one CSV per session: time + each electrode's µV + markers
    python datasets/cortext-p300/cortext_store.py delete ID ... [--remote]   # really delete (see below)
    python datasets/cortext-p300/cortext_store.py migrate          # move sessions from the old in-repo folder

Deleting is real: the files are removed from disk, and with --remote also from Hugging Face including its history
(delete the folder, then permanently delete the stored large files). Copies someone already downloaded stay with them.
"""
import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
CONFIG_FILE = HERE / "dataset.json"
CARD_FILE = HERE / "HF_DATASET_CARD.md"
LEGACY_ROOT = HERE                     # sessions used to be written next to this file (and committed to git)
SUBDIRS = ("sessions", "simulated")
INDEX_FILE = "index.csv"
SYNC_FILE = "sync_state.json"
INDEX_COLUMNS = ["session_id", "folder", "simulated", "bucket", "participant_id", "device", "recorded", "duration_s", "trials_valid",
                 "epochs_usable", "epochs_total", "bad_channels", "artifact_fraction", "pipeline_version", "size_mb",
                 "uploaded", "complete"]
# What `pull` fetches. "epochs" is enough to train a model and is a fraction of the size of "all".
PULL_PROFILES = {
    "epochs": ["meta.json", "trials.csv", "events.csv", "epochs.npz", "imagery.npz"],   # imagery.npz: only sessions with
    "all": ["meta.json", "trials.csv", "events.csv", "epochs.npz", "imagery.npz", "eeg_clean.npz", "eeg_raw.npz"],  # the imagery window
}
RAW_FILE = "eeg_raw.npz"
# Plain-text voltage tables: one gzipped CSV per session, in their own top-level folder (locally and on Hugging Face),
# for people who want "every electrode's voltage with its time" without numpy. See export_voltages().
VOLTAGE_DIR = "voltages"
VOLTAGE_SIGNALS = {"raw": "eeg_raw.npz", "clean": "eeg_clean.npz"}


class StorageError(RuntimeError):
    """Something the operator can fix (no token, repo not configured, session not found, ...)."""


# ------------------------------------------------------------------ configuration and paths
def load_config():
    cfg = {"name": "cortext-p300", "hf_repo": None, "hf_private": True,
           "upload_voltages": True, "voltage_signal": "raw"}   # also upload voltages/<id>_<signal>.csv.gz with each session
    try:
        cfg.update(json.loads(CONFIG_FILE.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass
    if os.environ.get("CORTEXT_HF_REPO"):
        cfg["hf_repo"] = os.environ["CORTEXT_HF_REPO"].strip()
    if isinstance(cfg.get("hf_repo"), str) and not cfg["hf_repo"].strip():
        cfg["hf_repo"] = None
    return cfg


def data_root(create=False):
    """This machine's copy of the dataset: $CORTEXT_DATA_DIR, else <repo>/data/cortext-p300 (git-ignored)."""
    override = os.environ.get("CORTEXT_DATA_DIR") or os.environ.get("CORTEXT_DATASET_DIR")  # the 2nd is the old name
    root = Path(override or (REPO_ROOT / "data" / load_config()["name"]))
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def safe_id(session_id):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(session_id))


def session_dir(session_id, simulated=False, root=None):
    return Path(root or data_root()) / ("simulated" if simulated else "sessions") / safe_id(session_id)


def find_session(session_id, root=None):
    """(path, simulated) of a local session folder, or None."""
    for sub in SUBDIRS:
        p = Path(root or data_root()) / sub / safe_id(session_id)
        if p.is_dir():
            return p, sub == "simulated"
    return None


def dir_size(path):
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


def display_path(path):
    """Repo-relative path when the path is inside the repo (nicer in the UI), else the absolute path."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# ------------------------------------------------------------------ sync state (what this machine has uploaded)
def load_sync_state(root=None):
    try:
        return json.loads((Path(root or data_root()) / SYNC_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_sync_state(state, root=None):
    root = Path(root or data_root(create=True))
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / (SYNC_FILE + ".tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, root / SYNC_FILE)


def _mark(session_id, root, **info):
    st = load_sync_state(root)
    st[safe_id(session_id)] = {**st.get(safe_id(session_id), {}), **info}
    save_sync_state(st, root)


def _forget(session_id, root):
    st = load_sync_state(root)
    if st.pop(safe_id(session_id), None) is not None:
        save_sync_state(st, root)


# ------------------------------------------------------------------ the local inventory
def scan_local(root=None):
    """One dict per session folder that has a meta.json, newest first."""
    root = Path(root or data_root())
    sync = load_sync_state(root)
    rows = []
    for sub in SUBDIRS:
        for meta_path in sorted((root / sub).glob("*/meta.json")):
            try:
                m = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            folder = meta_path.parent
            q = m.get("quality") or {}
            sid = m.get("session_id") or folder.name
            s = sync.get(folder.name, {})
            rows.append({
                "session_id": sid, "folder": f"{sub}/{folder.name}", "simulated": sub == "simulated", "bucket": m.get("bucket") or "",
                "participant_id": m.get("participant_id"), "device": m.get("device"), "recorded": m.get("created_at") or m.get("built_at"),
                "duration_s": q.get("duration_s"), "trials_valid": q.get("trials_valid"), "epochs_usable": q.get("epochs_usable"),
                "epochs_total": q.get("epochs_total"), "bad_channels": ";".join((q.get("bad_channels") or {}).keys()),
                "artifact_fraction": q.get("artifact_fraction"), "pipeline_version": m.get("pipeline_version"),
                "size_mb": round(dir_size(folder) / 1e6, 2), "uploaded": bool(s.get("uploaded_at") or s.get("downloaded_at")),
                "uploaded_at": s.get("uploaded_at"), "complete": (folder / RAW_FILE).exists(),
                "superseded_by": m.get("superseded_by") or "",
            })
    rows.sort(key=lambda r: str(r["recorded"] or ""), reverse=True)
    return rows


def set_bucket(session_id, bucket, root=None):
    """Label a session with a bucket such as "train", "test" or "pilot" ("" clears it).

    A bucket is only a label stored in meta.json, not a folder, so moving a session between buckets never moves files.
    The session then counts as not uploaded, because meta.json changed (only that small file is sent again).
    """
    root = Path(root or data_root())
    found = find_session(session_id, root)
    if not found:
        raise StorageError(f"Session {session_id} is not in {root}.")
    bucket = re.sub(r"\s+", "-", str(bucket or "").strip())[:40]
    meta_path = found[0] / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["bucket"] = bucket
    tmp = meta_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, meta_path)
    st = load_sync_state(root)
    if st.get(safe_id(session_id), {}).get("uploaded_at"):
        st[safe_id(session_id)].pop("uploaded_at")
        save_sync_state(st, root)
    rebuild_index(root)
    return bucket


def rebuild_index(root=None):
    """Write index.csv (a convenience for pandas / humans; the loader does not depend on it)."""
    root = Path(root or data_root(create=True))
    root.mkdir(parents=True, exist_ok=True)
    with open(root / INDEX_FILE, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(scan_local(root))
    return root / INDEX_FILE


# ------------------------------------------------------------------ Hugging Face
def _hf():
    try:
        import huggingface_hub
    except ImportError as exc:
        raise StorageError("huggingface_hub is not installed. Run: pip install huggingface_hub") from exc
    return huggingface_hub


def hf_token():
    """Token from $HF_TOKEN or the one saved by `hf auth login`; None when neither exists. Never stored in the repo."""
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var].strip()
    try:
        return _hf().get_token()
    except StorageError:
        return None


def _make_api():  # separate function so tests can swap in a fake
    return _hf().HfApi(token=hf_token())


def _api_and_repo():
    repo = load_config()["hf_repo"]
    if not repo:
        raise StorageError('No Hugging Face repo configured. Put its id in datasets/cortext-p300/dataset.json ("hf_repo": "your-org/cortext-p300") '
                           "or set CORTEXT_HF_REPO.")
    if not hf_token():
        raise StorageError("No Hugging Face token. Run `hf auth login` (or set HF_TOKEN) with a token that has write access to the dataset repo.")
    return _make_api(), repo


def hf_status(check=False):
    """Configuration facts, and with check=True one network round trip to see whether the repo is reachable."""
    cfg = load_config()
    out = {"repo": cfg["hf_repo"], "private": cfg["hf_private"], "configured": bool(cfg["hf_repo"]), "has_token": bool(hf_token()),
           "installed": True, "checked": False, "reachable": None, "user": None, "error": None, "remote_sessions": None}
    try:
        _hf()
    except StorageError as exc:
        out.update(installed=False, error=str(exc))
        return out
    if check and out["configured"] and out["has_token"]:
        out["checked"] = True
        try:
            api = _make_api()
            out["user"] = (api.whoami() or {}).get("name")
            out["remote_sessions"] = sorted(remote_session_ids(api, cfg["hf_repo"]))
            out["reachable"] = True
        except Exception as exc:  # network down, bad token, repo missing, ...
            out.update(reachable=False, error=_explain(exc))
    return out


def _explain(exc):
    name = type(exc).__name__
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    if name in ("RepositoryNotFoundError", "RevisionNotFoundError"):
        return "The Hugging Face repo was not found (or the token cannot see it). Run `cortext_store.py init`, or check the repo id and token."
    if name in ("HfHubHTTPError", "GatedRepoError") and "401" in text + str(getattr(exc, "response", "")):
        return "Hugging Face rejected the token (401). Run `hf auth login` again."
    if "403" in text:
        return "Hugging Face says this token may not do that (403). It needs write access to the dataset repo."
    return f"{name}: {text}" if text else name


_ID_RE = re.compile(r"^sessions/([^/]+)/meta\.json$")


def remote_session_ids(api=None, repo=None):
    if api is None:
        api, repo = _api_and_repo()
    files = api.list_repo_files(repo, repo_type="dataset")
    return {m.group(1) for f in files if (m := _ID_RE.match(f))}


def init_remote():
    """Create the (private) dataset repo if it does not exist and upload the dataset card."""
    api, repo = _api_and_repo()
    cfg = load_config()
    api.create_repo(repo, repo_type="dataset", private=bool(cfg["hf_private"]), exist_ok=True)
    if CARD_FILE.exists():
        api.upload_file(path_or_fileobj=str(CARD_FILE), path_in_repo="README.md", repo_id=repo, repo_type="dataset",
                        commit_message="Dataset card")
    return {"repo": repo, "private": bool(cfg["hf_private"]), "url": f"https://huggingface.co/datasets/{repo}"}


def push_session(session_id, root=None, allow_simulated=False, message=None):
    """Upload one local session folder to sessions/<id>/ on Hugging Face (replacing what was there)."""
    root = Path(root or data_root())
    found = find_session(session_id, root)
    if not found:
        raise StorageError(f"Session {session_id} is not in {root}. Build it first (it is built automatically when a session ends).")
    path, simulated = found
    if simulated and not allow_simulated:
        raise StorageError("Simulated sessions are not uploaded (they would pollute the shared dataset).")
    api, repo = _api_and_repo()
    sid = safe_id(session_id)
    try:
        # delete_patterns="*" clears stale files of an older build of this session in the same commit; identical files
        # are not re-uploaded (they are matched by content hash).
        info = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(path), path_in_repo=f"sessions/{sid}",
                                 commit_message=message or f"Add session {sid}", delete_patterns="*")
    except Exception as exc:
        raise StorageError(_explain(exc)) from exc
    _mark(sid, root, uploaded_at=time.strftime("%Y-%m-%dT%H:%M:%S"), repo=repo, commit=getattr(info, "oid", None) or str(info),
          size_mb=round(dir_size(path) / 1e6, 2))
    rebuild_index(root)
    out = {"session_id": session_id, "repo": repo, "path_in_repo": f"sessions/{sid}",
           "url": f"https://huggingface.co/datasets/{repo}/tree/main/sessions/{sid}"}
    cfg = load_config()
    if cfg.get("upload_voltages"):
        # The session itself is already safely uploaded; a problem with the extra table is reported, not raised.
        try:
            out["voltages"] = push_voltages(session_id, root=root, signal=cfg.get("voltage_signal") or "raw",
                                            allow_simulated=allow_simulated)
        except StorageError as exc:
            out["voltages"] = {"ok": False, "error": str(exc)}
    return out


# ------------------------------------------------------------------ plain voltage tables (CSV)
def voltage_file(session_id, signal="raw", root=None):
    """Local path of a session's voltage table: <data>/voltages/<id>_<signal>.csv.gz (same path inside the HF repo)."""
    return Path(root or data_root()) / VOLTAGE_DIR / f"{safe_id(session_id)}_{signal}.csv.gz"


def export_voltages(session_id, root=None, signal="raw", decimals=2):
    """Write every sample of every electrode, with its time, as one gzipped CSV. Returns facts about the file.

    Columns: time_s (seconds from the first sample), lsl_time (the clock used by events.csv and epochs' onset_lsl),
    one column per electrode in microvolts (e.g. Pz_uV), and event (markers that fell on that sample, e.g.
    "flash_on option=dog target=1"; several are joined with " | "; empty otherwise).
    signal "raw" = the untouched recording (contains DC offset and mains hum; times are the headset's own timestamps);
    "clean" = filtered, gap-split and resampled to an exact 250 Hz grid (what epochs.npz is cut from).
    Voltages are rounded to `decimals` places (0.01 µV is far below EEG noise, ~1 µV), which keeps the file small.
    """
    import gzip
    import io
    import numpy as np

    if signal not in VOLTAGE_SIGNALS:
        raise StorageError(f"signal must be one of {sorted(VOLTAGE_SIGNALS)}")
    root = Path(root or data_root())
    found = find_session(session_id, root)
    if not found:
        raise StorageError(f"Session {session_id} is not in {root}.")
    path, _ = found
    src = path / VOLTAGE_SIGNALS[signal]
    if not src.exists():
        raise StorageError(f"{src.name} is not on this machine for {session_id} "
                           f"(download it with: cortext_store.py pull --all {session_id}).")
    z = np.load(src, allow_pickle=False)
    data, times = np.asarray(z["data"], dtype=np.float64), np.asarray(z["times"], dtype=np.float64)
    labels = [str(l) for l in z["labels"]]
    if data.ndim != 2 or data.shape[1] != times.shape[0] or data.shape[0] != len(labels):
        raise StorageError(f"{src.name} has an unexpected shape {data.shape} (times {times.shape}, {len(labels)} labels).")

    # markers -> the nearest sample of THIS signal (events.csv's "sample" column indexes eeg_clean, not eeg_raw)
    marks = {}
    ev_file = path / "events.csv"
    if ev_file.exists() and len(times):
        with open(ev_file, newline="", encoding="utf-8") as fh:
            for m in csv.DictReader(fh):
                try:
                    t = float(m.get("lsl_time") or "")
                except ValueError:
                    continue
                i = int(np.searchsorted(times, t))
                if i > 0 and (i == len(times) or abs(times[i - 1] - t) <= abs(times[i] - t)):
                    i -= 1
                text = (m.get("event") or "").strip()
                if m.get("option_id"):
                    text += f" option={m['option_id']}"
                if str(m.get("is_target", "")).strip() in ("1", "True", "true"):
                    text += " target=1"
                marks.setdefault(i, []).append(text.replace(",", ";"))

    out = voltage_file(session_id, signal, root)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    block = np.column_stack([times - (times[0] if len(times) else 0.0), times, data.T])
    fmt = ["%.4f", "%.6f"] + [f"%.{int(decimals)}f"] * len(labels)
    header = ",".join(["time_s", "lsl_time"] + [f"{l}_uV" for l in labels] + ["event"])
    # mtime=0 -> the same recording always gives byte-identical files, so re-uploads are skipped by content hash
    with open(tmp, "wb") as raw_fh, gzip.GzipFile(fileobj=raw_fh, mode="wb", mtime=0, compresslevel=6) as gz:
        w = io.TextIOWrapper(gz, encoding="utf-8", newline="")
        w.write(header + "\n")
        chunk = 20000
        for s in range(0, len(block), chunk):
            buf = io.StringIO()
            np.savetxt(buf, block[s:s + chunk], fmt=fmt, delimiter=",")
            for k, line in enumerate(buf.getvalue().splitlines(), start=s):
                w.write(line + "," + " | ".join(marks.get(k, ())) + "\n")
        w.flush()
        w.detach()
    os.replace(tmp, out)
    return {"session_id": session_id, "signal": signal, "path": str(out), "samples": int(len(times)),
            "channels": labels, "events": sum(len(v) for v in marks.values()), "size_mb": round(out.stat().st_size / 1e6, 2),
            "duration_s": round(float(times[-1] - times[0]), 1) if len(times) else 0.0}


def push_voltages(session_id, root=None, signal="raw", allow_simulated=False):
    """Export a session's voltage table (if needed) and upload it to voltages/<id>_<signal>.csv.gz on Hugging Face."""
    root = Path(root or data_root())
    found = find_session(session_id, root)
    if found and found[1] and not allow_simulated:
        raise StorageError("Simulated sessions are not uploaded (they would pollute the shared dataset).")
    info = export_voltages(session_id, root=root, signal=signal)
    api, repo = _api_and_repo()
    rel = f"{VOLTAGE_DIR}/{Path(info['path']).name}"
    try:
        api.upload_file(path_or_fileobj=info["path"], path_in_repo=rel, repo_id=repo, repo_type="dataset",
                        commit_message=f"Voltage table for {safe_id(session_id)} ({signal})")
    except Exception as exc:
        raise StorageError(_explain(exc)) from exc
    return {"ok": True, **info, "path_in_repo": rel, "url": f"https://huggingface.co/datasets/{repo}/blob/main/{rel}"}


def pending_uploads(root=None):
    return [r["session_id"] for r in scan_local(root) if not r["simulated"] and not r["uploaded"] and r["complete"]]


def push_pending(root=None, on_progress=None):
    results = []
    for sid in pending_uploads(root):
        try:
            results.append({"ok": True, **push_session(sid, root)})
        except StorageError as exc:
            results.append({"ok": False, "session_id": sid, "error": str(exc)})
        if on_progress:
            on_progress(results[-1])
    return results


def pull(sessions=None, profile="epochs", root=None, voltages=False):
    """Download sessions from Hugging Face into the local data folder.

    profile "epochs" (default) fetches the small files you need to train; "all" also gets eeg_clean.npz and eeg_raw.npz.
    voltages=True also fetches each session's plain CSV voltage table(s) into <data>/voltages/.
    Existing local files are verified against the server and only changed ones are fetched again.
    """
    if profile not in PULL_PROFILES:
        raise StorageError(f"profile must be one of {sorted(PULL_PROFILES)}")
    api, repo = _api_and_repo()
    root = Path(root or data_root(create=True))
    root.mkdir(parents=True, exist_ok=True)
    try:
        ids = sorted(remote_session_ids(api, repo))
        if sessions:
            ids = [s for s in ids if s in {safe_id(x) for x in sessions}]
        if not ids:
            return {"repo": repo, "sessions": [], "note": "nothing to download"}
        patterns = [f"sessions/{sid}/{name}" for sid in ids for name in PULL_PROFILES[profile]]
        if voltages:
            patterns += [f"{VOLTAGE_DIR}/{sid}_{sig}.csv.gz" for sid in ids for sig in VOLTAGE_SIGNALS]
        _hf().snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(root), allow_patterns=patterns, token=hf_token())
    except StorageError:
        raise
    except Exception as exc:
        raise StorageError(_explain(exc)) from exc
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    for sid in ids:
        if (root / "sessions" / sid / "meta.json").exists():
            _mark(sid, root, downloaded_at=now, repo=repo, profile=profile)
    rebuild_index(root)
    return {"repo": repo, "sessions": ids, "profile": profile}


# ------------------------------------------------------------------ deleting
def log_files(session_id, log_dir):
    """The bridge's raw logs for one session: eeg_<id>_<stamp>.csv/.json, markers_<id>_<stamp>.csv, p300_adaptive_<id>.npz."""
    if not log_dir or not os.path.isdir(log_dir):
        return []
    sid = re.escape(safe_id(session_id))
    pat = re.compile(rf"^((eeg|markers)_{sid}_\d{{8}}-\d{{6}}\.(csv|json)|p300_adaptive_{sid}\.npz)$")
    return sorted(os.path.join(log_dir, f) for f in os.listdir(log_dir) if pat.match(f))


_LOG_RE = re.compile(r"^(?:eeg|markers)_(.+)_\d{8}-\d{6}\.(?:csv|json)$")


def scan_logs(log_dir):
    """{safe_session_id: {"files": n, "bytes": n}} for every session that has raw logs in the bridge's log folder."""
    out = {}
    if not log_dir or not os.path.isdir(log_dir):
        return out
    for f in os.listdir(log_dir):
        m = _LOG_RE.match(f)
        if m:
            d = out.setdefault(m.group(1), {"files": 0, "bytes": 0})
            d["files"] += 1
            try:
                d["bytes"] += os.path.getsize(os.path.join(log_dir, f))
            except OSError:
                pass
    return out


def delete_session(session_id, root=None, log_dir=None, remote=False, purge_remote=True):
    """Really delete a session from this machine, and with remote=True from Hugging Face as well.

    Returns a report that says exactly what was removed where, and what could not be.
    The browser's own copy (IndexedDB) is deleted by the web app; this function cannot reach it.
    """
    root = Path(root or data_root())
    sid = safe_id(session_id)
    report = {"session_id": session_id, "dataset_folders": [], "log_files": [], "bytes_freed": 0, "errors": [], "remote": None}

    for sub in SUBDIRS:
        for p in (root / sub / sid, root / sub / (sid + ".tmp")):
            if p.exists():
                size = dir_size(p)
                try:
                    shutil.rmtree(p)
                    report["dataset_folders"].append(display_path(p))
                    report["bytes_freed"] += size
                except OSError as exc:
                    report["errors"].append(f"could not delete {p}: {exc}")
    for signal in VOLTAGE_SIGNALS:
        p = voltage_file(sid, signal, root)
        if p.exists():
            try:
                report["bytes_freed"] += p.stat().st_size
                p.unlink()
                report["dataset_folders"].append(display_path(p))
            except OSError as exc:
                report["errors"].append(f"could not delete {p}: {exc}")
    for f in log_files(session_id, log_dir):
        try:
            size = os.path.getsize(f)
            os.remove(f)
            report["log_files"].append(os.path.basename(f))
            report["bytes_freed"] += size
        except OSError as exc:
            report["errors"].append(f"could not delete {os.path.basename(f)}: {exc} (is it still being written?)")
    _forget(sid, root)
    if root.exists():
        rebuild_index(root)

    if remote:
        report["remote"] = delete_remote(sid, purge=purge_remote)
        if not report["remote"]["ok"]:
            report["errors"].append("Hugging Face: " + report["remote"]["error"])
    report["ok"] = not report["errors"]
    return report


def delete_remote(session_id, purge=True):
    """Delete sessions/<id>/ from the Hugging Face repo; with purge, also erase the stored files from its history."""
    out = {"ok": False, "deleted": False, "purged_files": 0, "note": "", "error": ""}
    try:
        api, repo = _api_and_repo()
    except StorageError as exc:
        out["error"] = str(exc)
        return out
    sid = safe_id(session_id)
    prefix = f"sessions/{sid}"
    voltage_paths = {f"{VOLTAGE_DIR}/{sid}_{s}.csv.gz" for s in VOLTAGE_SIGNALS}
    try:
        for rel in sorted(voltage_paths & set(api.list_repo_files(repo, repo_type="dataset"))):
            api.delete_file(path_in_repo=rel, repo_id=repo, repo_type="dataset", commit_message=f"Delete voltage table {rel}")
            out["deleted"] = True
    except Exception as exc:
        out["error"] = _explain(exc)
        return out
    try:
        api.delete_folder(path_in_repo=prefix, repo_id=repo, repo_type="dataset", commit_message=f"Delete session {sid}")
        out["deleted"] = True
    except Exception as exc:
        if type(exc).__name__ in ("EntryNotFoundError", "RemoteEntryNotFoundError") or "404" in str(exc):
            out["ok"] = True
            if not out["deleted"]:
                out["note"] = "Not on Hugging Face (nothing to delete there)."
                return out
        else:
            out["error"] = _explain(exc)
            return out
    out["ok"] = True
    if purge:
        try:
            lfs = [f for f in api.list_lfs_files(repo, repo_type="dataset")
                   if str(getattr(f, "filename", "")).startswith(prefix + "/") or str(getattr(f, "filename", "")) in voltage_paths]
            if lfs:
                api.permanently_delete_lfs_files(repo, lfs, rewrite_history=True, repo_type="dataset")
            out["purged_files"] = len(lfs)
            out["note"] = f"Removed from the repo and permanently erased {len(lfs)} stored file(s) from its history."
        except Exception as exc:
            out["note"] = ("Removed from the repo, but the old versions may still be in its history "
                           f"({_explain(exc)}). Purge them under the repo's Settings > Storage on huggingface.co.")
    else:
        out["note"] = "Removed from the repo. Older versions stay in its history (purge was off)."
    return out


# ------------------------------------------------------------------ one-off migration from the old layout
def migrate_legacy(root=None):
    """Move sessions from datasets/cortext-p300/{sessions,simulated} (the old in-repo location) to the data folder."""
    root = Path(root or data_root(create=True))
    moved = []
    for sub in SUBDIRS:
        old = LEGACY_ROOT / sub
        if not old.is_dir() or old.resolve() == (root / sub).resolve():
            continue
        for p in sorted(old.iterdir()):
            dest = root / sub / p.name
            if p.is_dir() and not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(p), str(dest))
                moved.append(f"{sub}/{p.name}")
        try:
            old.rmdir()
        except OSError:
            pass
    legacy_index = LEGACY_ROOT / INDEX_FILE
    if legacy_index.exists() and legacy_index.resolve() != (root / INDEX_FILE).resolve():
        legacy_index.unlink()
    rebuild_index(root)
    return {"moved": moved, "root": str(root)}


# ------------------------------------------------------------------ command line
def _fmt_mb(x):
    return f"{x:.1f} MB" if x is not None else "?"


def _cmd_status(a):
    root = data_root()
    rows = scan_local(root)
    hf = hf_status(check=not a.offline)
    print(f"local data folder : {root}  ({'exists' if root.exists() else 'not created yet'})")
    print(f"local sessions    : {sum(not r['simulated'] for r in rows)} real, {sum(r['simulated'] for r in rows)} simulated, "
          f"{_fmt_mb(sum(r['size_mb'] or 0 for r in rows))}")
    print(f"not uploaded yet  : {len(pending_uploads(root))}")
    print(f"Hugging Face repo : {hf['repo'] or 'NOT CONFIGURED (set hf_repo in datasets/cortext-p300/dataset.json)'}")
    print(f"token             : {'found' if hf['has_token'] else 'missing (run: hf auth login)'}")
    if hf["checked"]:
        if hf["reachable"]:
            reach = "yes, as %s; %d session(s) on the server" % (hf["user"], len(hf["remote_sessions"]))
        else:
            reach = "NO - %s" % hf["error"]
        print("reachable         : " + reach)
    for r in rows:
        tag = "[sim] " if r["simulated"] else "      "
        where = "uploaded" if r["uploaded"] else "LOCAL ONLY"
        note = "" if r["complete"] else "  (no raw file: pulled with epochs only)"
        print("  %s%-34s %-8s %9s  %s%s" % (tag, r["session_id"], r["participant_id"] or "-", _fmt_mb(r["size_mb"]), where, note))


def _cmd_push(a):
    ids = a.sessions or pending_uploads()
    if not ids:
        print("nothing to upload")
        return
    for sid in ids:
        r = push_session(sid, allow_simulated=a.include_simulated)
        print(f"uploaded {sid} -> {r['url']}")


def _cmd_pull(a):
    r = pull(a.sessions or None, profile="all" if a.all else "epochs", voltages=a.voltages)
    print(f"downloaded {len(r['sessions'])} session(s) ({r.get('profile', '-')}) from {r['repo']} into {data_root()}")


def _cmd_voltages(a):
    ids = a.sessions or [r["session_id"] for r in scan_local() if not r["simulated"] or a.include_simulated]
    if not ids:
        print("no sessions on this machine")
        return
    failed = 0
    for sid in ids:
        try:
            if a.no_upload:
                r = export_voltages(sid, signal=a.signal)
                where = display_path(Path(r["path"]))
            else:
                r = push_voltages(sid, signal=a.signal, allow_simulated=a.include_simulated)
                where = r["url"]
            print(f"{sid}: {r['samples']} samples x {len(r['channels'])} electrodes, {r['duration_s']} s, "
                  f"{r['events']} markers, {_fmt_mb(r['size_mb'])} -> {where}")
        except StorageError as exc:
            failed += 1
            print(f"{sid}: skipped - {exc}")
    if failed:
        sys.exit(f"{failed} session(s) failed")


def _cmd_delete(a):
    if not a.yes:
        what = "from this machine AND from Hugging Face" if a.remote else "from this machine"
        if input(f"Permanently delete {', '.join(a.sessions)} {what}? Type 'delete' to confirm: ").strip() != "delete":
            sys.exit("cancelled")
    log_dir = os.path.join(REPO_ROOT, "cortext-lab", "bridge", "logs")
    for sid in a.sessions:
        r = delete_session(sid, log_dir=log_dir, remote=a.remote)
        print(f"{sid}: removed {len(r['dataset_folders'])} folder(s), {len(r['log_files'])} log file(s), {_fmt_mb(r['bytes_freed'] / 1e6)}"
              + (f"; Hugging Face: {r['remote']['note'] or r['remote']['error']}" if r["remote"] else "")
              + ("" if r["ok"] else f"  ERRORS: {r['errors']}"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("status"); p.add_argument("--offline", action="store_true", help="skip the Hugging Face network check"); p.set_defaults(fn=_cmd_status)
    sub.add_parser("init").set_defaults(fn=lambda a: print(json.dumps(init_remote(), indent=1)))
    p = sub.add_parser("push"); p.add_argument("sessions", nargs="*"); p.add_argument("--include-simulated", action="store_true"); p.set_defaults(fn=_cmd_push)
    p = sub.add_parser("pull"); p.add_argument("sessions", nargs="*"); p.add_argument("--all", action="store_true", help="also download eeg_clean.npz and eeg_raw.npz")
    p.add_argument("--voltages", action="store_true", help="also download the CSV voltage tables (voltages/<id>_<signal>.csv.gz)"); p.set_defaults(fn=_cmd_pull)
    p = sub.add_parser("voltages", help="write (and upload) one CSV per session: time + every electrode's voltage + markers")
    p.add_argument("sessions", nargs="*", help="default: every real session on this machine")
    p.add_argument("--signal", choices=sorted(VOLTAGE_SIGNALS), default="raw", help="raw (default, untouched) or clean (filtered)")
    p.add_argument("--no-upload", action="store_true", help="only write the file locally (<data>/voltages/)")
    p.add_argument("--include-simulated", action="store_true"); p.set_defaults(fn=_cmd_voltages)
    p = sub.add_parser("delete"); p.add_argument("sessions", nargs="+"); p.add_argument("--remote", action="store_true"); p.add_argument("--yes", action="store_true"); p.set_defaults(fn=_cmd_delete)
    sub.add_parser("migrate").set_defaults(fn=lambda a: print(json.dumps(migrate_legacy(), indent=1)))
    sub.add_parser("index").set_defaults(fn=lambda a: print(f"wrote {rebuild_index()}"))
    a = ap.parse_args(argv)
    try:
        a.fn(a)
    except StorageError as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tests for datasets/cortext-p300/cortext_store.py: local inventory, upload/download/delete against a FAKE Hugging Face,
real deletion of every local copy, and the loader. Needs numpy only. Run: python tests/test_store.py

The Hugging Face calls are exercised against an in-memory stand-in (the sandbox has no access to huggingface.co), so
this proves our logic and our use of the huggingface_hub API surface, not the live service.
"""
import json
import os
import sys
import tempfile
import types
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "datasets" / "cortext-p300"))
TMP = tempfile.mkdtemp(prefix="cortext-store-")
os.environ["CORTEXT_DATA_DIR"] = str(Path(TMP) / "data")
os.environ["CORTEXT_HF_REPO"] = "acme/cortext-p300"
os.environ["HF_TOKEN"] = "hf_fake"
import cortext_store as store  # noqa: E402
import cortext_data  # noqa: E402

ROOT = Path(os.environ["CORTEXT_DATA_DIR"])
LOGS = Path(TMP) / "logs"
LOGS.mkdir()


def make_session(sid, simulated=False, raw=True, participant="P01"):
    d = ROOT / ("simulated" if simulated else "sessions") / sid
    d.mkdir(parents=True)
    meta = {"session_id": sid, "participant_id": participant, "device": "test", "created_at": "2026-10-24T13:38:00", "simulated": simulated,
            "pipeline_version": "1.0", "quality": {"duration_s": 60, "trials_valid": 3, "epochs_usable": 6, "epochs_total": 8, "bad_channels": {}, "artifact_fraction": 0}}
    (d / "meta.json").write_text(json.dumps(meta))
    (d / "trials.csv").write_text("trial_id\n0\n")
    (d / "events.csv").write_text("event\nflash_on\n")
    n = 8
    np.savez_compressed(d / "epochs.npz", X=np.random.randn(n, 2, 5).astype(np.float32), times=np.linspace(-0.2, 0.8, 5), labels=np.array(["Fz", "Pz"]),
                        srate=250, y=np.array([1, 0] * 4, np.int8), trial_id=np.arange(n), trial_index=np.arange(n), option_id=np.array(["a", "b"] * 4),
                        option_position=np.zeros(n, np.int8), category=np.array(["x"] * n), repeat_index=np.zeros(n, np.int16),
                        flash_index=np.arange(n, dtype=np.int16), mode=np.array(["flash"] * n), trial_valid=np.ones(n, bool),
                        rejected_artifact=np.zeros(n, bool), clicked_option_id=np.array(["a"] * n), onset_lsl=np.arange(n, dtype=float),
                        session_id=np.array([sid] * n))
    t = 100 + np.arange(10) / 250
    np.savez_compressed(d / "eeg_clean.npz", data=np.zeros((2, 10), np.float32), times=t, srate=250, labels=np.array(["Fz", "Pz"]))
    if raw:
        np.savez_compressed(d / "eeg_raw.npz", data=np.arange(20, dtype=np.float32).reshape(2, 10) * 1.234, times=t, srate=250,
                            labels=np.array(["Fz", "Pz"]))
    return d


def make_logs(sid):
    names = [f"eeg_{sid}_20261024-133800.csv", f"eeg_{sid}_20261024-133800.json", f"markers_{sid}_20261024-133800.csv", f"p300_adaptive_{sid}.npz"]
    for n in names:
        (LOGS / n).write_text("x")
    return names


class FakeApi:
    """The slice of huggingface_hub.HfApi that cortext_store uses, backed by a dict {path_in_repo: bytes}."""
    repo = {}
    calls = []
    fail_purge = False

    def whoami(self):
        return {"name": "tester"}

    def create_repo(self, repo_id, repo_type=None, private=None, exist_ok=False):
        self.calls.append(("create_repo", repo_id, repo_type, private, exist_ok))

    def upload_file(self, path_or_fileobj, path_in_repo, repo_id, repo_type=None, commit_message=None):
        self.repo[path_in_repo] = Path(path_or_fileobj).read_bytes()

    def upload_folder(self, repo_id, folder_path, path_in_repo, repo_type=None, commit_message=None, delete_patterns=None):
        assert repo_type == "dataset" and delete_patterns == "*"
        for k in [k for k in self.repo if k.startswith(path_in_repo + "/")]:
            del self.repo[k]
        for f in Path(folder_path).iterdir():
            self.repo[f"{path_in_repo}/{f.name}"] = f.read_bytes()
        self.calls.append(("upload_folder", path_in_repo))
        return types.SimpleNamespace(oid="deadbeef")

    def delete_file(self, path_in_repo, repo_id, repo_type=None, commit_message=None):
        del self.repo[path_in_repo]
        self.calls.append(("delete_file", path_in_repo))

    def list_repo_files(self, repo_id, repo_type=None):
        return sorted(self.repo)

    def delete_folder(self, path_in_repo, repo_id, repo_type=None, commit_message=None):
        gone = [k for k in self.repo if k.startswith(path_in_repo + "/")]
        if not gone:
            raise type("EntryNotFoundError", (Exception,), {})("404 not found")
        for k in gone:
            del self.repo[k]
        self.calls.append(("delete_folder", path_in_repo))

    def list_lfs_files(self, repo_id, repo_type=None):
        return [types.SimpleNamespace(filename=k, file_oid=k) for k in self.__class__.lfs]

    def permanently_delete_lfs_files(self, repo_id, files, rewrite_history=True, repo_type=None):
        if self.fail_purge:
            raise RuntimeError("purge not allowed")
        self.calls.append(("purge", sorted(f.filename for f in files), rewrite_history))
        self.__class__.lfs = [k for k in self.__class__.lfs if k not in {f.filename for f in files}]
    lfs = []


def fake_snapshot_download(repo_id, repo_type, local_dir, allow_patterns, token=None):
    import fnmatch
    for path, data in FakeApi.repo.items():
        if any(fnmatch.fnmatch(path, p) for p in allow_patterns):
            dest = Path(local_dir) / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)


store._make_api = FakeApi
store._hf = lambda: types.SimpleNamespace(snapshot_download=fake_snapshot_download, get_token=lambda: None)


def main():
    # ---- local inventory
    a, b, sim = "P01_20261024133800_abc", "P02_20261024140000_xyz", "SIM_20261024150000_sss"
    make_session(a); make_session(b, participant="P02"); make_session(sim, simulated=True)
    rows = store.scan_local()
    assert {r["session_id"] for r in rows} == {a, b, sim}
    assert [r["session_id"] for r in rows if r["simulated"]] == [sim]
    assert all(r["complete"] and not r["uploaded"] for r in rows)
    store.rebuild_index()
    assert (ROOT / "index.csv").read_text().startswith("session_id,folder,simulated")
    assert store.pending_uploads() == [a, b] or sorted(store.pending_uploads()) == [a, b], store.pending_uploads()  # simulated never pending
    print("inventory: ok (simulated sessions are never pending uploads)")

    # ---- buckets: a label in meta.json, filterable in the loader, no files move
    assert store.set_bucket(b, " held out ") == "held-out"
    assert [r["bucket"] for r in store.scan_local() if r["session_id"] == b] == ["held-out"]
    assert cortext_data.load_epochs(root=str(ROOT), bucket="held-out")[0].shape[0] == 8
    assert cortext_data.load_epochs(root=str(ROOT), bucket=["held-out", "nothing"])[0].shape[0] == 8
    assert cortext_data.load_epochs(root=str(ROOT), bucket="train")[0].size == 0
    store.set_bucket(b, "")
    print("buckets: ok")

    # ---- loader reads the new location, and says what to do when it is empty
    X, y, info = cortext_data.load_epochs(root=str(ROOT))
    assert X.shape == (16, 2, 5) and y.sum() == 8 and len(set(info["groups"])) == 16, X.shape
    assert cortext_data.load_epochs(root=str(ROOT), include_simulated=True)[0].shape[0] == 24
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert cortext_data.load_epochs(root=str(Path(TMP) / "empty"))[0].size == 0
        assert any("cortext_store.py pull" in str(x.message) for x in w)
    print("loader: ok")

    # ---- push
    try:
        store.push_session(sim)
        raise AssertionError("simulated session must not be uploaded")
    except store.StorageError:
        pass
    r = store.push_session(a)
    assert r["path_in_repo"] == f"sessions/{a}" and f"sessions/{a}/eeg_raw.npz" in FakeApi.repo
    assert store.scan_local()[0 if rows[0]["session_id"] == a else 1] is not None
    assert [x for x in store.scan_local() if x["session_id"] == a][0]["uploaded"] is True
    assert store.pending_uploads() == [b]
    res = store.push_pending()
    assert [x["ok"] for x in res] == [True] and store.pending_uploads() == []
    print("push: ok (uploaded, recorded in sync_state.json, pending list shrinks)")

    # ---- voltage tables: uploaded with each session into voltages/, one CSV row per sample
    import gzip
    assert r["voltages"]["ok"] and r["voltages"]["path_in_repo"] == f"voltages/{a}_raw.csv.gz", r["voltages"]
    assert f"voltages/{b}_raw.csv.gz" in FakeApi.repo
    rows_v = gzip.decompress(FakeApi.repo[f"voltages/{a}_raw.csv.gz"]).decode().splitlines()
    assert rows_v[0] == "time_s,lsl_time,Fz_uV,Pz_uV,event" and len(rows_v) == 11, rows_v[:2]
    assert rows_v[1].startswith("0.0000,100.000000,0.00,12.34,") and rows_v[2].startswith("0.0040,100.004000,1.23,"), rows_v[1:3]
    (ROOT / "sessions" / a / "events.csv").write_text("lsl_time,event,option_id,is_target\n100.0121,flash_on,dog,1\n100.0119,trial_start,,\nbad,x,,\n")
    v = store.export_voltages(a)
    lines = gzip.open(v["path"]).read().decode().splitlines()
    assert lines[4].endswith(",flash_on option=dog target=1 | trial_start") and v["events"] == 2, lines[4]   # nearest sample (100.012)
    assert store.export_voltages(a, signal="clean")["channels"] == ["Fz", "Pz"]
    try:
        store.export_voltages(sim)
        store.push_voltages(sim)
        raise AssertionError("simulated voltage tables must not be uploaded")
    except store.StorageError:
        pass
    os.remove(ROOT / "sessions" / b / "eeg_raw.npz")
    rb = store.push_session(b)
    assert rb["voltages"]["ok"] is False and "pull --all" in rb["voltages"]["error"], rb   # reported, session still uploaded
    make_session_raw = np.savez_compressed  # restore b's raw file for later steps
    make_session_raw(ROOT / "sessions" / b / "eeg_raw.npz", data=np.zeros((2, 10), np.float32), times=100 + np.arange(10) / 250,
                     srate=250, labels=np.array(["Fz", "Pz"]))
    print("voltages: ok (CSV in voltages/, time + one column per electrode + markers, missing raw reported not raised)")

    # a rebuilt session replaces the old files remotely instead of leaving stale ones
    (ROOT / "sessions" / a / "eeg_clean.npz").unlink()
    store.push_session(a)
    assert f"sessions/{a}/eeg_clean.npz" not in FakeApi.repo
    print("re-push: ok (stale remote files cleared)")

    # ---- status
    st = store.hf_status(check=True)
    assert st["reachable"] and st["user"] == "tester" and sorted(st["remote_sessions"]) == sorted([a, b]), st
    assert store.hf_status()["checked"] is False
    print("status: ok")

    # ---- pull into a second machine (fresh data dir): epochs only, then the rest
    other = Path(TMP) / "other"
    out = store.pull(root=other)
    assert sorted(out["sessions"]) == sorted([a, b])
    assert (other / "sessions" / a / "epochs.npz").exists() and not (other / "sessions" / a / "eeg_raw.npz").exists()
    got = {r["session_id"]: r for r in store.scan_local(other)}
    assert got[a]["uploaded"] and not got[a]["complete"]
    assert store.pending_uploads(other) == []     # downloaded sessions are not "pending upload"
    store.pull([a], profile="all", root=other)
    assert (other / "sessions" / a / "eeg_raw.npz").exists()
    store.pull([a], root=other, voltages=True)
    assert (other / "voltages" / f"{a}_raw.csv.gz").exists() and not (other / "voltages" / f"{b}_raw.csv.gz").exists()
    print("pull: ok (epochs-only by default, raw on request, nothing re-uploaded)")

    # ---- delete: local + logs
    names = make_logs(a)
    make_logs(b)
    rep = store.delete_session(a, log_dir=str(LOGS))
    assert rep["ok"] and sorted(rep["log_files"]) == sorted(names), rep
    assert not (ROOT / "sessions" / a).exists() and not any(a in f for f in os.listdir(LOGS))
    assert any(b in f for f in os.listdir(LOGS)), "another session's logs must be untouched"
    assert (ROOT / "sessions" / b).exists()
    assert a not in store.load_sync_state(ROOT) and a not in (ROOT / "index.csv").read_text()
    assert not any(ROOT.joinpath("voltages").glob(f"{a}_*")), "local voltage tables go with the session"
    assert f"sessions/{a}/epochs.npz" in FakeApi.repo, "local-only delete must not touch the server"
    print("delete (local): ok (dataset folder, raw logs, adaptive model, sync entry, index row; neighbours untouched)")

    # ---- delete: remote, with purge of stored files
    FakeApi.lfs = [f"sessions/{a}/eeg_raw.npz", f"sessions/{a}/epochs.npz", f"sessions/{b}/eeg_raw.npz", f"voltages/{b}_raw.csv.gz"]
    assert f"voltages/{b}_raw.csv.gz" in FakeApi.repo
    rep = store.delete_session(b, log_dir=str(LOGS), remote=True)
    assert rep["ok"] and rep["remote"]["deleted"] and rep["remote"]["purged_files"] == 2, rep
    assert not any(k.startswith(f"voltages/{b}_") for k in FakeApi.repo) and f"voltages/{a}_raw.csv.gz" in FakeApi.repo
    assert not any(k.startswith(f"sessions/{b}/") for k in FakeApi.repo)
    assert any(k.startswith(f"sessions/{a}/") for k in FakeApi.repo)
    assert FakeApi.lfs == [f"sessions/{a}/eeg_raw.npz", f"sessions/{a}/epochs.npz"], "only this session's stored files are purged"
    assert ("purge", sorted([f"sessions/{b}/eeg_raw.npz", f"voltages/{b}_raw.csv.gz"]), True) in FakeApi.calls
    print("delete (remote): ok (folder removed, only this session's stored files purged from history)")

    # remote delete of something that is not there is fine; a failing purge is reported honestly, not hidden
    assert store.delete_remote("nope")["ok"]
    FakeApi.fail_purge = True
    FakeApi.lfs = [f"sessions/{a}/epochs.npz"]
    out = store.delete_remote(a)
    assert out["ok"] and out["purged_files"] == 0 and "may still be in its history" in out["note"], out
    FakeApi.fail_purge = False
    print("delete (remote edge cases): ok")

    # ---- configuration errors are readable
    os.environ.pop("CORTEXT_HF_REPO")
    try:
        store.push_session("anything")
    except store.StorageError as exc:
        assert "dataset" in str(exc).lower() or "not in" in str(exc)
    os.environ["CORTEXT_HF_REPO"] = ""
    real_cfg, store.CONFIG_FILE = store.CONFIG_FILE, Path(TMP) / "no-dataset.json"   # the repo's dataset.json now names a repo
    assert store.hf_status()["configured"] is False
    try:
        store._api_and_repo()
        raise AssertionError("should need a repo")
    except store.StorageError as exc:
        assert "dataset.json" in str(exc)
    store.CONFIG_FILE = real_cfg
    print("config errors: ok")

    # ---- migration from the old in-repo folder
    legacy = Path(TMP) / "legacy"
    (legacy / "sessions" / "OLD_1").mkdir(parents=True)
    (legacy / "sessions" / "OLD_1" / "meta.json").write_text(json.dumps({"session_id": "OLD_1"}))
    (legacy / "index.csv").write_text("x")
    store.LEGACY_ROOT = legacy
    out = store.migrate_legacy(ROOT)
    assert out["moved"] == ["sessions/OLD_1"] and (ROOT / "sessions" / "OLD_1" / "meta.json").exists() and not (legacy / "index.csv").exists()
    print("migrate: ok")
    print("store OK")


if __name__ == "__main__":
    main()

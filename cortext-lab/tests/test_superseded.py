#!/usr/bin/env python3
"""Sessions redone in the app ("superseded") are skipped by the loaders unless asked for. Run: python tests/test_superseded.py"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

TMP = tempfile.mkdtemp(prefix="cortext-sup-")
os.environ["CORTEXT_DATA_DIR"] = TMP
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "datasets" / "cortext-p300"))

import cortext_data  # noqa: E402


def make(sid, superseded_by=None):
    d = Path(TMP) / "sessions" / sid
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"session_id": sid, "created_at": "2026-10-25T09:00:00", "superseded_by": superseded_by, "quality": {}}))
    n = 4
    np.savez_compressed(d / "epochs.npz", X=np.zeros((n, 2, 5), np.float32), times=np.linspace(-0.2, 0.8, 5), labels=np.array(["Fz", "Pz"]),
                        y=np.array([1, 0] * 2, np.int8), trial_id=np.arange(n), trial_index=np.arange(n), option_id=np.array(["a", "b"] * 2),
                        option_position=np.zeros(n, np.int8), category=np.array(["x"] * n), repeat_index=np.zeros(n, np.int16),
                        flash_index=np.arange(n, dtype=np.int16), mode=np.array(["flash"] * n), trial_valid=np.ones(n, bool),
                        rejected_artifact=np.zeros(n, bool), clicked_option_id=np.array(["a"] * n), onset_lsl=np.arange(n, dtype=float),
                        session_id=np.array([sid] * n))


def main():
    make("A_old", superseded_by="A_redo")
    make("A_redo")
    ids = lambda rows: sorted(r["session_id"] for r in rows)
    assert ids(cortext_data.list_sessions(TMP)) == ["A_redo"]
    assert ids(cortext_data.list_sessions(TMP, include_superseded=True)) == ["A_old", "A_redo"]
    _, y, info = cortext_data.load_epochs(root=TMP)
    assert set(info["session_id"]) == {"A_redo"} and len(y) == 4
    _, y, info = cortext_data.load_epochs(sessions=["A_old"], root=TMP)
    assert set(info["session_id"]) == {"A_old"}, "naming a superseded session loads it"
    print("superseded OK")


if __name__ == "__main__":
    main()

import base64
import json
import os
import socket
import struct
import threading
import urllib.request

import pytest

from cortext_fusion import ws as wsproto
from cortext_fusion.cli import main
from cortext_fusion.config import load_config
from cortext_fusion.server import Predictor, meta_from_request, serve, serve_ws

# what the CorText Lab app sends (numeric trial ids, item ids, positions)
APP_TRIAL = {"type": "trial_start", "session_id": "P01_x", "trial_id": 7,
             "options": [{"id": "car_01", "category": "car", "label": "car", "position": 2},
                         {"id": "cat_02", "category": "cat", "label": "cat", "position": 0},
                         {"id": "dog_01", "category": "dog", "label": "dog", "position": 1},
                         {"id": "tree_03", "category": "tree", "label": "tree", "position": 3}]}
ORDER = ["cat_02", "dog_01", "car_01", "tree_03"]


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture
def servers(mock_dir):
    pred = Predictor.from_config(load_config(mock_dir / "mock_config.yaml"))
    pred.final_wait_s = 0.5
    http = serve(pred, port=0)
    wss = serve_ws(pred, port=0)
    for s in (http, wss):
        threading.Thread(target=s.serve_forever, daemon=True).start()
    yield pred, f"http://127.0.0.1:{http.server_address[1]}", wss.server_address[1]
    http.shutdown()
    wss.shutdown()


class WSClient:
    """Tiny browser-like WebSocket client (masked frames), enough to emulate the app."""

    def __init__(self, port):
        self.s = socket.create_connection(("127.0.0.1", port), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode()
        self.s.sendall((f"GET / HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                        f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n").encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            resp += self.s.recv(1024)
        head, self.buf = resp.split(b"\r\n\r\n", 1)
        assert b"101" in head and wsproto.accept_key(key).encode() in head

    def send(self, obj):
        self.s.sendall(wsproto.build_frame(0x1, json.dumps(obj).encode(), mask=os.urandom(4)))

    def _read(self, n):
        while len(self.buf) < n:
            self.buf += self.s.recv(65536)
        out, self.buf = self.buf[:n], self.buf[n:]
        return out

    def recv(self):
        b1, b2 = self._read(2)
        n = b2 & 0x7F
        if n == 126:
            (n,) = struct.unpack(">H", self._read(2))
        elif n == 127:
            (n,) = struct.unpack(">Q", self._read(8))
        return json.loads(self._read(n))

    def recv_until(self, typ):
        while True:
            m = self.recv()
            if m.get("type") == typ:
                return m


def test_meta_from_app_message_orders_by_position():
    m = meta_from_request(APP_TRIAL)
    assert m.option_ids == ORDER and m.trial_id == "P01_x:7" and m.local_trial_id == "7"


def test_app_websocket_flow_with_model_over_http(servers):
    """App -> ws trial_start/flash/trial_end; P300 process -> HTTP per-flash scores; app gets live + prediction."""
    pred, http, wsport = servers
    app = WSClient(wsport)
    app.send(APP_TRIAL)
    first = app.recv_until("live")
    assert first["trial_id"] == 7  # same JSON type the app sent (it compares with ===)
    fi = 0
    for rep in range(4):
        for oid in ORDER:
            app.send({"type": "flash", "trial_id": 7, "option_id": oid, "flash_index": fi, "repeat_index": rep,
                      "t_ms": 0, "wall_ms": 0})
            # the model knows only the app's trial id and flash index (from the LSL markers)
            code, r = post(http + "/sequential/flash", {"trial_id": 7, "flash_index": fi,
                                                         "value": 0.85 if oid == "dog_01" else 0.15})
            assert code == 200, r
            fi += 1
    live = app.recv_until("live")
    assert set(live["p300"]) == set(ORDER)
    app.send({"type": "trial_end", "trial_id": 7})
    p = app.recv_until("prediction")
    assert p["trial_id"] == 7 and p["best_option_id"] == "dog_01" and p["confidence"] > 0.5
    assert p["mock"] is False and "dog" in p["message"]


def test_app_prediction_from_per_trial_predict(servers):
    pred, http, wsport = servers
    app = WSClient(wsport)
    app.send(APP_TRIAL)
    app.recv_until("live")
    app.send({"type": "trial_end", "trial_id": 7})
    # model posts its whole-trial output after the trial ended; options come from the live trial
    outs = {"p300": {"option_id": ORDER * 2, "score": [0, 0, 3, 0] * 2},
            "category": {"p_cat": 0.1, "p_dog": 0.1, "p_car": 0.7, "p_tree": 0.1}}
    code, r = post(http + "/predict", {"trial_id": 7, "session_id": "P01_x", "outputs": outs})
    assert code == 200 and r["predicted_option"] == "car_01", r
    p = app.recv_until("prediction")
    assert p["best_option_id"] == "car_01" and p["category_scores"]


def test_no_scores_means_no_answer(servers):
    pred, http, wsport = servers
    app = WSClient(wsport)
    app.send({**APP_TRIAL, "trial_id": 8})
    app.recv_until("live")
    app.send({"type": "trial_end", "trial_id": 8})
    p = app.recv_until("prediction")
    assert p["best_option_id"] is None and p["unsure"] is True


def test_http_errors_are_clear(servers):
    _, http, _ = servers
    code, r = post(http + "/sequential/flash", {"trial_id": 999, "option_id": "x", "value": 0.5})
    assert code == 400 and "no live trial" in r["error"]
    code, r = post(http + "/predict", {"outputs": {}})
    assert code == 400 and "trial_id" in r["error"]
    with urllib.request.urlopen(http + "/health") as resp:
        h = json.loads(resp.read())
    assert h["mock"] is False and set(h["models"]) == {"p300", "category"}


def test_http_only_trial_and_bad_outputs(servers):
    _, http, _ = servers
    trial = {"trial_id": "t1", "options": [{"option_id": o, "category": o.split("_")[0]} for o in ORDER]}
    code, r = post(http + "/predict", {**trial, "outputs": {"p300": [1, 2], "mystery": [1]}})
    assert code == 200 and r["unsure"] and len(r["warnings"]) >= 2
    post(http + "/sequential/start", trial)
    for rep in range(8):
        for o in ORDER:
            code, r = post(http + "/sequential/flash", {"trial_id": "t1", "option_id": o,
                                                       "value": 0.9 if o == "tree_03" else 0.1, "repeat_index": rep})
    assert code == 200 and r["predicted_option"] == "tree_03" and r["stopped_early"]
    code, r = post(http + "/sequential/end", {"trial_id": "t1"})
    assert r["best_option_id"] == "tree_03"


def test_mock_mode_over_websocket():
    pred = Predictor.mock_only(seed=0)
    wss = serve_ws(pred, port=0)
    threading.Thread(target=wss.serve_forever, daemon=True).start()
    app = WSClient(wss.server_address[1])
    app.send(APP_TRIAL)
    for fi, oid in enumerate(ORDER * 3):
        app.send({"type": "flash", "trial_id": 7, "option_id": oid, "flash_index": fi, "repeat_index": fi // 4})
    app.send({"type": "trial_end", "trial_id": 7})
    p = app.recv_until("prediction")
    assert p["mock"] is True and p["best_option_id"] in ORDER
    wss.shutdown()


def test_mock_predictor_with_hint():
    p = Predictor.mock_only(mock_accuracy=1.0, seed=0)
    r = p.predict(meta_from_request({"trial_id": "x", "options": [{"option_id": o} for o in ORDER]}),
                  request={"hint_option": "car_01"})
    assert r["mock"] is True and r["predicted_option"] == "car_01"


def test_cli_end_to_end(mock_dir, tmp_path, capsys):
    assert main(["describe", str(mock_dir / "outputs" / "p300_per_flash.csv"),
                 "--markers", str(mock_dir / "lab_export")]) == 0
    out = capsys.readouterr().out
    assert "per_flash" in out and "matching the stimulus log: 120/120" in out
    cal = tmp_path / "cal.yaml"
    sess = sorted(p.name for p in (mock_dir / "lab_export").iterdir())
    assert main(["calibrate", str(mock_dir / "mock_config.yaml"), "--out", str(cal),
                 "--exclude-sessions", sess[-1]]) == 0
    assert cal.exists()
    assert main(["fuse", str(mock_dir / "mock_config.yaml"), "--out", str(tmp_path / "p.csv")]) == 0
    assert (tmp_path / "p.csv").read_text().count("\n") > 50
    assert main(["evaluate", str(mock_dir / "mock_config.yaml"), "--out", str(tmp_path / "rep"), "--no-plots"]) == 0
    assert (tmp_path / "rep" / "report.md").exists()
    assert main(["mock", str(tmp_path / "m"), "--trials", "5", "--participants", "1", "--sessions", "1"]) == 0


def test_cli_watch_once(mock_dir, tmp_path):
    out = tmp_path / "live.jsonl"
    assert main(["watch", str(mock_dir / "config_p300_per_trial_folder.yaml"),
                 "--folder", f"p300={mock_dir / 'outputs' / 'p300_per_trial'}", "--once", "--out", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert len(lines) > 50 and "predicted_option" in json.loads(lines[0])

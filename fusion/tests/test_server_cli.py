import json
import threading
import urllib.request

import pytest

from cortext_fusion.cli import main
from cortext_fusion.config import load_config
from cortext_fusion.server import Predictor, serve

TRIAL = {"trial_id": "live1", "options": [{"option_id": "A", "category": "cat"}, {"option_id": "B", "category": "dog"},
                                          {"option_id": "C", "category": "car"}, {"option_id": "D", "category": "tree"}],
         "target_category": "dog"}


@pytest.fixture
def server_url(mock_dir):
    pred = Predictor.from_config(load_config(mock_dir / "mock_config.yaml"))
    httpd = serve(pred, port=0)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_health_and_predict(server_url):
    with urllib.request.urlopen(server_url + "/health") as r:
        h = json.loads(r.read())
    assert h["mock"] is False and set(h["models"]) == {"p300", "category"}
    out = {"p300": {"option_id": list("ABCD") * 3, "score": [0, 3, 0, 0] * 3}}
    code, r = post(server_url + "/predict", {**TRIAL, "outputs": out})
    assert code == 200 and r["predicted_option"] == "B" and r["message"].startswith("EEG indicates B (dog)")


def test_predict_bad_output_is_a_warning_not_a_crash(server_url):
    code, r = post(server_url + "/predict", {**TRIAL, "outputs": {"p300": [1, 2], "mystery": [1]}})
    assert code == 200 and r["unsure"] and len(r["warnings"]) >= 2
    code, r = post(server_url + "/predict", {"outputs": {}})
    assert code == 400 and "trial_id" in r["error"]


def test_sequential_endpoints(server_url):
    post(server_url + "/sequential/start", TRIAL)
    for rep in range(8):
        for o in "ABCD":
            code, r = post(server_url + "/sequential/flash",
                           {"trial_id": "live1", "option_id": o, "value": 0.9 if o == "D" else 0.1, "repeat_index": rep})
    assert code == 200 and r["predicted_option"] == "D" and r["stopped_early"]
    assert post(server_url + "/sequential/end", {"trial_id": "live1"})[1] == {"ok": True}


def test_mock_predictor_mode():
    p = Predictor.mock_only(mock_accuracy=1.0, seed=0)
    from cortext_fusion.server import meta_from_request
    r = p.predict(meta_from_request(TRIAL), request={"hint_option": "C"})
    assert r["mock"] is True and r["predicted_option"] == "C"
    p.seq_start(meta_from_request(TRIAL))
    for _ in range(10):
        for o in "ABCD":
            r = p.seq_flash({"trial_id": "live1", "option_id": o, "hint_option": "A"})
    assert r["mock"] is True and r["predicted_option"] == "A"


def test_cli_end_to_end(mock_dir, tmp_path, capsys):
    assert main(["describe", str(mock_dir / "outputs" / "p300_per_flash.csv"), "--markers", str(mock_dir / "markers.csv")]) == 0
    assert "per_flash" in capsys.readouterr().out
    cal = tmp_path / "cal.yaml"
    assert main(["calibrate", str(mock_dir / "mock_config.yaml"), "--out", str(cal),
                 "--exclude-sessions", "P02_S2"]) == 0
    assert cal.exists()
    assert main(["fuse", str(mock_dir / "mock_config.yaml"), "--out", str(tmp_path / "p.csv")]) == 0
    assert (tmp_path / "p.csv").read_text().count("\n") > 50
    assert main(["evaluate", str(mock_dir / "mock_config.yaml"), "--out", str(tmp_path / "rep"), "--no-plots"]) == 0
    assert (tmp_path / "rep" / "report.md").exists()
    assert main(["mock", str(tmp_path / "m"), "--trials", "5", "--participants", "1", "--sessions", "1"]) == 0


def test_cli_watch_once(mock_dir, tmp_path, capsys):
    out = tmp_path / "live.jsonl"
    assert main(["watch", str(mock_dir / "config_p300_per_trial_folder.yaml"),
                 "--folder", f"p300={mock_dir / 'outputs' / 'p300_per_trial'}", "--once", "--out", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert len(lines) > 50 and "predicted_option" in json.loads(lines[0])

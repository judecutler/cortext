#!/usr/bin/env python3
"""Bridge voice endpoints (stdlib only, no network): /api/voice/status, /api/tts, /api/narrate against a FAKE
OpenAI-compatible LLM server. Run: python tests/test_voice.py"""
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PORT = 9100 + (os.getpid() % 200)
LLM_PORT = PORT + 300
seen = []


class FakeLLM(BaseHTTPRequestHandler):
    reply = "Hmm... I'm about 78% sure Jude saw the dog, option B!"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode())
        seen.append({"path": self.path, "auth": self.headers.get("Authorization"), "body": body})
        out = json.dumps({"choices": [{"message": {"content": FakeLLM.reply}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=10) as r:
        return r.status, r.headers, r.read()


def post(path, obj, origin=None):
    h = {"Content-Type": "application/json"}
    if origin:
        h["Origin"] = origin
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(obj).encode(), headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


FACTS = {"name": "Jude", "question": "Which one is the dog?", "guess_letter": "B", "guess_label": "dog", "confidence_percent": 78,
         "simulated": False, "true_answer_id": "dog"}


def main():
    llm = ThreadingHTTPServer(("127.0.0.1", LLM_PORT), FakeLLM)
    threading.Thread(target=llm.serve_forever, daemon=True).start()
    env = {**os.environ, "OPENAI_BASE_URL": f"http://127.0.0.1:{LLM_PORT}/v1", "OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "fake-1",
           "ANTHROPIC_API_KEY": ""}
    proc = subprocess.Popen([sys.executable, os.path.join(ROOT, "bridge", "bridge.py"), "--port", str(PORT), "--no-lsl", "--no-csv"],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            try:
                get("/api/status")
                break
            except Exception:
                time.sleep(0.1)
        code, _, body = get("/api/voice/status")
        st = json.loads(body)
        assert code == 200 and st["llm"]["openai"]["configured"] and st["llm"]["openai"]["model"] == "fake-1", st
        assert "simulation" in st["rules"] and st["default_persona"], st

        # narration with a custom context: the context replaces the persona, the fixed rules stay
        code, r = post("/api/narrate", {"provider": "openai", "facts": FACTS, "context": "You are a grumpy pirate."})
        assert code == 200 and r["ok"] and "78%" in r["text"] and r["custom_context"], r
        sys_msg = seen[-1]["body"]["messages"][0]["content"]
        assert sys_msg.startswith("You are a grumpy pirate.") and "simulation" in sys_msg, sys_msg
        assert seen[-1]["auth"] == "Bearer test-key" and seen[-1]["path"] == "/v1/chat/completions"
        user = seen[-1]["body"]["messages"][1]["content"]
        assert "Jude" in user and "true_answer_id" not in user, user      # unknown keys (like the truth) are dropped

        # default persona when no context
        post("/api/narrate", {"provider": "openai", "facts": FACTS})
        assert seen[-1]["body"]["messages"][0]["content"].startswith("You are the narrator"), seen[-1]

        # a reply without the real percent is refused (the browser then uses its template)
        FakeLLM.reply = "Hmm, definitely the dog!"
        code, r = post("/api/narrate", {"provider": "openai", "facts": FACTS})
        assert not r["ok"] and "confidence" in r["error"], r

        # unconfigured provider / cross-site
        code, r = post("/api/narrate", {"provider": "anthropic", "facts": FACTS})
        assert not r["ok"] and "ANTHROPIC_API_KEY" in r["error"], r
        code, r = post("/api/narrate", {"provider": "openai", "facts": FACTS}, origin="http://evil.example")
        assert code == 403, (code, r)

        # TTS: either an engine works, or a clear 503 telling you what to install
        try:
            code, hdr, data = get("/api/tts?text=Which%20one%20is%20the%20dog%3F")
            assert code == 200 and len(data) > 500 and hdr["Content-Type"].startswith("audio/"), (code, hdr)
            for i in range(8):   # regression: pyttsx3 sometimes returned before its file was filled
                code, hdr, data = get(f"/api/tts?text=Narration%20line%20number%20{i}%20with%20{30 + i}%25%20confidence")
                assert code == 200 and len(data) > 500, (i, code)
            tts = f"TTS OK via {hdr['X-TTS-Engine']} (9 lines, last {len(data)} bytes)"
        except urllib.error.HTTPError as e:
            msg = json.loads(e.read().decode())["error"]
            assert e.code == 503 and ("pip install" in msg or st["tts_engines"]), msg
            tts = f"TTS unavailable here (expected without engines/network): {msg[:90]}"
        print("voice OK: status, narrate (custom context, default persona, percent guard, missing key, CSRF);", tts)
    finally:
        proc.terminate()
        proc.wait(5)
        llm.shutdown()


if __name__ == "__main__":
    main()

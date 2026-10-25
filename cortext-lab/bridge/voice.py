"""Voice for the demo: text-to-speech files and LLM-written narration, served by the bridge.

    GET  /api/voice/status[?check=1]      which TTS engines / LLM providers are usable
    GET  /api/tts?text=...&voice=...&rate=...   -> audio (mp3 from edge-tts, or wav from pyttsx3), cached on disk
    POST /api/narrate  {provider, facts}  -> {"ok": true, "text": "..."}  (the browser falls back to its own template)

Everything here is optional; the web app works without it (the browser's built-in speech is the fallback).

TTS engines, best first (free, no account):
  * edge-tts  (pip install edge-tts)  Microsoft's neural "Read aloud" voices. Needs internet. Best quality.
  * pyttsx3   (pip install pyttsx3)   the operating system's offline voices (SAPI on Windows, NSSpeech on macOS,
                                      espeak on Linux). Run in a subprocess because it is not thread-safe.

LLM providers for narration (keys stay in the bridge, never in the browser):
  * ollama     free + local.  Install https://ollama.com, then `ollama pull llama3.2`. No key.
  * openai     any OpenAI-compatible endpoint: OpenAI, Groq (free tier), OpenRouter, LM Studio ...
               key: OPENAI_API_KEY, base URL: OPENAI_BASE_URL, model: OPENAI_MODEL
  * anthropic  Claude. key: ANTHROPIC_API_KEY, model: ANTHROPIC_MODEL
Settings can also go in bridge/voice.local.json (git-ignored), e.g.
  {"anthropic_api_key": "sk-ant-...", "anthropic_model": "claude-haiku-5-5",
   "openai_base_url": "https://api.groq.com/openai/v1", "openai_api_key": "gsk_...", "openai_model": "llama-3.1-8b-instant",
   "ollama_url": "http://localhost:11434", "ollama_model": "llama3.2"}

Standard library only.
"""
import asyncio
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "voice.local.json")
CACHE_DIR = os.path.join(HERE, "logs", "tts_cache")
MAX_TEXT = 400
DEFAULT_EDGE_VOICE = "en-US-AndrewMultilingualNeural"

_tts_lock = threading.Lock()
_edge_down_until = 0.0   # after a network failure, skip edge-tts for a minute so offline demos are not slowed down


class VoiceError(Exception):
    pass


def load_config():
    cfg = {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            cfg = json.load(fh) or {}
    except FileNotFoundError:
        pass
    except ValueError as exc:
        cfg = {"_error": f"voice.local.json is not valid JSON: {exc}"}
    env = os.environ.get
    return {
        "anthropic_api_key": env("ANTHROPIC_API_KEY") or cfg.get("anthropic_api_key"),
        "anthropic_model": env("ANTHROPIC_MODEL") or cfg.get("anthropic_model") or "claude-haiku-5-5",
        "openai_api_key": env("OPENAI_API_KEY") or cfg.get("openai_api_key"),
        "openai_base_url": (env("OPENAI_BASE_URL") or cfg.get("openai_base_url") or "https://api.openai.com/v1").rstrip("/"),
        "openai_model": env("OPENAI_MODEL") or cfg.get("openai_model") or "gpt-4o-mini",
        "ollama_url": (env("OLLAMA_URL") or cfg.get("ollama_url") or "http://localhost:11434").rstrip("/"),
        "ollama_model": env("OLLAMA_MODEL") or cfg.get("ollama_model") or "llama3.2",
        "edge_voice": cfg.get("edge_voice") or DEFAULT_EDGE_VOICE,
        "error": cfg.get("_error"),
    }


def _has(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _ollama_up(url, timeout=0.8):
    try:
        with urllib.request.urlopen(url + "/api/tags", timeout=timeout) as r:
            tags = json.loads(r.read().decode() or "{}")
            return True, [m.get("name") for m in tags.get("models", [])]
    except Exception:
        return False, []


def status(check=False):
    cfg = load_config()
    tts = []
    if _has("edge_tts"):
        tts.append("edge")
    if _has("pyttsx3"):
        tts.append("pyttsx3")
    llm = {
        "anthropic": {"configured": bool(cfg["anthropic_api_key"]), "model": cfg["anthropic_model"]},
        "openai": {"configured": bool(cfg["openai_api_key"]), "model": cfg["openai_model"], "base_url": cfg["openai_base_url"]},
        "ollama": {"configured": None, "model": cfg["ollama_model"], "url": cfg["ollama_url"]},
    }
    if check:
        up, models = _ollama_up(cfg["ollama_url"])
        llm["ollama"]["configured"] = up
        llm["ollama"]["models"] = models
        if up and models and not any(m.split(":")[0] == cfg["ollama_model"].split(":")[0] for m in models):
            llm["ollama"]["warning"] = f"model {cfg['ollama_model']} is not pulled: run  ollama pull {cfg['ollama_model']}"
    return {"tts_engines": tts, "default_tts": tts[0] if tts else None, "edge_voice": cfg["edge_voice"], "llm": llm, "config_error": cfg["error"]}


# ---------------------------------------------------------------- TTS
def _clean_text(text):
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        raise VoiceError("empty text")
    return text[:MAX_TEXT]


def _rate_pct(rate):
    """rate as a multiplier (1.0 = normal) -> edge-tts '+10%' string."""
    try:
        r = float(rate)
    except (TypeError, ValueError):
        r = 1.0
    r = min(2.0, max(0.5, r))
    return f"{round((r - 1) * 100):+d}%"


def synthesize(text, voice=None, rate=1.0, engine=None):
    """Return (bytes, content_type, engine_used). Cached by (engine, voice, rate, text)."""
    text = _clean_text(text)
    engines = status()["tts_engines"]
    if not engines:
        raise VoiceError("no TTS engine installed on the bridge: pip install edge-tts  (or pyttsx3 for offline voices)")
    order = [engine] if engine in engines else []
    order += [e for e in engines if e not in order]
    global _edge_down_until
    if len(order) > 1 and order[0] == "edge" and time.time() < _edge_down_until:
        order = order[1:] + ["edge"]   # edge-tts failed recently (offline?): try the local voice first
    errors = []
    for eng in order:
        neural = bool(voice and "Neural" in voice)
        v = ("" if neural else (voice or "")) if eng == "pyttsx3" else (voice if neural else load_config()["edge_voice"])
        ext = "mp3" if eng == "edge" else "wav"
        key = hashlib.sha1(json.dumps([eng, v, _rate_pct(rate), text]).encode()).hexdigest()[:20]
        path = os.path.join(CACHE_DIR, f"{key}.{ext}")
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            os.makedirs(CACHE_DIR, exist_ok=True)
            tmp = path + ".part"
            try:
                if eng == "edge":
                    _edge_tts(text, v, rate, tmp)
                else:
                    _pyttsx3(text, v, rate, tmp)
                os.replace(tmp, path)
            except Exception as exc:  # network down, voice missing ...
                errors.append(f"{eng}: {exc}")
                if eng == "edge":
                    _edge_down_until = time.time() + 60
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                continue
        with open(path, "rb") as fh:
            data = fh.read()
        return data, ("audio/mpeg" if ext == "mp3" else "audio/wav"), eng
    raise VoiceError("; ".join(errors) or "TTS failed")


def _edge_tts(text, voice, rate, out_path):
    import edge_tts  # type: ignore

    async def go():
        await edge_tts.Communicate(text, voice, rate=_rate_pct(rate)).save(out_path)

    asyncio.run(asyncio.wait_for(go(), timeout=20))
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise VoiceError("edge-tts returned no audio (no internet?)")


_PYTTSX3_SCRIPT = r"""
import sys, json, pyttsx3
a = json.loads(sys.argv[1])
e = pyttsx3.init()
if a["voice"]:
    for v in e.getProperty("voices"):
        if a["voice"].lower() in (v.name or "").lower() or a["voice"] == v.id:
            e.setProperty("voice", v.id); break
e.setProperty("rate", int(e.getProperty("rate") * a["rate"]))
e.save_to_file(a["text"], a["out"])
e.runAndWait()
"""


def _pyttsx3(text, voice, rate, out_path):
    args = json.dumps({"text": text, "voice": voice, "rate": min(2.0, max(0.5, float(rate or 1))), "out": out_path})
    with _tts_lock:
        r = subprocess.run([sys.executable, "-c", _PYTTSX3_SCRIPT, args], capture_output=True, text=True, timeout=30)
    if r.returncode != 0 or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise VoiceError((r.stderr or "pyttsx3 produced no audio").strip().splitlines()[-1][:200])


# ---------------------------------------------------------------- narration
# The editable part (mood / persona / experiment context). The web app's Setup tab can replace it per demo.
DEFAULT_PERSONA = (
    "You are the narrator of a live brain-computer-interface demo in front of an audience. An EEG model has just "
    "read a participant's brain responses while answer options were presented one by one, and you announce its guess "
    "out loud. Sound warm, curious and a little theatrical; start with a short thinking filler such as 'Hmm...' or 'Okay...'."
)
# The fixed part: always appended, whatever the persona says, so the narration stays honest and speakable.
RULES = (
    "Rules that override everything above: write ONE or TWO short spoken sentences, at most 40 words, plain text only "
    "(no emojis, markdown, quotes or stage directions). You MUST say the confidence exactly as the whole-number percent "
    "given, the option letter and the option label, and refer to the participant by the given name. Never claim "
    "certainty; if the confidence is under 40 percent sound openly unsure. If 'simulated' is true you MUST make clear "
    "it is a simulation, not a real brain reading. If the participant's click is given, react to whether it matches "
    "the model's guess. Write numbers as digits."
)
MAX_CONTEXT = 2000


def system_prompt(context=None):
    persona = re.sub(r"\s+", " ", str(context or "")).strip()[:MAX_CONTEXT] or DEFAULT_PERSONA
    return persona + "\n\n" + RULES


def _facts_text(f):
    keep = {k: f.get(k) for k in ("name", "question", "guess_letter", "guess_label", "confidence_percent",
                                  "runner_up_letter", "runner_up_label", "runner_up_percent", "simulated",
                                  "clicked_letter", "clicked_label", "click_matches_guess", "trial_number", "modality") if f.get(k) is not None}
    return "Facts for this trial (JSON): " + json.dumps(keep)


def _post_json(url, payload, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", **headers}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise VoiceError(f"HTTP {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        raise VoiceError(f"cannot reach {url}: {exc.reason}")


def _tidy(text):
    text = re.sub(r"[*_#`\"“”]", "", str(text or "")).strip()
    text = re.sub(r"\s+", " ", text)
    return text[:300]


def narrate(provider, facts, timeout=6.0, context=None):
    """Ask an LLM for one narration line. Raises VoiceError; the browser then uses its template."""
    cfg = load_config()
    user = _facts_text(facts or {})
    system = system_prompt(context)
    t0 = time.time()
    if provider == "anthropic":
        if not cfg["anthropic_api_key"]:
            raise VoiceError("no ANTHROPIC_API_KEY (env var or bridge/voice.local.json)")
        r = _post_json("https://api.anthropic.com/v1/messages",
                       {"model": cfg["anthropic_model"], "max_tokens": 120, "temperature": 0.9, "system": system,
                        "messages": [{"role": "user", "content": user}]},
                       {"x-api-key": cfg["anthropic_api_key"], "anthropic-version": "2023-06-01"}, timeout)
        text = "".join(b.get("text", "") for b in r.get("content", []) if b.get("type") == "text")
        model = cfg["anthropic_model"]
    elif provider in ("openai", "ollama"):
        if provider == "ollama":
            base, key, model = cfg["ollama_url"] + "/v1", None, cfg["ollama_model"]
        else:
            if not cfg["openai_api_key"]:
                raise VoiceError("no OPENAI_API_KEY (env var or bridge/voice.local.json)")
            base, key, model = cfg["openai_base_url"], cfg["openai_api_key"], cfg["openai_model"]
        r = _post_json(base + "/chat/completions",
                       {"model": model, "max_tokens": 120, "temperature": 0.9,
                        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
                       {"Authorization": f"Bearer {key}"} if key else {}, timeout)
        text = ((r.get("choices") or [{}])[0].get("message") or {}).get("content", "")
    else:
        raise VoiceError(f"unknown narration provider {provider!r}")
    text = _tidy(text)
    if not text:
        raise VoiceError("the model returned no text")
    pct = facts.get("confidence_percent") if facts else None
    if pct is not None and str(pct) not in text:
        raise VoiceError("the model left out the confidence; using the template")   # keep the numbers honest
    return {"ok": True, "text": text, "provider": provider, "model": model, "custom_context": bool(context), "ms": round((time.time() - t0) * 1000)}

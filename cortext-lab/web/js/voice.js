// Voice for the demo: reading the question aloud, spoken (audio) stimuli, and the narrator that announces the model's
// guess. Everything is optional and free:
//   * speech comes from the bridge (/api/tts: edge-tts neural voices, or the OS voices via pyttsx3) when installed,
//     otherwise from the browser's built-in speechSynthesis;
//   * narration text comes from an LLM through the bridge (/api/narrate: Ollama, any OpenAI-compatible API, or Claude),
//     otherwise from the templates below.
//
// Timing rule for EEG: audio STIMULI are decoded into Web Audio buffers up front, their leading silence is trimmed, and
// they start inside the same animation frame as the marker, so the onset lines up with flash_on (plus the measured
// output latency, which is written into the marker). Only if no buffer can be made does it fall back to
// speechSynthesis, and those markers are flagged audio_timing = "approximate". Question audio and narration never
// overlap the EEG window: the flash phase starts after the question has finished, and narration runs after it.

export const DEFAULT_PERSONA =
  "You are the narrator of a live brain-computer-interface demo in front of an audience. An EEG model has just " +
  "read a participant's brain responses while answer options were presented one by one, and you announce its guess " +
  "out loud. Sound warm, curious and a little theatrical; start with a short thinking filler such as 'Hmm...' or 'Okay...'.";

/** Mood presets for the narrator's context (Setup → Voice & narrator). The text is fully editable there. */
export const NARRATOR_MOODS = [
  ['default', 'Warm & theatrical (default)', DEFAULT_PERSONA],
  ['sports', 'Sports commentator',
    'You are an excitable sports commentator calling a live brain-computer-interface match. An EEG model just read the ' +
    "participant's brain while the options were presented, and you call its pick like the final seconds of a big game."],
  ['magician', 'Stage mind-reader',
    'You are a stage mind-reader with a mysterious, slightly playful manner. An EEG headset has just read the ' +
    "participant's brain responses, and you reveal what it detected as if pulling it from the air, building a little suspense."],
  ['scientist', 'Calm lab scientist',
    'You are a calm, precise research scientist reporting the output of an EEG P300 classifier during a lab session. ' +
    'Neutral tone, no theatrics or filler words; report the guess and confidence plainly, like a lab log read aloud.'],
  ['skeptic', 'Dry skeptic',
    'You are a dry, witty skeptic narrating an EEG brain-reading demo. You respect the data but enjoy pointing out ' +
    'uncertainty, and you keep expectations firmly grounded.'],
];

const clamp = (x, a, b) => Math.min(b, Math.max(a, x));
const pick = (arr, rng) => arr[Math.floor(rng() * arr.length) % arr.length];
const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);

// ------------------------------------------------------------------ narration text (pure; unit-tested)
/** What the narrator is told about one trial. No ids, no ground truth (the true answer is never included). */
export function narrationFacts({ trial, pred, name = '', when = 'after_click', modality = 'visual' }) {
  if (!pred || pred.timeout || pred.error || pred.best_option_id == null) return null;
  const byId = Object.fromEntries(trial.options.map((o) => [o.id, o]));
  const best = byId[pred.best_option_id];
  if (!best) return null;
  const probs = pred.probabilities && Object.keys(pred.probabilities).length ? pred.probabilities : null;
  const conf = probs ? probs[best.id] : pred.confidence;
  const ranked = probs ? Object.entries(probs).filter(([id]) => id !== best.id && byId[id]).sort((a, b) => b[1] - a[1]) : [];
  const ru = ranked.length ? byId[ranked[0][0]] : null;
  const f = {
    name: (name || '').trim() || null,
    question: trial.prompt,
    trial_number: trial.trial_index + 1,
    guess_letter: best.letter,
    guess_label: best.label,
    confidence_percent: conf == null || Number.isNaN(conf) ? null : Math.round(clamp(conf, 0, 1) * 100),
    runner_up_letter: ru ? ru.letter : null,
    runner_up_label: ru ? ru.label : null,
    runner_up_percent: ru ? Math.round(ranked[0][1] * 100) : null,
    simulated: pred.source === 'mock' || !!pred.simulated,
    modality,
  };
  if (when === 'after_click' && trial.clicked_answer_id) {
    const c = byId[trial.clicked_answer_id];
    f.clicked_letter = c?.letter ?? null;
    f.clicked_label = c?.label ?? null;
    f.click_matches_guess = trial.clicked_answer_id === best.id;
  }
  return f;
}

/** Offline narration: a few varied lines by confidence band. Always honest about simulated predictions. */
export function templateNarration(f, rng = Math.random, { plain = false } = {}) {
  if (f && plain) {
    let t = `${f.simulated ? 'Simulated run. ' : ''}Model output: option ${f.guess_letter}, the ${f.guess_label}, at ${f.confidence_percent ?? '?'}% confidence.`;
    if (f.click_matches_guess != null && f.clicked_letter) t += ` The participant chose ${f.clicked_letter}, which ${f.click_matches_guess ? 'agrees' : 'disagrees'} with the model.`;
    return t;
  }
  if (!f) return pick(['Hmm... I couldn’t get a clear read on that one.', 'Hmm. The signal was too noisy that time; no call from me.'], rng);
  const who = f.name;
  const subj = who || 'you';
  const poss = who ? `${who}’s` : 'your';
  const verb = f.modality === 'audio' ? 'heard' : f.modality === 'audiovisual' ? 'locked onto' : 'saw';
  const L = f.guess_letter, lab = f.guess_label, p = f.confidence_percent ?? '?';
  let line;
  if (f.simulated) {
    line = `Simulation only, not a real brain reading: the mock model is about ${p}% sure on option ${L}, the ${lab}.`;
  } else if (p !== '?' && p >= 70) {
    line = pick([
      `Hmm... based on the brain patterns, I'm about ${p}% sure ${subj} ${verb} the ${lab} at option ${L}. So that's probably the answer.`,
      `Okay... ${poss} brain really reacted to option ${L}. I'm about ${p}% sure it's the ${lab}.`,
      `Hmm... that's a strong response. About ${p}% says ${subj} ${verb} the ${lab}, option ${L}.`,
    ], rng);
  } else if (p !== '?' && p >= 40) {
    line = pick([
      `Hmm... the signal leans toward option ${L}, the ${lab}, at about ${p}%.`,
      `Okay, I'm seeing a lean: option ${L}, the ${lab}, at about ${p}%.`,
    ], rng);
    if (f.runner_up_label && f.runner_up_percent >= 20) line += ` But the ${f.runner_up_label} at ${f.runner_up_percent}% isn't out of it yet.`;
  } else {
    line = pick([
      `Hmm, this one's tricky. My best guess is option ${L}, the ${lab}, but only about ${p}%.`,
      `Honestly? ${cap(poss)} brain is keeping its secrets. Option ${L}, the ${lab}, edges ahead at just ${p}%.`,
    ], rng);
  }
  if (f.click_matches_guess === true) line += ` And ${subj} picked ${L} too. Brain and hand agree!`;
  else if (f.click_matches_guess === false && f.clicked_letter) line += ` But ${subj} went with ${f.clicked_letter}, the ${f.clicked_label}. Interesting...`;
  return line;
}

/** LLM narration through the bridge, falling back to the template. Resolves within ~timeoutMs. */
export async function composeNarration(facts, { provider = 'template', context = '', timeoutMs = 5000, rng = Math.random, plain = false } = {}) {
  const t0 = performance.now();
  const templateNarrationP = (f) => templateNarration(f, rng, { plain });
  if (facts && provider && provider !== 'template') {
    const ac = new AbortController();
    const timer = setTimeout(() => ac.abort(), timeoutMs);
    try {
      const r = await fetch('/api/narrate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, signal: ac.signal,
        body: JSON.stringify({ provider, facts, context: context || null, timeout: timeoutMs / 1000 }),
      });
      const j = await r.json();
      if (j.ok && j.text) return { text: j.text, provider, model: j.model || null, ms: Math.round(performance.now() - t0) };
      return { text: templateNarrationP(facts), provider: 'template', fallback_error: j.error || 'no text', ms: Math.round(performance.now() - t0) };
    } catch (e) {
      return { text: templateNarrationP(facts), provider: 'template', fallback_error: e.name === 'AbortError' ? `timed out after ${timeoutMs} ms` : String(e.message || e), ms: Math.round(performance.now() - t0) };
    } finally { clearTimeout(timer); }
  }
  return { text: templateNarrationP(facts), provider: 'template', ms: 0 };
}

// ------------------------------------------------------------------ audio
/** Seconds of leading silence in a decoded buffer (so the onset is the first sound, not the file start). */
export function leadingSilence(samples, sampleRate, { rel = 0.05, floor = 0.003, padMs = 5 } = {}) {
  let peak = 0;
  for (let i = 0; i < samples.length; i++) { const a = Math.abs(samples[i]); if (a > peak) peak = a; }
  const thr = Math.max(floor, peak * rel);
  for (let i = 0; i < samples.length; i++) if (Math.abs(samples[i]) > thr) return Math.max(0, i / sampleRate - padMs / 1000);
  return 0;
}

export class Voice {
  /** opts: { engine: 'auto' | 'browser', voice (bridge/edge-tts voice), browserVoice (speechSynthesis voice name), rate } */
  constructor({ engine = 'auto', voice = '', browserVoice = '', rate = 1 } = {}) {
    this.engine = engine;
    this.voiceName = voice || '';
    this.browserVoiceName = browserVoice || '';
    this.rate = clamp(Number(rate) || 1, 0.5, 2);
    this._ctx = null;
    this.clips = new Map();       // key -> Promise<clip|null>
    this.stim = new Map();        // option id -> clip|null for the current trial
    this.sources = new Set();
    this.serverTts = engine !== 'browser';   // turned off after the bridge says it has no TTS engine
    this.lastServerError = null;
  }

  get ctx() {
    if (!this._ctx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      this._ctx = AC ? new AC({ latencyHint: 'interactive' }) : null;
    }
    return this._ctx;
  }
  /** Call from a click/key handler: browsers only start audio after a user gesture. */
  unlock() { try { const c = this.ctx; if (c && c.state === 'suspended') c.resume(); } catch { /* ignore */ } }
  get outputLatencyMs() {
    const c = this._ctx;
    return c ? Math.round(((c.baseLatency || 0) + (c.outputLatency || 0)) * 1000 * 10) / 10 : null;
  }

  async _decode(ab, engine) {
    const c = this.ctx;
    if (!c) return null;
    const buffer = await c.decodeAudioData(ab);
    const offset = leadingSilence(buffer.getChannelData(0), buffer.sampleRate);
    return { buffer, offset, engine, duration_ms: Math.round((buffer.duration - offset) * 1000), trimmed_ms: Math.round(offset * 1000) };
  }

  /** Speech for `text` as a decoded buffer, from the bridge. null when there is no server TTS (caller falls back). */
  clip(text) {
    if (!this.serverTts || !text) return Promise.resolve(null);
    const key = `tts|${this.voiceName}|${this.rate}|${text}`;
    if (!this.clips.has(key)) {
      const q = new URLSearchParams({ text, rate: String(this.rate) });
      if (this.voiceName) q.set('voice', this.voiceName);
      const ac = new AbortController();
      const timer = setTimeout(() => ac.abort(), 9000);   // never leave the room waiting: fall back to the browser's voice
      const p = fetch(`/api/tts?${q}`, { signal: ac.signal }).then(async (r) => {
        if (!r.ok) {
          const j = await r.json().catch(() => ({}));
          this.lastServerError = j.error || `HTTP ${r.status}`;
          if (r.status === 503 && /no TTS engine/.test(this.lastServerError)) this.serverTts = false;
          return null;
        }
        return this._decode(await r.arrayBuffer(), r.headers.get('X-TTS-Engine') || 'server');
      }).catch((e) => { this.lastServerError = e.name === 'AbortError' ? 'bridge voice took over 9 s' : String(e.message || e); return null; })
        .finally(() => clearTimeout(timer));
      this.clips.set(key, p);
      p.then((c) => { if (!c) this.clips.delete(key); });   // retry next time
    }
    return this.clips.get(key);
  }

  /** An uploaded / configured audio file (item.audio) as a decoded buffer. */
  fileClip(url) {
    const key = `file|${url}`;
    if (!this.clips.has(key)) {
      const p = fetch(url).then((r) => (r.ok ? r.arrayBuffer() : null)).then((ab) => (ab ? this._decode(ab, 'file') : null)).catch(() => null);
      this.clips.set(key, p);
    }
    return this.clips.get(key);
  }

  /** Warm the cache (e.g. every question and option label at session start). */
  prefetch(texts) { if (this.serverTts) [...new Set(texts.filter(Boolean))].forEach((t) => this.clip(t)); }

  /** Start a decoded clip NOW. Returns { ended: Promise, latency_ms }. */
  playClip(c) {
    const ctx = this.ctx;
    const src = ctx.createBufferSource();
    src.buffer = c.buffer;
    src.connect(ctx.destination);
    const ended = new Promise((res) => { src.onended = () => { this.sources.delete(src); res(); }; });
    src.start(0, c.offset);
    this.sources.add(src);
    return { src, ended, latency_ms: this.outputLatencyMs };
  }

  _browserVoice() {
    const ss = window.speechSynthesis;
    if (!ss) return null;
    const vs = ss.getVoices();
    const want = this.browserVoiceName;
    if (want) { const v = vs.find((x) => x.name === want) || vs.find((x) => x.name.toLowerCase().includes(want.toLowerCase())); if (v) return v; }
    return vs.find((x) => /natural|neural|google us english|aria|jenny|guy|samantha/i.test(x.name) && /^en/i.test(x.lang)) || vs.find((x) => /^en/i.test(x.lang)) || null;
  }

  /** Browser speech; resolves when finished (or after a safety timeout). onStart(perfNow) fires when audio begins. */
  speakBrowser(text, { onStart } = {}) {
    const ss = window.speechSynthesis;
    if (!ss || !text) return Promise.resolve(false);
    return new Promise((res) => {
      const u = new SpeechSynthesisUtterance(text);
      const v = this._browserVoice();
      if (v) { u.voice = v; u.lang = v.lang; }
      u.rate = this.rate;
      let done = false;
      const finish = (ok) => { if (!done) { done = true; clearTimeout(t); res(ok); } };
      const t = setTimeout(() => finish(true), 3000 + text.length * 110 / this.rate);
      u.onstart = () => onStart?.(performance.now());
      u.onend = () => finish(true);
      u.onerror = () => finish(false);
      ss.speak(u);
    });
  }

  /** Say something (question, narration): server voice if possible, else the browser's. Resolves when done. */
  async speak(text) {
    if (!text) return { engine: null };
    if (this.engine !== 'browser') {
      const c = await this.clip(text);
      if (c && this.ctx) { this.unlock(); const p = this.playClip(c); await p.ended; return { engine: c.engine }; }
    }
    const ok = await this.speakBrowser(text);
    return { engine: ok ? 'browser' : null };
  }

  /** Decode one buffer per option for this trial: the item's own audio file if it has one, else its spoken text. */
  async prepareStimuli(options) {
    this.stim.clear();
    await Promise.all(options.map(async (o) => {
      let c = o.audio ? await this.fileClip(o.audio) : null;
      if (!c && this.engine !== 'browser') c = await this.clip(o.text || o.label);
      this.stim.set(o.id, c);
    }));
    const missing = options.filter((o) => !this.stim.get(o.id)).length;
    return { ready: options.length - missing, missing };
  }

  /** Play option `o` NOW (call inside the onset frame). Returns what to store in the marker. */
  playStim(o, frameTs) {
    this.stopSources();                           // never two stimuli at once
    const c = this.stim.get(o.id);
    if (c && this.ctx && this.ctx.state === 'running') {
      const p = this.playClip(c);
      const lag = performance.now() - frameTs;    // JS time between the frame timestamp and the start call
      return { audio_engine: c.engine, audio_timing: 'buffer', audio_latency_ms: p.latency_ms,
        audio_offset_ms: Math.round((lag + (p.latency_ms || 0)) * 10) / 10, audio_ms: c.duration_ms };
    }
    window.speechSynthesis?.cancel();
    this.speakBrowser(o.text || o.label);
    return { audio_engine: 'browser', audio_timing: 'approximate' };
  }

  stopSources() { for (const s of this.sources) { try { s.stop(); } catch { /* already stopped */ } } this.sources.clear(); }
  stopAll() { this.stopSources(); try { window.speechSynthesis?.cancel(); } catch { /* ignore */ } }
}

/** Browser voices for the Setup dropdown (they load asynchronously in Chrome). */
export function browserVoices() {
  return new Promise((res) => {
    const ss = window.speechSynthesis;
    if (!ss) return res([]);
    const now = ss.getVoices();
    if (now.length) return res(now);
    const t = setTimeout(() => res(ss.getVoices()), 1500);
    ss.addEventListener?.('voiceschanged', () => { clearTimeout(t); res(ss.getVoices()); }, { once: true });
  });
}

/** A few good free neural voices (edge-tts). Any edge-tts voice name works; type it in the box. */
export const EDGE_VOICES = [
  ['en-US-AndrewMultilingualNeural', 'Andrew (US, warm)'], ['en-US-AvaMultilingualNeural', 'Ava (US, expressive)'],
  ['en-US-BrianMultilingualNeural', 'Brian (US, casual)'], ['en-US-EmmaMultilingualNeural', 'Emma (US, cheerful)'],
  ['en-US-AriaNeural', 'Aria (US)'], ['en-US-GuyNeural', 'Guy (US)'], ['en-GB-RyanNeural', 'Ryan (UK)'],
  ['en-GB-SoniaNeural', 'Sonia (UK)'], ['en-AU-WilliamNeural', 'William (AU)'],
];

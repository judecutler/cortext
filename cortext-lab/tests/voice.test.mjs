// Narrator text + audio helpers. Run: node --test tests/
import test from 'node:test';
import assert from 'node:assert/strict';
import { narrationFacts, templateNarration, composeNarration, leadingSilence, NARRATOR_MOODS } from '../web/js/voice.js';
import { mulberry32 } from '../web/js/util.js';

const trial = (clicked = null) => ({
  prompt: 'Which one is the dog?', trial_index: 2, clicked_answer_id: clicked,
  options: [['cat', 'A'], ['dog', 'B'], ['cup', 'C'], ['circle', 'D']].map(([id, letter], position) => ({ id, letter, position, label: id, text: id, is_target: id === 'dog' })),
});
const pred = (over = {}) => ({ source: 'eeg', best_option_id: 'dog', probabilities: { cat: 0.12, dog: 0.78, cup: 0.06, circle: 0.04 }, ...over });

test('facts: guess, percent, runner-up; never the true answer', () => {
  const f = narrationFacts({ trial: trial(), pred: pred(), name: 'Jude', when: 'before_click' });
  assert.equal(f.guess_letter, 'B');
  assert.equal(f.guess_label, 'dog');
  assert.equal(f.confidence_percent, 78);
  assert.equal(f.runner_up_label, 'cat');
  assert.equal(f.runner_up_percent, 12);
  assert.equal(f.trial_number, 3);
  assert.equal(f.click_matches_guess, undefined);   // before the click: nothing about the click
  assert.ok(!JSON.stringify(f).includes('is_target') && !('true_answer_id' in f));
});

test('facts: click only after the click; no prediction -> null', () => {
  const f = narrationFacts({ trial: trial('cat'), pred: pred(), name: '', when: 'after_click' });
  assert.equal(f.clicked_letter, 'A');
  assert.equal(f.click_matches_guess, false);
  assert.equal(f.name, null);
  assert.equal(narrationFacts({ trial: trial(), pred: { timeout: true } }), null);
  assert.equal(narrationFacts({ trial: trial(), pred: null }), null);
  // scores without probabilities fall back to the confidence field
  assert.equal(narrationFacts({ trial: trial(), pred: pred({ probabilities: null, confidence: 0.55 }) }).confidence_percent, 55);
});

test('template: says the percent, the letter, the label and the name', () => {
  const rng = mulberry32(1);
  for (let i = 0; i < 20; i++) {
    const t = templateNarration(narrationFacts({ trial: trial(), pred: pred(), name: 'Jude', when: 'before_click' }), rng);
    assert.match(t, /78%/); assert.match(t, /\bB\b/); assert.match(t, /dog/); assert.match(t, /Jude/); assert.match(t, /brain waves/);
  }
  const t = templateNarration(narrationFacts({ trial: trial('dog'), pred: pred(), name: 'Jude', when: 'after_click' }), rng);
  assert.match(t, /agree/);
  assert.match(templateNarration(narrationFacts({ trial: trial(), pred: pred(), name: '' }), mulberry32(3)), /you|your/i);
});

test('template: honest about simulated predictions and low confidence', () => {
  const sim = templateNarration(narrationFacts({ trial: trial(), pred: pred({ source: 'mock' }), name: 'Jude' }));
  assert.match(sim, /Simulation only/);
  assert.doesNotMatch(sim, /brain patterns/);
  const simEeg = templateNarration(narrationFacts({ trial: trial(), pred: pred({ source: 'eeg', simulated: true }), name: 'Jude' }));
  assert.match(simEeg, /Simulated EEG/); assert.match(simEeg, /78%/); assert.doesNotMatch(simEeg, /mock/);
  const low = templateNarration(narrationFacts({ trial: trial(), pred: pred({ probabilities: { cat: 0.3, dog: 0.32, cup: 0.2, circle: 0.18 } }) }), mulberry32(2));
  assert.match(low, /32%/);
  assert.match(low, /tricky|secrets/);
  for (const p of [0.45, 0.6]) {   // every band says the name (the demo line: "Based on Jude's brain waves, I'm thinking answer B is the dog")
    const r = { cat: (1 - p) / 3, dog: p, cup: (1 - p) / 3, circle: (1 - p) / 3 };
    for (let i = 0; i < 10; i++) assert.match(templateNarration(narrationFacts({ trial: trial(), pred: pred({ probabilities: r }), name: 'Jude' }), mulberry32(i)), /Jude/);
  }
  for (let i = 0; i < 10; i++) assert.match(templateNarration(narrationFacts({ trial: trial(), pred: pred({ probabilities: { cat: 0.3, dog: 0.32, cup: 0.2, circle: 0.18 } }), name: 'Jude' }), mulberry32(i)), /Jude/);
  assert.match(templateNarration(null), /clear read|noisy/);
  const plain = templateNarration(narrationFacts({ trial: trial('dog'), pred: pred(), name: 'Jude', when: 'after_click' }), Math.random, { plain: true });
  assert.equal(plain, 'Model output: option B, the dog, at 78% confidence. The participant chose B, which agrees with the model.');
});

test('composeNarration: template provider needs no network', async () => {
  const n = await composeNarration(narrationFacts({ trial: trial(), pred: pred(), name: 'Jude' }), { provider: 'template' });
  assert.equal(n.provider, 'template');
  assert.match(n.text, /78%/);
});

test('composeNarration: LLM failure falls back to the template', async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = async () => ({ json: async () => ({ ok: false, error: 'no key' }) });
  try {
    const n = await composeNarration(narrationFacts({ trial: trial(), pred: pred(), name: 'Jude' }), { provider: 'anthropic', context: 'pirate' });
    assert.equal(n.provider, 'template');
    assert.equal(n.fallback_error, 'no key');
    assert.match(n.text, /78%/);
    let sent = null;
    globalThis.fetch = async (url, o) => { sent = JSON.parse(o.body); return { json: async () => ({ ok: true, text: 'Arr, 78% says the dog at B!', model: 'm' }) }; };
    const m = await composeNarration(narrationFacts({ trial: trial(), pred: pred(), name: 'Jude' }), { provider: 'ollama', context: 'Be a pirate.' });
    assert.equal(m.provider, 'ollama');
    assert.equal(sent.context, 'Be a pirate.');
    assert.equal(sent.facts.name, 'Jude');
  } finally { globalThis.fetch = realFetch; }
});

test('moods: every preset has text', () => {
  assert.ok(NARRATOR_MOODS.length >= 4);
  for (const [id, label, text] of NARRATOR_MOODS) assert.ok(id && label && text.length > 40);
});

test('leadingSilence finds the first sound', () => {
  const sr = 1000;
  const x = new Float32Array(1000);
  for (let i = 300; i < 600; i++) x[i] = 0.5 * Math.sin(i);
  x[10] = 0.001;   // tiny noise is ignored
  const s = leadingSilence(x, sr);
  assert.ok(Math.abs(s - 0.295) < 0.01, `got ${s}`);
  assert.equal(leadingSilence(new Float32Array(100), sr), 0);
});

test('uploads: an audio file named like an item becomes its spoken stimulus', async () => {
  const { buildSetFromFiles } = await import('../web/js/stimulusSets.js');
  const f = (name, type) => new File(['x'], name, { type });
  const noJson = await buildSetFromFiles([f('dog.svg', 'image/svg+xml'), f('dog.wav', 'audio/wav'), f('cat.svg', 'image/svg+xml')]);
  const byId = Object.fromEntries(noJson.config.items.map((i) => [i.id, i]));
  assert.equal(byId.dog.audio, 'dog.wav');
  assert.equal(byId.cat.audio, null);
  assert.ok('dog.wav' in noJson.files);
  const cfg = new File([JSON.stringify({ items: [{ id: 'dog', category: 'a', label: 'dog', image: 'dog.svg' }, { id: 'cat', category: 'a', label: 'cat', image: 'cat.svg' }] })], 's.json', { type: 'application/json' });
  const withJson = await buildSetFromFiles([cfg, f('dog.svg', 'image/svg+xml'), f('cat.svg', 'image/svg+xml'), f('DOG.mp3', 'audio/mpeg')]);
  assert.equal(withJson.config.items.find((i) => i.id === 'dog').audio, 'DOG.mp3');
});

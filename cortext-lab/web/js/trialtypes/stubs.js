// Planned trial types. Selecting one in setup is disabled until `implemented: true` and run() exist.
// Each stub documents what it should do so the next person can fill it in.

const notYet = (id) => async () => { throw new Error(`Trial type "${id}" is not implemented yet`); };

export const stubTypes = [
  {
    id: 'passive_view', implemented: false, run: notYet('passive_view'),
    label: 'Passive viewing (RSA / dissimilarity)',
    plan: 'Show many concepts one at a time, attention-matched (e.g. rare oddball detection task), no choice. ' +
          'Emit stim_on markers with concept id + category; analysis compares EEG distances with embedding distances.',
  },
  {
    id: 'picture_word_match', implemented: false, run: notYet('picture_word_match'),
    label: 'Picture + word match / mismatch',
    plan: 'Pair a picture with a written or spoken word; match vs mismatch conditions (stimulus-disconnect). ' +
          'Use stimulus types image | text | audio from the config; log condition in marker.extra.',
  },
  {
    id: 'imagery_hidden', implemented: false, run: notYet('imagery_hidden'),
    label: 'Imagery with hidden options',
    plan: 'Cue a concept, hide the options, participant imagines it; options appear only at the response phase. ' +
          'Train on viewing, test on imagining. Reuse the mcq_flash response phase.',
  },
  {
    id: 'free_choice', implemented: false, run: notYet('free_choice'),
    label: 'Free choice (favourite)',
    plan: 'No true answer: participant secretly picks a favourite among flashed options and reports it afterwards. ' +
          'Set true_answer_id from the report; caution: may decode affect rather than meaning.',
  },
];

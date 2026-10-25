"""
P300 stimulus presentation: 4-choice selection (e.g. food options).

All options stay on screen (dimmed). One option at a time "flashes" to full
brightness for 100 ms, with a 400 ms flash-to-flash interval (same timing as the
EPFL dataset). Flashes are locked to screen frames, and an LSL marker is sent on
the exact frame each flash appears.

Usage (run as a script, NOT inside a Jupyter notebook):
    python legacy/psychopy_p300/p300_stimulus.py --participant P01
    python legacy/psychopy_p300/p300_stimulus.py --participant test --choices 2 --reps 3 --windowed   # quick test

Put your images in an "images" folder next to this script:
    images/pizza.png, images/rice.png, images/noodles.png, images/water.png
If an image is missing, a blue square is shown instead so you can test without images.

Press ESC at any time to quit (the marker log is still saved).
"""
import argparse
import csv
import os
import random
from datetime import datetime

from psychopy import core, event, visual
from pylsl import StreamInfo, StreamOutlet, local_clock

HERE = os.path.dirname(os.path.abspath(__file__))   # images/ and data/ live next to this script

# ----------------------------- settings -----------------------------------
OPTIONS = [
    ("c", os.path.join(HERE, "images", "c.png")),
    ("cu", os.path.join(HERE, "images", "cu.png")),
    ("cnd", os.path.join(HERE, "images", "cnd.png")),
    ("d", os.path.join(HERE, "images", "d.png")),
]
FLASH_S = 0.100      # how long an option is bright
SOA_S = 0.400        # time from one flash onset to the next
DIM = 0.35           # opacity of options when not flashing (1.0 = full)
CUE_S = 2.0          # show which option to attend to
PRE_S = 1.0          # pause after cue, before flashing starts
POST_S = 1.5         # pause after flashing ends
REST_EVERY = 5       # offer a break every N choices

# ----------------------------- arguments ----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--participant", default="test")
parser.add_argument("--choices", type=int, default=20, help="number of target choices")
parser.add_argument("--reps", type=int, default=10, help="repetitions per choice")
parser.add_argument("--windowed", action="store_true", help="don't use fullscreen")
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

rng = random.Random(args.seed)
n_opts = len(OPTIONS)

# ----------------------------- LSL marker stream ---------------------------
info = StreamInfo("P300Markers", "Markers", 1, 0, "string", f"p300-stim-{args.participant}")
outlet = StreamOutlet(info)
markers_log = []  # (lsl_time, marker text)


def send_marker(text):
    """Called right after the screen flip (via win.callOnFlip)."""
    ts = local_clock()
    outlet.push_sample([text], ts)
    markers_log.append((ts, text))


# ----------------------------- window and stimuli --------------------------
win = visual.Window(
    fullscr=not args.windowed,
    size=(1280, 720),
    color=(-1, -1, -1),
    units="height",
    allowGUI=False,
    waitBlanking=True,
)
win.recordFrameIntervals = True

fps = win.getActualFrameRate(nIdentical=20, nMaxFrames=240, nWarmUpFrames=20, threshold=1)
if fps is None:
    fps = 60.0
    print("WARNING: could not measure the refresh rate, assuming 60 Hz")
flash_frames = round(FLASH_S * fps)
soa_frames = round(SOA_S * fps)
print(f"Refresh rate: {fps:.1f} Hz")
print(f"Flash: {flash_frames} frames = {flash_frames / fps * 1000:.0f} ms (target {FLASH_S * 1000:.0f})")
print(f"SOA:   {soa_frames} frames = {soa_frames / fps * 1000:.0f} ms (target {SOA_S * 1000:.0f})")
if abs(soa_frames / fps - SOA_S) > 0.005:
    print("WARNING: this display can't hit the target timing exactly. "
          "60 Hz displays work best (100 ms = 6 frames, 400 ms = 24 frames).")

SIZE = 0.24
XS = [(i - (n_opts - 1) / 2) * 0.34 for i in range(n_opts)]


class OptionStim:
    def __init__(self, label, image_path, x):
        if os.path.exists(image_path):
            self.pic = visual.ImageStim(win, image=image_path, pos=(x, 0), size=(SIZE, SIZE))
        else:
            self.pic = visual.Rect(win, width=SIZE, height=SIZE, pos=(x, 0),
                                   fillColor="steelblue", lineColor=None)
        self.label = visual.TextStim(win, text=label, pos=(x, -SIZE / 2 - 0.06),
                                     height=0.045, color="white", opacity=0.7)

    def draw(self, bright):
        self.pic.opacity = 1.0 if bright else DIM
        self.pic.draw()
        self.label.draw()


stims = [OptionStim(label, path, x) for (label, path), x in zip(OPTIONS, XS)]
cue_box = visual.Rect(win, width=SIZE + 0.04, height=SIZE + 0.04, lineColor="lime",
                      lineWidth=6, fillColor=None)
cue_text = visual.TextStim(win, text="", pos=(0, 0.3), height=0.06, color="lime")
msg = visual.TextStim(win, text="", pos=(0, 0), height=0.045, color="white", wrapWidth=1.4)
fixation = visual.TextStim(win, text="+", pos=(0, 0.3), height=0.08, color="white")


def check_quit():
    if "escape" in event.getKeys(["escape"]):
        raise KeyboardInterrupt


def draw_all(bright_idx=None):
    for i, s in enumerate(stims):
        s.draw(bright=(i == bright_idx))


def show_message(text, wait_for_space=True, seconds=2.0):
    msg.text = text
    if wait_for_space:
        event.clearEvents()
        while "space" not in event.getKeys(["space", "escape"]):
            msg.draw()
            win.flip()
    else:
        for _ in range(round(seconds * fps)):
            msg.draw()
            win.flip()
            check_quit()


def wait_frames(seconds, draw_fn=None, marker=None):
    for f in range(round(seconds * fps)):
        if draw_fn:
            draw_fn()
        if f == 0 and marker:
            win.callOnFlip(send_marker, marker)
        win.flip()
        check_quit()


def make_flash_sequence(n_reps):
    """Each repetition = every option once, random order. Avoid back-to-back repeats."""
    seq, last = [], None
    for _ in range(n_reps):
        while True:
            perm = list(range(n_opts))
            rng.shuffle(perm)
            if perm[0] != last:
                break
        seq.extend(perm)
        last = perm[-1]
    return seq


def run_choice(choice_idx, target):
    label = OPTIONS[target][0]
    cue_text.text = f"Focus on: {label}"

    def draw_cue():
        draw_all()
        cue_box.pos = (XS[target], 0)
        cue_box.draw()
        cue_text.draw()

    wait_frames(CUE_S, draw_cue, marker=f"cue|choice={choice_idx}|target={target}")
    wait_frames(PRE_S, lambda: (draw_all(), fixation.draw()))

    seq = make_flash_sequence(args.reps)
    win.callOnFlip(send_marker, f"choice_start|choice={choice_idx}|target={target}")
    for k, opt in enumerate(seq):
        rep = k // n_opts
        marker = f"flash|choice={choice_idx}|rep={rep}|opt={opt}|target={int(opt == target)}"
        for f in range(soa_frames):
            draw_all(bright_idx=opt if f < flash_frames else None)
            if f == 0:
                win.callOnFlip(send_marker, marker)
            win.flip()
            check_quit()

    wait_frames(POST_S, draw_all, marker=f"choice_end|choice={choice_idx}")


# ----------------------------- main ---------------------------------------
targets = [i % n_opts for i in range(args.choices)]
rng.shuffle(targets)

try:
    show_message(
        "P300 task\n\n"
        "Four options are shown. One will be highlighted with a green box.\n"
        "Watch that option and silently COUNT each time it flashes.\n"
        "Try not to blink or move during the flashing.\n\n"
        "Press SPACE to start."
    )
    send_marker("experiment_start")
    for c, target in enumerate(targets):
        run_choice(c, target)
        if (c + 1) % REST_EVERY == 0 and c + 1 < len(targets):
            show_message(f"Break ({c + 1}/{len(targets)} done).\nPress SPACE when ready.")
    send_marker("experiment_end")
    show_message("Done. Thank you!", wait_for_space=False, seconds=2.0)
except KeyboardInterrupt:
    print("Stopped early with ESC.")
finally:
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(HERE, "data", f"{args.participant}_{stamp}_markers.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lsl_time", "marker"])
        w.writerows(markers_log)
    print(f"Saved {len(markers_log)} markers to {path}")
    print(f"Dropped frames: {win.nDroppedFrames}")
    win.close()
    core.quit()
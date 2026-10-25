"""P300 signal processing shared by the live bridge scorer and the offline trainer (eeg/train_p300.py).

Everything here must behave identically live and offline, otherwise a model trained on recorded sessions would see
different features than the live scorer gives it. So:
  * the filter is CAUSAL (a running Butterworth band-pass, applied sample by sample in time order, live and offline);
  * epochs are cut by TIMESTAMP and averaged into fixed time bins, so dropped Bluetooth samples don't shift anything;
  * features = per-channel mean amplitude in BIN_MS bins from TMIN to TMAX after each stimulus onset, minus the mean of
    the BASELINE window before it. Units are microvolts.

Needs numpy + scipy (both already in the repo's requirements.txt).
"""
import json

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

import filters as F

BAND = (0.5, 20.0)        # Hz. Causal filters bend slow waves; 0.5 Hz / order 2 keeps the P300 shape (1 Hz/order 4 flipped it)
TMIN, TMAX = 0.0, 0.8     # s after onset
BASELINE = (-0.1, 0.0)    # s
BIN_MS = 40               # 20 bins over 0-800 ms
HEURISTIC_WINDOW = (0.25, 0.45)   # s: P300 peak latency range
REJECT_UV = 100.0         # an epoch whose filtered, baseline-corrected signal exceeds this anywhere is rejected
# Channels whose 1-20 Hz signal is above this std are flagged "noisy" (a clean P300-band EEG is ~5-25 uV std).
NOISY_STD_UV = 40.0
FLAT_STD_UV = 0.5
# TP9/TP10 are the Muse's P300 sites (temporo-parietal, referenced to Fpz); they are not on the Unicorn, so its picks
# are unchanged.
PARIETAL = ["Pz", "POz", "Cz", "PO7", "PO8", "Oz", "PO3", "PO4", "P3", "P4", "CPz", "TP9", "TP10"]
UNICORN_MONTAGE = ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "POz", "PO8"]
MUSE_MONTAGE = ["TP9", "AF7", "AF8", "TP10"]        # Muse 2 / Muse S / Muse 2016, as muselsl names them, 256 Hz


def reject_picks(labels):
    """Channels the artifact threshold looks at. Forehead channels (Fp*, AF*) are left out when the headset has others:
    on a Muse every blink is 100+ uV at AF7/AF8 and would throw away the epoch although the P300 channels (TP9/TP10)
    are fine. None = check every channel (the Unicorn has no Fp/AF channels, so nothing changes for it)."""
    labels = [str(l) for l in (labels or [])]
    keep = [i for i, l in enumerate(labels) if not l.upper().startswith(("FP", "AF"))]
    return keep if keep and len(keep) < len(labels) else None


class CausalFilter:
    """Band-pass that keeps its state between chunks (so a live stream and an offline pass give the same output).
    The default filter chain (filters.DEFAULT_SPEC); the bridge itself now runs a filters.Chain."""

    def __init__(self, srate, n_channels, band=BAND, order=2):
        self.sos = butter(order, [band[0], band[1]], btype="bandpass", fs=srate, output="sos")
        self.zi = None
        self.n_channels = n_channels

    def __call__(self, x):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return x
        if self.zi is None:  # start from the first sample's value to avoid a huge step response from the DC offset
            self.zi = sosfilt_zi(self.sos)[:, :, None] * x[0][None, None, :]
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


def n_bins():
    return int(round((TMAX - TMIN) * 1000 / BIN_MS))


def epoch(ts, data, onset, reject_uv=REJECT_UV, check=None):
    """Return a (n_channels, n_bins) baseline-corrected epoch for one onset, or None if the data doesn't cover it.

    ts: (n,) sample timestamps in seconds (sorted); data: (n, C) filtered microvolts; onset: seconds.
    reject_uv: the running filter chain's artifact threshold (default REJECT_UV).
    check: channel indices the threshold applies to (reject_picks(labels)); None = all channels.
    """
    lo = np.searchsorted(ts, onset + BASELINE[0], "left")
    hi = np.searchsorted(ts, onset + TMAX, "right")
    if lo >= hi or ts[lo] > onset + BASELINE[0] + 0.02 or ts[hi - 1] < onset + TMAX - 0.02:
        return None
    t = ts[lo:hi] - onset
    x = data[lo:hi]
    base = x[(t >= BASELINE[0]) & (t < BASELINE[1])]
    if len(base) == 0:
        return None
    x = x - base.mean(axis=0)
    if np.abs(x if check is None else x[:, check]).max() > reject_uv:
        return None  # artifact (blink, movement, loose electrode): reject rather than let it drive the score
    nb = n_bins()
    idx = np.floor((t - TMIN) * 1000 / BIN_MS).astype(int)
    out = np.empty((data.shape[1], nb))
    for b in range(nb):
        m = idx == b
        if not m.any():
            return None  # a gap in the data (dropped samples): don't guess
        out[:, b] = x[m].mean(axis=0)
    return out


def epochs(ts, data, onsets, reject_uv=REJECT_UV, check=None):
    """List of epochs (None where not covered or rejected as an artifact)."""
    return [epoch(ts, data, o, reject_uv, check) for o in onsets]


def heuristic_score(ep, labels):
    """Untrained score: mean amplitude 250-500 ms over parietal/central channels (bigger = more P300-like)."""
    picks = [i for i, l in enumerate(labels) if l in PARIETAL] or list(range(ep.shape[0]))
    b0 = int((HEURISTIC_WINDOW[0] - TMIN) * 1000 / BIN_MS)
    b1 = int((HEURISTIC_WINDOW[1] - TMIN) * 1000 / BIN_MS)
    return float(ep[picks, b0:b1].mean())


class LinearP300:
    """Shrinkage LDA on flattened epochs. score(x) is a log-likelihood ratio target vs non-target (Gaussian, shared cov).

    Saved/loaded as .npz so a model trained offline (eeg/train_p300.py) or adapted live can be reused.
    `filters` is the filter chain (bridge/filters.py) its features were made with; the bridge only uses the model
    while that chain runs. Files saved before chains existed were all made with filters.DEFAULT_SPEC.
    """

    def __init__(self, w=None, b=0.0, mean=None, std=None, labels=None, temperature=1.0, meta=None, filters=None):
        self.w, self.b, self.mean, self.std = w, b, mean, std
        self.labels = list(labels or [])
        self.temperature = temperature
        self.meta = dict(meta or {})
        self.filters = F.normalize(filters or F.DEFAULT_SPEC)

    @property
    def filter_signature(self):
        return F.signature(self.filters)

    @staticmethod
    def fit(X, y, labels, meta=None, filters=None):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-9
        Z = (X - mean) / std
        m1, m0 = Z[y == 1].mean(axis=0), Z[y == 0].mean(axis=0)
        R = np.vstack([Z[y == 1] - m1, Z[y == 0] - m0])
        cov, _ = ledoit_wolf(R)
        w = np.linalg.solve(cov, m1 - m0)
        b = -0.5 * w @ (m1 + m0)
        return LinearP300(w, b, mean, std, labels, 1.0, meta, filters)

    def score(self, X):
        Z = (np.atleast_2d(X) - self.mean) / self.std
        return Z @ self.w + self.b

    def reorder_for(self, labels):
        """Return the channel index mapping live labels -> model order, or raise if a channel is missing."""
        missing = [l for l in self.labels if l not in labels]
        if missing:
            raise ValueError(f"model needs channels {missing} that the EEG stream does not have ({labels})")
        return [labels.index(l) for l in self.labels]

    def save(self, path):
        bp = next((l for l in self.filters["layers"] if l["enabled"] and l["type"] == "bandpass"), None)
        band = (bp["low"], bp["high"]) if bp else (0.0, 0.0)   # informational only; filter_chain is what counts
        np.savez(path, w=self.w, b=self.b, mean=self.mean, std=self.std, labels=np.array(self.labels),
                 temperature=self.temperature, band=np.array(band), tmin=TMIN, tmax=TMAX, bin_ms=BIN_MS,
                 baseline=np.array(BASELINE), meta=np.array(repr(self.meta)),
                 filter_chain=np.array(json.dumps(self.filters)))

    @staticmethod
    def load(path):
        z = np.load(path, allow_pickle=False)
        if float(z["bin_ms"]) != BIN_MS or float(z["tmax"]) != TMAX:
            raise ValueError("model was trained with different epoch settings than this code uses")
        if "filter_chain" in z.files:
            try:
                filters = F.normalize(json.loads(str(z["filter_chain"])))
            except (ValueError, F.FilterError) as exc:
                raise ValueError(f"model's filter chain is unreadable: {exc}")
        elif tuple(z["band"]) == BAND:
            filters = F.normalize(F.DEFAULT_SPEC)    # saved before filter chains existed: the default chain
        else:
            raise ValueError("model was trained with different filter settings than this code uses")
        meta = {}
        try:
            import ast
            meta = ast.literal_eval(str(z["meta"]))
        except Exception:
            pass
        return LinearP300(z["w"], float(z["b"]), z["mean"], z["std"], [str(l) for l in z["labels"]],
                          float(z["temperature"]), meta, filters)


def ledoit_wolf(X):
    """Ledoit-Wolf shrunk covariance of centered data X (n, d). Returns (cov, shrinkage)."""
    n, d = X.shape
    S = X.T @ X / n
    mu = np.trace(S) / d
    F = mu * np.eye(d)
    X2 = X ** 2
    phi = (X2.T @ X2 / n - S ** 2).sum()
    gamma = ((S - F) ** 2).sum()
    k = 0.0 if gamma == 0 else min(1.0, max(0.0, (phi / gamma) / n))
    return (1 - k) * S + k * F, k


def option_evidence(flash_options, flash_scores, option_ids):
    """Sum and mean of per-flash scores per option. Sum of LLRs = log-evidence that the option is the target."""
    sums = {o: 0.0 for o in option_ids}
    counts = {o: 0 for o in option_ids}
    for o, s in zip(flash_options, flash_scores):
        if o in sums and s is not None and np.isfinite(s):
            sums[o] += float(s)
            counts[o] += 1
    means = {o: (sums[o] / counts[o] if counts[o] else 0.0) for o in option_ids}
    return sums, means, counts


def softmax(d, k=1.0):
    keys = list(d)
    if not keys:
        return {}
    v = np.array([k * d[x] for x in keys], float)
    v = np.exp(v - v.max())
    v /= v.sum()
    return {x: float(p) for x, p in zip(keys, v)}


# Percentages for the untrained "p3000 function" (pipeline/make_p3000_function.py). A fixed softmax slope on summed
# microvolts is only right for one noise level: with realistic 1/f background noise it said 96% on trials with no P300
# at all (simulated). So its evidence is first divided by THIS trial's own noise: the spread of its per-flash scores
# times sqrt(flashes per option), i.e. how many standard errors each option's total is. On simulated EEG with 1/f noise
# (3 rounds) slope 0.5 gives about 40% average stated confidence and at most ~60% with no P300, against 39-47% accuracy
# for a 5-8 uV P300. The cap keeps the narrator from ever saying "100%" for an uncalibrated detector.
P3000_Z_SLOPE = 0.5
P3000_CAP = 0.9


def noise_normalised_probs(flash_options, flash_scores, option_ids, slope=P3000_Z_SLOPE, cap=P3000_CAP):
    """softmax(slope * z) with z = option sum / (std of this trial's flash scores * sqrt(n flashes of that option)).
    The top probability is capped at `cap` (the remainder is shared by the others in proportion). Returns (probs, z)."""
    sums, _, counts = option_evidence(flash_options, flash_scores, option_ids)
    v = np.array([x for x in flash_scores if x is not None and np.isfinite(x)], float)
    sd = float(v.std(ddof=1)) if len(v) > 2 else 0.0
    if not sd > 1e-9:
        return {o: 1.0 / len(option_ids) for o in option_ids}, {o: 0.0 for o in option_ids}
    z = {o: sums[o] / (sd * np.sqrt(max(counts[o], 1))) for o in option_ids}
    return cap_top(softmax(z, slope), cap), z


def cap_top(pr, cap=P3000_CAP):
    """Limit the top probability to `cap`, sharing the rest among the others in proportion (ranking unchanged).
    Every live percentage goes through this: no uncalibrated scorer should announce "100% sure"."""
    if not pr or len(pr) < 2:
        return pr
    best = max(pr, key=pr.get)
    if pr[best] <= cap:
        return pr
    rest = 1.0 - pr[best]
    n_other = len(pr) - 1
    return {o: (cap if o == best else ((1.0 - cap) * p / rest if rest > 1e-12 else (1.0 - cap) / n_other)) for o, p in pr.items()}


def fit_temperature(trial_evidence):
    """Choose k for softmax(k * summed LLR) that maximizes the true option's log-probability.

    trial_evidence: list of (dict option->summed score, true_option). In-sample: an optimistic, rough calibration.
    """
    if not trial_evidence:
        return 1.0
    best_k, best_ll = 1.0, -np.inf
    for k in np.geomspace(0.02, 3.0, 40):
        ll = sum(np.log(max(softmax(ev, k)[t], 1e-12)) for ev, t in trial_evidence if t in ev)
        if ll > best_ll:
            best_k, best_ll = float(k), ll
    return best_k

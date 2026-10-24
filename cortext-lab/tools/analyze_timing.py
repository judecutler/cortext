#!/usr/bin/env python3
"""Summarise marker timing from a bridge CSV (bridge/logs/markers_*.csv). Standard library only.

    python tools/analyze_timing.py bridge/logs/markers_XXXX.csv
    python tools/analyze_timing.py markers.csv --event timing_flash --photodiode pd_onsets.csv

Reports
  * transit: bridge receive time minus the page timestamp mapped into bridge time (WebSocket + event loop delay;
    this is what the LSL timestamp correction removes)
  * onset-to-onset intervals of the chosen event vs the intended interval (page-side scheduling jitter)
  * optional screen lag: pass a CSV with one column `lsl_timestamp` holding photodiode onset times (same clock as
    the LSL timestamps in the marker CSV) and each marker is paired with the first photodiode onset after it.
"""
import argparse
import csv
import statistics as st


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def summarise(name, xs, unit="ms"):
    if not xs:
        print(f"{name}: no data")
        return
    print(f"{name}: n={len(xs)} mean={st.mean(xs):.2f}{unit} sd={st.pstdev(xs):.2f} "
          f"median={st.median(xs):.2f} p95={pct(xs, 95):.2f} max={max(xs):.2f} min={min(xs):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--event", default="flash_on")
    ap.add_argument("--interval", type=float, help="intended onset-to-onset interval in ms (timing test)")
    ap.add_argument("--photodiode", help="CSV with an lsl_timestamp column of photodiode onsets (seconds)")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.csv, newline="")) if r["event"] == a.event]
    print(f"{len(rows)} '{a.event}' markers")
    summarise("transit (bridge recv - mapped page time)", [float(r["est_transit_ms"]) for r in rows if r["est_transit_ms"]])
    ts = [float(r["page_t_ms"]) for r in rows if r["page_t_ms"]]
    diffs = [b - a_ for a_, b in zip(ts, ts[1:])]
    summarise("onset-to-onset interval (page clock)", diffs)
    if a.interval and diffs:
        summarise(f"interval error vs {a.interval} ms", [d - a.interval for d in diffs])
    if a.photodiode:
        pd = sorted(float(r["lsl_timestamp"]) for r in csv.DictReader(open(a.photodiode, newline="")))
        lags, j = [], 0
        for r in rows:
            if not r["lsl_timestamp"]:
                continue
            t = float(r["lsl_timestamp"])
            while j < len(pd) and pd[j] < t - 0.005:
                j += 1
            if j < len(pd):
                lags.append((pd[j] - t) * 1000)
        summarise("marker -> photodiode onset (screen lag incl. display latency)", lags)
        print("Positive = screen lights up AFTER the marker. Subtract the mean from analysis epochs if you want exact onsets;"
              " the SD is the jitter you cannot remove.")


if __name__ == "__main__":
    main()

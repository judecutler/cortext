#!/usr/bin/env python3
"""Stream a Muse headband (Muse 2, Muse S, Muse 2016) to LSL so CorText Lab can record and score it.

    pip install muselsl                                  # once (uses Bluetooth LE through bleak; no dongle needed)
    python cortext-lab/eeg/stream_muse.py                # finds the first Muse in range and streams it
    python cortext-lab/eeg/stream_muse.py --list         # just list the Muses in range
    python cortext-lab/eeg/stream_muse.py --address 00:55:DA:B0:12:34   # a specific one (several in the room)

What goes to LSL is the Muse's EEG in microvolts, 256 Hz, channels TP9 AF7 AF8 TP10 (+ "Right AUX", which the bridge
drops). Then run the bridge as usual; it finds any stream of type EEG. On the Agent page pick
"p3000_function_muse" (the bridge also switches to it by itself when a Muse connects while a p3000 function runs).

Alternatives that also work with the bridge: BlueMuse (Windows, "Start LSL stream") and `muselsl stream --lsltime`.
The Mind Monitor phone app does NOT (it sends OSC, not LSL).

Tips: wet the forehead and behind-the-ear sensors a little; push the band down so TP9/TP10 touch skin behind the
ears, not hair; wait until the chips on the Experiment tab are green (often 1-2 minutes for dry sensors).
"""
import argparse
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--address", default=None, help="Bluetooth address (or macOS UUID) of the Muse")
    ap.add_argument("--name", default=None, help="Muse name to look for, e.g. Muse-12AB")
    ap.add_argument("--list", action="store_true", help="list Muses in range and exit")
    ap.add_argument("--backend", default="auto", help="muselsl backend: auto (bleak), bgapi (BLED112 dongle), gatt")
    args = ap.parse_args()
    try:
        from muselsl import list_muses, stream
    except ImportError:
        sys.exit("muselsl is not installed. Run:  pip install muselsl")

    address = args.address
    if not address:
        print("Looking for Muses (turn the headband on: lights should be moving)...")
        muses = list_muses(backend=args.backend) or []
        for m in muses:
            print(f"  {m.get('name')}  {m.get('address')}")
        if args.list:
            return
        if args.name:
            muses = [m for m in muses if str(m.get("name", "")).lower() == args.name.lower()]
        if not muses:
            sys.exit("No Muse found. Is it on, charged, and not connected to a phone app? (Close Muse/Mind Monitor.)")
        if len(muses) > 1:
            print(f"Several Muses in range; using {muses[0].get('name')}. Pick one with --address or --name.")
        address = muses[0]["address"]
    print(f"Streaming {address} to LSL (type EEG). Leave this running; Ctrl+C to stop.")
    # lsl_time=True is essential: by default muselsl stamps samples with time.time() (wall clock), while the app's
    # markers use LSL's clock, so every epoch would be cut at the wrong place. (The bridge also detects and corrects
    # wall-clock stamps, for people who run `muselsl stream` without --lsltime.)
    try:
        stream(address, backend=args.backend, lsl_time=True)
    except TypeError:                       # older muselsl without these keywords
        stream(address)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stream the g.tec headset (Unicorn / BCI Core-8) to LSL so CorText Lab can record and score it.

    python cortext-lab/eeg/stream_eeg.py            # finds the first headset in range, opens the scope window
    python cortext-lab/eeg/stream_eeg.py --no-scope # no window (Ctrl+C to stop)

This replaces bci.py while running an experiment: only one program can hold the Bluetooth headset at a time.
The scope window shows the same filtered view as bci.py. What goes to LSL is the RAW EEG (8 channels, microvolts,
250 Hz, electrode names Fz C3 Cz C4 Pz PO7 POz PO8); the bridge filters it itself so recordings keep the raw data.

Needs gpype (already installed if bci.py runs) plus pylsl:  pip install "gpype[lsl]"
"""
import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="CorTextEEG", help="LSL stream name (the bridge finds any stream of type EEG)")
    ap.add_argument("--serial", default=None, help="headset serial, if several are in range")
    ap.add_argument("--no-scope", action="store_true", help="don't open the scope window")
    ap.add_argument("--mains", type=int, default=60, choices=[50, 60], help="power-line frequency for the scope's notch")
    args = ap.parse_args()

    try:
        import gpype as gp
    except ImportError:
        sys.exit("gpype is not installed. Run:  pip install \"gpype[lsl]\"")
    if not hasattr(gp, "LSLSender"):
        sys.exit("This gpype has no LSLSender. Run:  pip install -U \"gpype[lsl]\"")

    app = None if args.no_scope else gp.MainApp()
    p = gp.Pipeline()
    source = gp.BCICore(serial=args.serial) if args.serial else gp.BCICore()   # first headset in range
    eeg = gp.ChannelSelector(roles=gp.Constants.ChannelRoles.SIGNAL)            # drop the timing/index channels
    p.connect(source, eeg)
    p.connect(eeg, gp.LSLSender(stream_name=args.name))                        # raw EEG -> LSL

    if app is not None:  # same view as bci.py
        bandpass = gp.Bandpass(f_lo=1, f_hi=30)
        notch = gp.Bandstop(f_lo=args.mains - 2, f_hi=args.mains + 2)
        scope = gp.TimeSeriesScope(amplitude_limit=50, time_window=10)
        p.connect(eeg, bandpass)
        p.connect(bandpass, notch)
        p.connect(notch, scope)
        app.add_widget(scope)

    p.start()
    print(f"Streaming headset EEG to LSL as '{args.name}'. Now start the bridge: python cortext-lab/bridge/bridge.py")
    try:
        if app is not None:
            app.run()            # close the window to stop
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        p.stop()
        p.close()


if __name__ == "__main__":
    main()

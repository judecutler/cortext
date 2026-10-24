import gpype as gp

if __name__ == "__main__":
    app = gp.MainApp()
    p = gp.Pipeline()

    source = gp.BCICore()                        # finds the first headset in range
    bandpass = gp.Bandpass(f_lo=1, f_hi=30)      # keep brain rhythms
    notch = gp.Bandstop(f_lo=58, f_hi=62)        # remove 60 Hz US power-line noise
    scope = gp.TimeSeriesScope(amplitude_limit=50, time_window=10)

    p.connect(source, bandpass)
    p.connect(bandpass, notch)
    p.connect(notch, scope)
    app.add_widget(scope)

    p.start()
    app.run()      # close the window to stop
    p.stop()
    p.close()

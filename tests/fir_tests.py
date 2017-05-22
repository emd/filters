from nose import tools
import numpy as np
from matplotlib.mlab import psd
from filters.fir import Kaiser, NER


def test_Kaiser_getNumTaps():
    # From Oppenheim & Schafer "Discrete-Time Signal Processing" 3rd Ed.
    # Section 7.6.1 (pg 545), for a lowpass filter with
    #
    #   ripple = -60 dB, and
    #   Delta omega = 0.2 * pi,
    #
    # they obtain M = 37; the number of taps is M + 1 = 38, then.
    # However, :py:class:`Kaiser <filters.fir.Kaiser>` corresponds
    # to a Type I FIR filter, which requires an *odd* number of taps.
    # Thus, we should expect that the number of taps returned by
    # `Kaiser.getNumTaps()` is 39.
    #
    # Note: Oppenheim and Schafer utilize a frequency normalization
    # such that Delta omega = pi corresponds to the Nyquist *angular*
    # frequency (i.e. the normalized Nyquist frequency is unity).
    # To make contact with their calculations, then, we need to
    # ensure that
    #
    #   Delta omega / angular Nyquist rate = (0.2 * pi) / pi = 0.2
    Ntaps_expected = 39

    # Initialize frequencies with unity Nyquist frequency
    fNy = 1.
    Fs = 2 * fNy
    width = 0.2 * fNy
    f_6dB = np.array([0.5])

    # Scaling factors -- uniformly altering timescale with
    # a scaling factor should *not* alter required number of taps
    scalings = np.array([1., 10., 0.1])

    for scaling in scalings:
        lpf = Kaiser(
            -60,
            width * scaling,
            f_6dB * scaling,
            pass_zero=True,
            Fs=(Fs * scaling))

        tools.assert_equal(
            lpf.getNumTaps(),
            Ntaps_expected)

    return


def test_Kaiser_getResponse():
    # Create a high-pass filter
    ripple = -60
    width = 5e3
    f_6dB = 10e3
    Fs = 4e6
    hpf = Kaiser(ripple, width, f_6dB, pass_zero=False, Fs=Fs)

    # Power response @ `f_6dB` should be -6 dB
    f, H = hpf.getResponse(f=f_6dB)
    np.testing.assert_allclose(
        20 * np.log10(np.abs(H)),
        -6,
        rtol=0.1)

    # Power response (dB) in stopband should be <= `ripple`
    f = np.arange(0, f_6dB - (0.5 * width), 10)
    f, H = hpf.getResponse(f=f)
    tools.assert_less_equal(
        np.max(20 * np.log10(np.abs(H))),
        ripple)

    # Power response in passband should be:
    #
    #   >= (1 - `delta`), and
    #   <= (1 + `delta`),
    #
    # where `delta` is the ripple expressed in amplitude
    # (as opposed to dB)
    delta = 10 ** (ripple / 20.)

    f = np.arange(f_6dB + (0.5 * width), 0.5 * Fs, 1000)
    f, H = hpf.getResponse(f=f)

    tools.assert_greater_equal(
        np.min(np.abs(H)),
        1 - delta)

    tools.assert_less_equal(
        np.max(np.abs(H)),
        1 + delta)

    return


def test_Kaiser_applyTo():
    # Create a high-pass filter
    ripple = -60
    width = 5e3
    f_6dB = 10e3
    Fs = 200e3
    hpf = Kaiser(ripple, width, f_6dB, pass_zero=False, Fs=Fs)

    # Sum of two sinusoidal signals, with one signal clearly
    # in the stopband and the other clearly in the passband
    t = np.arange(0, 0.1, 1. / Fs)
    fstop = f_6dB - (0.5 * width)
    fpass = f_6dB + (0.5 * width)
    ystop = np.cos(2 * np.pi * fstop * t)
    ypass = np.cos(2 * np.pi * fpass * t)
    y = ystop + ypass

    # Apply filter and determine region w/o boundary effects
    yfilt = hpf.applyTo(y)
    valid = hpf.getValidSlice()

    # Compute autospectral densities of raw and filtered signal
    NFFT = 2048  # ~10 ms
    noverlap = NFFT // 2
    asd_raw, f = psd(y[valid], Fs=Fs, NFFT=NFFT, noverlap=noverlap)
    asd_filt, f = psd(yfilt[valid], Fs=Fs, NFFT=NFFT, noverlap=noverlap)

    # Compare to expectations
    tmp, Hstop = hpf.getResponse(f=fstop)
    tmp, Hpass = hpf.getResponse(f=fpass)

    dfstop = np.abs(fstop - f)
    dfpass = np.abs(fpass - f)
    ind_fstop = np.where(dfstop == np.min(dfstop))[0]
    ind_fpass = np.where(dfpass == np.min(dfpass))[0]

    tools.assert_almost_equal(
        asd_raw[ind_fstop] * (np.abs(Hstop) ** 2),
        asd_filt[ind_fstop])

    tools.assert_almost_equal(
        asd_raw[ind_fpass] * (np.abs(Hpass) ** 2),
        asd_filt[ind_fpass])

    return


def test_Kaiser_delay():
    # Create a bandpass filter
    ripple = -60
    width = 5e3
    f_6dB = [40e3, 60e3]
    Fs = 200e3
    bpf = Kaiser(ripple, width, f_6dB, pass_zero=False, Fs=Fs)

    # Create linearly chirping signal from 10 kHz to 100 kHz.
    # Note that for a linear chirp, the phase evolution goes as
    # (ramp rate) * t^2, and the instantaneous frequency goes
    # as 2 * (ramp rate) * t
    t = np.arange(0, 0.1, 1. / Fs)
    fstart = 10e3
    fstop = 50e3
    m = (fstop - fstart) / (t[-1] - t[0])
    f = fstart + (m * t)
    y = np.cos(2 * np.pi * f * t)

    # Filter chirped signal
    yfilt = bpf.applyTo(y)
    valid = bpf.getValidSlice()

    # Cross correlate raw and filtered signals
    xcorr = np.correlate(y[valid], yfilt[valid], mode='full')

    # Determine delay corresponding to each value in `xcorr`.
    # For `np.correlate(y1, y2, mode='full')`, the returned array
    # has dimensions `M + N - 1`, where `M = len(y1)` and `N = len(y2)`,
    # with the center point corresponding to zero delay.
    N = len(y[valid])
    tau = np.arange(-(N - 1), N - 1)

    # If filter really does not delay signal, then peak cross correlation
    # should occur at zero delay
    imax = np.where(np.abs(xcorr) == np.max(np.abs(xcorr)))[0]
    tools.assert_equal(tau[imax], 0)

    return


@tools.raises(ValueError)
def test_NER__init__ripple():
    # This ripple size is too large
    NER(0.1, [2., 3.], 1., 5.)


@tools.raises(ValueError)
def test_NER__init__trans():
    # This transition size is too small for the given passband
    NER(0.01, [2.25, 2.75], 1., 5., pass_zero=False)


def test_NER_getRippleDecibels():
    # Kaiser & Reed, RSI 48, 1447 (1977) state in their paper that for
    # epsilon = 0.02 the corresponding ripple value (in dB) is ~34
    filt = NER(0.02, [0.3], 0.2, 2.)
    tools.assert_almost_equal(filt.getRippleDecibels(), 34, delta=0.1)


def test_NER_getNumTaps():
    # Kaiser & Reed, RSI 48, 1447 (1977), Fig. 3 states that for
    # beta = 0.3, delta = 0.2, and epsilon = 0.02, they obtain
    # Np = 10 pairs of terms, for a total of 2 * Np + 1 = 21 taps
    filt = NER(0.02, [0.3], 0.2, 2.)
    tools.assert_equal(filt.getNumTaps(), 21)


def test_NER_getFrequencyResponse():
    # Create a bandpass filter that passes signal from 26 < f [Hz] < 28
    # for a signal sampled at 200 Hz
    bpf = NER(0.02, [26, 28], 0.5, 200, pass_zero=False)

    # The filter gain should be -3 dB (i.e. ~0.5) at the cutoff frequencies
    f, h = bpf.getFrequencyResponse(freqs_or_N=bpf.cutoff)
    np.testing.assert_almost_equal(np.abs(h), 10 ** -0.3, decimal=2)


def test_NER_applyTo_LPF():
    # Generate signal at `f0`
    Fs = 20
    t = np.arange(0, 10, 1. / Fs)
    f0 = 1 + np.random.rand()
    y0 = np.cos(2 * np.pi * f0 * t)

    # Add some distortion at 3 * `f0`
    y = y0 + (0.1 * np.cos(2 * np.pi * (3 * f0) * t))

    # Create low-pass filter that should only pass `f0`
    lpf = NER(0.02, [2 * f0], 0.25 * f0, Fs)

    # Filter out 3 * `f0` component and compare "valid" component
    # of result to `y0`
    yfilt = lpf.applyTo(y)
    valid = lpf.getValidSlice()
    np.testing.assert_almost_equal(yfilt[valid], y0[valid], decimal=2)


def test_NER_applyTo_BPF():
    # Generate signal at `f0`
    Fs = 20
    t = np.arange(0, 100, 1. / Fs)
    f0 = 1 + np.random.rand()
    y0 = np.cos(2 * np.pi * f0 * t)

    # Add some distortion at 3 * `f0`
    y = y0 + (0.1 * np.cos(2 * np.pi * (3 * f0) * t))

    # Add some low frequency variation
    y += 5 * np.cos(2 * np.pi * (0.01 * f0) * t)

    # Create band-pass filter that should only pass `f0`
    flo = 0.5 * f0
    fhi = 2 * f0
    bpf = NER(0.02, [flo, fhi], 0.1 * (fhi - flo), Fs, pass_zero=False)

    # Filter out 3 * `f0` component and compare "valid" component
    # of result to `y0`
    yfilt = bpf.applyTo(y)
    valid = bpf.getValidSlice()
    np.testing.assert_almost_equal(yfilt[valid], y0[valid], decimal=1)

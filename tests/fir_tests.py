from nose import tools
import numpy as np
from filters.fir import NER


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

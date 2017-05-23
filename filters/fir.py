'Class and functions for creating and using FIR filters.'


import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


class Kaiser(object):
    '''Type I FIR filter designed via Kaiser windowing.

    This is essentially a wrapper around Scipy's filter routines
    with additional insight from Kaiser & Reed, RSI 48, 1447 (1977),
    which describes requirements for a FIR filter with *zero* delay.

    '''
    def __init__(self, ripple, width, f_6dB, pass_zero=True, Fs=1.):
        '''Initialize Type I FIR filter designed via Kaiser windowing.

        Parameters:
        -----------
        ripple - float
            The ripple (in dB) of the filter's passband(s) and stopband(s).
            For example, if 1% variations in the passband amplitude are
            acceptable, then

                ripple = 20 * log10(0.01) = -40

            [ripple] = dB

        width - float
            The width of the transition region between passband and stopband.
            [width] = [Fs] (i.e. same units as sample rate)

        f_6dB - (`N`,), array_like
            A monotonically increasing array specifying the
            *power* 6-dB points of the filter; that is,

                -6 = 20 * log10(|H(f_6dB)|)

            Typically, in hardware applications, a filter's
            power 3-dB points (rather than 6-dB points) are
            specified. However, note that the power 6-dB point
            corresponds to the amplitude 3-dB point, and
            the *amplitude* 3-dB point is the convention
            used in scipy's filter-design routines.
            Compatibility with scipy's routines drives
            our atypical choice of specifying the filter's
            power 6-dB points.
            [f_6dB] = [Fs] (i.e. same units as sample rate)

        pass_zero - bool
            If True, the DC gain is unity; otherwise the DC gain is zero.

        Fs - float
            Sample rate of discrete-time signal.
            [Fs] = AU

        '''
        self.ripple = ripple
        self.width = width
        self.f_6dB = f_6dB
        self.pass_zero = pass_zero
        self.Fs = Fs

        self.b = self.getFilterCoefficients()

    def getNumTaps(self):
        'Get number of filter "taps" for desired ripple and width.'
        Ntaps = signal.kaiserord(
            -self.ripple,
            self.width / (0.5 * self.Fs))[0]

        # Ensure that the number of taps is *odd*, as required
        # for a Type I FIR filter
        Ntaps = (2 * (Ntaps // 2)) + 1

        return Ntaps

    def getFilterCoefficients(self):
        'Get feedforward filter coefficients.'
        return signal.firwin(
            self.getNumTaps(),
            cutoff=self.f_6dB,
            width=self.width,
            pass_zero=self.pass_zero,
            nyq=(0.5 * self.Fs))

    def getResponse(self, f=None):
        'Get frequency response H(f) of filter.'
        if f is not None:
            omega = np.pi * (f / (0.5 * self.Fs))
        else:
            omega = None

        omega, H = signal.freqz(self.b, worN=omega)

        if f is None:
            f = omega * (0.5 * self.Fs / np.pi)

        return f, H

    def plotResponse(self, f=None):
        'Plot frequency response |H(f)|.'
        f, H = self.getResponse(f=f)

        plt.figure()
        plt.semilogy(f, np.abs(H))
        plt.xlabel('f')
        plt.ylabel('|H(f)|')
        plt.show()

        return

    def applyTo(self, y):
        '''Apply filter to signal `y`, where `y` was sampled at `self.Fs`.

        An FIR filter is applied via convolution of the signal `y`
        with the filter coefficients `self.b`.

        Parameters:
        -----------
        y - array_like, (`N`,)
            The input signal to be filtered. The signal should
            be sampled at rate `self.Fs`.
            [y] = arbitrary units

        Returns:
        --------
        yfilt - array_like, (`N`)
            The filtered signal, of the same data type as
            the input signal `y`.
            [yfilt] = [y]

        '''
        # Obtain data type of signal
        y = np.asarray(y)
        dtype = y.dtype

        # When convolving a vector of shape (`N`,) with a
        # kernel of shape (`k`,) convolution is *fastest*
        # via an FFT if
        #
        #               k >= 4 log_2 (N)
        #
        # Otherwise, a straightforward convolution is faster.
        # Relevant background information here:
        #
        #       http://programmers.stackexchange.com/a/172839
        #
        if self.b.size >= (4 * np.log2(y.size)):
            yfilt = signal.fftconvolve(y, self.b, mode='same')
        else:
            yfilt = np.convolve(y, self.b, mode='same')

        # Return filtered signal with same type as input signal
        return yfilt.astype(dtype)

    def getValidSlice(self):
        '''Get slice corresponding to `valid` data points in filtered signal,
        for which boundary effects of the filter are not visible.

        '''
        # The filter has (2 * `N`) + 1 taps, so the first `N` and last `N`
        # points in the filtered signal display boundary effects
        N = (len(self.b) - 1) // 2

        return slice(N, -N, None)

'Class and functions for creating and using FIR filters.'


import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import kaiser_beta


class Kaiser(object):
    '''Type I FIR filter designed via Kaiser windowing.

    '''
    def __init__(self, Fs, stopband, ripple, width):
        '''Initialize Type I FIR filter designed via Kaiser windowing.

        Parameters:
        -----------
        Fs - float
            Sample rate of discrete-time signal.
            [Fs] = AU

        stopband - (`N`,), array_like, with `N` an *even* integer
            A monotonically increasing array, with each sequential pair
            specifying the lower and upper boundaries of a stopband.

            If a value in `stopband` sits below zero or above the
            Nyquist frequency (i.e. 0.5 * `Fs`), a ValueError is raised.
            A ValueError is also raised if `N` is not even.

            [stopband] = [Fs] (i.e. same units as sample rate)

        ripple - float
            The ripple (in dB) of the filter's stopband(s) and passband(s).
            For example, if 1% variations in the stopband amplitude are
            acceptable, then

                ripple = 20 * log10(0.01) = -40

            [ripple] = dB

        width - float
            The width of the transition region between passband and stopband.
            [width] = [Fs] (i.e. same units as sample rate)

        '''
        self.Fs = Fs
        self.ripple = ripple
        self.width = width

        self.b = self.getFilterCoefficients(stopband)

    def getNumTaps(self):
        'Get number of filter "taps" for desired ripple and width.'
        Ntaps = signal.kaiserord(
            -self.ripple,
            self.width / (0.5 * self.Fs))[0]

        # Ensure that the number of taps is *odd*, as required
        # for a Type I FIR filter
        Ntaps = (2 * (Ntaps / 2)) + 1

        return Ntaps

    def _processStopband(self, stopband):
        '''Process `stopband` to produce input that is compatible with
        :py:function:`firwin <scipy.signal.firwin>`.

        '''
        # Each *pair* of values in `stopband` defines a stopband
        if len(stopband) % 2 != 0:
            raise ValueError('`stopband` must be an array of *even* length')

        stopband = list(stopband)

        # Process lower bound
        if stopband[0] < 0:
            raise ValueError('`stopband` values must be >= 0')
        elif stopband[0] == 0:
            pass_zero = False
            stopband.remove(0)
        else:
            pass_zero = True

        # Process upper bound
        if stopband[-1] > (0.5 * self.Fs):
            raise ValueError('`stopband` values must be <= (0.5 * `Fs`)')
        elif stopband[-1] == (0.5 * self.Fs):
            stopband.pop()

        return stopband, pass_zero

    def getFilterCoefficients(self, stopband):
        'Get feedforward filter coefficients.'
        cutoff, pass_zero = self._processStopband(stopband)

        return signal.firwin(
            self.getNumTaps(),
            cutoff=cutoff,
            width=self.width,
            pass_zero=pass_zero,
            nyq=(0.5 * self.Fs))


class NER(object):
    '''Nearly Equal Ripple (NER) finite impulse response (FIR) filter object.

    This is essentially a wrapper around

                `:py:function:`firwin <scipy.signal.firwin>`

    with additional insight from Kaiser & Reed, RSI 48, 1447 (1977),
    which describes requirements:

        - linear phase FIR filters, and
        - number of taps for given ripple, passband, and transition band

    '''
    def __init__(self, ripple, cutoff, trans, Fs, pass_zero=True):
        '''Initialize NER filter.

        Parameters:
        -----------
        ripple - float
            The ripple in the filter's passband (1 +/- ripple) and
            stopband (0 +/- ripple). For the following formalism,
            it is assumed that

                                ripple <= 0.02

            If this condition is not satisfied, a ValueError is raised.
            [ripple] = unitless

        cutoff - 1d array_like, (`N`,)
            If `pass_zero` is True, pass DC to `cutoff[0]`,
            reject frequencies `cutoff[0]` < f < `cutoff[1]`,
            pass frequencies `cutoff[1]` < f < `cutoff[2]`, etc.
            If `pass_zero` is False, the opposite is true.

            For a typical example of a bandpass filter,
            `pass_zero` = False, `cutoff[0]` = fmin, and
            `cutoff[1]` = fmax.

            [cutoff] = [Fs]

        trans - float
            The transition bandwidth between the passband and the stopband.
            To have a "true" passband, we must have

                                `trans` < `passband`

            where `passband` is determined from the minimum frequency
            interval specified in `cutoff`. If this condition is not met,
            a Value Error is raised.
            [trans] = [Fs]

        Fs - float
            Sampling rate of digital signal.
            [Fs] = samples / [time], where [time] is any convenient unit

        pass_zero - bool
            If True, the filter's first passband is from DC to `cutoff[0]`.
            If False, the filter's first stopband is from DC to `cutoff[0]`.

        '''
        self.Fs = Fs

        # Is the condition on ripple size satisfied?
        if ripple <= 0.02:
            self.ripple = ripple
            self.ripple_dB = self.getRippleDecibels()
        else:
            raise ValueError('The NER formalism requires `ripple` <= 0.02')

        # Is the condition on transition bandwidth satisfied?
        intervals = np.diff(np.concatenate((
            np.array([0]), np.asarray(cutoff), np.array([Fs / 2.]))))

        if np.min(intervals) > (0.5 * trans):
            self.pass_zero = pass_zero
            self.cutoff = np.asarray(cutoff)
            self.trans = trans
        else:
            raise ValueError(
                'For the desired passband(s), the NER formalism requires '
                'that `trans` < %f' % (2 * np.min(intervals)))

        # Get filter (feedforward) coefficients
        self.b = self.getFilterCoefficients()

    def getRippleDecibels(self):
        'Get value of ripple in dB.'
        # Eq. (11) of Kaiser & Reed.
        return -20 * np.log10(self.ripple)

    def getNumTaps(self):
        '''Get number of "taps" for desired falloff and accuracy.
        The relevant formulas are given in Eq. (12) and (13) of
        Kaiser & Reed, RSI 48, 1447 (1977).

        '''
        # Eq. (12) of Kaiser & Reed
        if self.ripple_dB > 21:
            Kf = 0.13927 * (self.ripple_dB - 7.95)
        else:
            # Note: this limit should functionally never be used
            # due to the constraint that `self.ripple` < 0.02,
            # while `lambda_DB` < 21 requires `self.ripple` > 0.089.
            # This is included for completeness...
            Kf = 1.8445

        # Normalize transition bandwidth to Nyquist frequency
        # to make contact with Kaiser & Reed paper
        delta = self.trans / (self.Fs / 2.)

        # Eq. (13) of Kaiser & Reed
        Np = int(np.ceil((Kf / (2 * delta)) + 0.75))

        # Now, the returned value for the number of taps is
        # *guaranteed* to be an *odd* integer. As discussed
        # in the paragraph following Eq. (6) in Kaiser & Reed,
        # an odd number of taps prevents the filter from
        # introducing phase shifts
        return (2 * Np) + 1

    def getFilterCoefficients(self):
        'Get coefficients via `:py:function:`firwin <scipy.signal.firwin>`.'
        return signal.firwin(self.getNumTaps(), cutoff=self.cutoff,
                             window=('kaiser', kaiser_beta(self.ripple_dB)),
                             pass_zero=self.pass_zero, nyq=(0.5 * self.Fs))

    def getFrequencyResponse(self, freqs_or_N=None, plot=False):
        '''Get frequency response of filter.

        Parameters:
        -----------
        freqs_or_N: {None, int, array_like}, optional
            If None (default), then compute at 512 frequencies
            equally spaced from DC to the Nyquist frequency.

            If a single integer, the compute at that many frequencies,
            again, equally spaced from DC to the Nyquist frequency.

            If an array_like, compute the response at the frequencies given,
            where the frequencies are given in the same units as `self.Fs`

        plot - bool
            If true, plot the frequency response curve.

        Returns:
        -------
        (f, h): tuple

        f - array_like, (`M`,)
            The frequencies at which `h` was computed.
            [f] = [self.Fs]

        h - array_like, (`M`,)
            The frequency response. This is generally a complex value,
            so look at `np.abs(h)` to see the magnitude of the
            transfer function.
            [h] = unitless

        '''
        # Unit conversion factor from `self.Fs` to radians / sample
        Fs_to_wnorm = np.pi / (0.5 * self.Fs)

        if isinstance(freqs_or_N, (list, tuple, np.ndarray)):
            # Copy array and cast to numpy array of floats
            worN = np.asarray(freqs_or_N).astype('float')

            # Convert frequencies from units of [`self.Fs`] to radians / sample
            worN *= Fs_to_wnorm
        else:
            worN = freqs_or_N

        w, h = signal.freqz(self.b, a=1, worN=worN)

        # Convert `w` to units of ['self.Fs']
        f = w / Fs_to_wnorm

        if plot:
            plt.figure()
            plt.semilogy(f, np.abs(h))
            plt.xlabel('f')
            plt.ylabel('|h|')
            plt.show()

        return f, h

    def getValidSlice(self):
        '''Get slice corresponding to `valid` data points in filtered signal,
        for which boundary effects of the filter are not visible.

        '''
        # The filter has (2 * `Np`) + 1 taps, so the first `Np` and last `Np`
        # points in the filtered signal display boundary effects
        num_taps = self.b.size
        Np = (num_taps - 1) / 2

        return slice(Np, -Np, None)

    def applyTo(self, y, check_fftconvolve=True):
        '''Apply filter to signal `y`, where `y` was sampled at `self.Fs`.

        An FIR filter is applied via convolution of the signal `y`
        with the filter coefficients.

        Parameters:
        -----------
        y - array_like, (`N`,)
            The input signal to be filtered. The signal should
            be sampled at rate `self.Fs`.
            [y] = arbitrary units

        check_fftconvolve - bool
            If True, check to see if convolution should be computed
            via the FFT. While `fftconvolve(...)` can be faster than
            direct convolution via `convolve(...)`, it also requires
            more memory overhead. If `fftconvolve(...)` raises
            MemoryError, one can set check_fftconvolve to False
            to perform the direct (slower) convolution, potentially
            avoiding a MemoryError.

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
        # However, `fftconvolve` can incur large memory overhead,
        # potentially resulting in a `MemoryError`. A dirty hack
        # is just to resort to the (slower) direct convolution...
        if check_fftconvolve and self.b.size >= (4 * np.log2(y.size)):
            yfilt = signal.fftconvolve(y, self.b, mode='same')
        else:
            yfilt = np.convolve(y, self.b, mode='same')

        # Return filtered signal with same type as input signal
        return yfilt.astype(dtype)

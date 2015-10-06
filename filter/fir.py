'Class and functions for creating and using FIR filters.'


import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt


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

                                ripple < 0.02

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
        if ripple < 0.02:
            self.ripple = ripple
        else:
            raise ValueError('The NER formalism requires `ripple` < 0.02')

        # Is the condition on transition bandwidth satisfied?
        intervals = np.diff(np.concatenate((
            np.array([0]), np.asarray(cutoff), np.array([Fs / 2.]))))

        if np.min(intervals) > (0.5 * trans):
            self.cutoff = np.asarray(cutoff)
            self.trans = trans
        else:
            raise ValueError(
                'For the desired passband(s), the NER formalism requires '
                'that `trans` < %f' % (2 * np.min(intervals)))

    def getNumTaps(self):
        '''Get number of "taps" for desired falloff and accuracy.'''
        # Ensure it is *odd* so that we get zero phase delay!!!
        pass

    def filter(self):
        # Convolve with given signal, optimized convolution method
        pass

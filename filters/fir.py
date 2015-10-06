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
        '''Get number of "taps" for desired falloff and accuracy.
        The relevant formulas are given in Eq. (12) and (13) of
        Kaiser & Reed, RSI 48, 1447 (1977).

        '''
        # Eq. (11) of Kaiser & Reed
        lambda_dB = -20 * np.log10(self.ripple)

        # Eq. (12) of Kaiser & Reed
        if lambda_dB > 21:
            Kf = 0.13927 * (lambda_dB - 7.95)
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
        Np = int((Kf / (2 * delta)) + 0.75)

        # Further, as is discussed in the paragraph following
        # Eq. (6) in Kaiser & Reed, we want to have an *odd*
        # number of taps (e.g. odd `Np`) such that the filter
        # introduces *no* phase delay
        if Np % 2 == 0:
            Np += 1

        return Np

    def filter(self):
        # Convolve with given signal, optimized convolution method
        pass

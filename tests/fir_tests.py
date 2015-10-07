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


def test_NER_getNumTaps():
    # Kaiser & Reed, RSI 48, 1447 (1977), Fig. 3 states that for
    # beta = 0.3, delta = 0.2, and epsilon = 0.02, they obtain
    # Np = 10 pairs of terms, for a total of 2 * Np + 1 = 21 taps
    filt = NER(0.02, [0.3], 0.2, 2.)
    tools.assert_equal(filt.getNumTaps(), 21)

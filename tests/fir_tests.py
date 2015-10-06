from nose import tools
import numpy as np
from filters.fir import NER


@tools.raises(ValueError)
def test_NER__init__ripple():
    # This ripple size is too small
    NER(0.1, [2., 3.], 1., 5.)


@tools.raises(ValueError)
def test_NER__init__trans():
    # This transition size is too small for the given passband
    NER(0.01, [2.25, 2.75], 1., 5., pass_zero=False)

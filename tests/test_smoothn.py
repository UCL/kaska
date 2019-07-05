'''
Test the smoothn functionality, including the examples from the Matlab original

Author: Timothy Spain, t.spain@ucl.ac.uk
Date: 5th July 2019

'''

import pytest
import numpy as np

import sys
sys.path.append("../kaska")

import smoothn

def test_robust1d():
    x = np.linspace(0, 100, 256)
    y_base = np.cos(x/10) + (x/50.)**2
    np.random.seed(3141592653)
    y_noise = y_base + np.random.randn(len(x))/10
    y = np.copy(y_noise) ; y[70:85:5] = (5.5, 5, 6)
    (z, s, flag, wtot) = smoothn.smoothn(y)
    (zr, sr, flag, wtot) = smoothn.smoothn(y, isrobust=True)
    res = z - y_base
    resr = zr - y_base
    
    unrobust_target = 0.7503929639274534
    robust_target = 0.18943656067148762
    
    prec = 1e-12
    
    assert np.abs(np.max(np.abs(res)) -  unrobust_target) < prec
    assert np.abs(np.max(np.abs(resr)) -  robust_target) < prec

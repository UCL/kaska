'''
Test the smoothn functionality, including the examples from the Matlab original

Author: Timothy Spain, t.spain@ucl.ac.uk
Date: 5th July 2019

'''

import pytest
import numpy as np
import numpy.ma as ma

import sys

from .. import smoothn

# Make a test from the first example of the smoothn code
def test_robust1d():
    # The target resdiuals for this test
    unrobust_target = 0.7503929639274534
    robust_target = 0.18943656067148762

    prec = 1e-5

    general_robustness_test(isrobust=None,
                            target=unrobust_target,
                            s=None, prec=prec)
    general_robustness_test(isrobust=True,
                            target=robust_target,
                            s=None, prec=prec)

def general_robustness_test(isrobust=None, target=0.0, s=None, prec=1e-12):
    x = np.linspace(0, 100, 256)
    y_base = np.cos(x/10) + (x/50.)**2
    np.random.seed(3141592653)
    y_noise = y_base + np.random.randn(len(x))/10
    y = np.copy(y_noise) ; y[70:85:5] = (5.5, 5, 6)
    if (isrobust is None):
        (z, s, flag, wtot) = smoothn.smoothn(y, s=s)
    else:
        (z, s, flag, wtot) = smoothn.smoothn(y, s=s, isrobust=isrobust)

    res = z - y_base

    assert np.abs(np.max(np.abs(res)) - target) < prec
    
    
def test_fixed_order():
    # The target resdiuals for this test
    unrobust_target = 0.7503929639274534
    robust_target = 0.17266123460914873
    # The fixed smoothing order that should achieve those residuals
    unrobust_order = 56.93236088601813
    robust_order = 27.61712142163073
    prec = 1e-5

    general_robustness_test(isrobust=None, 
                            target=unrobust_target,
                            s=unrobust_order, prec=prec)
    general_robustness_test(isrobust=True,
                            target=robust_target,
                            s=robust_order, prec=prec)


def test_sd_weights():
    x = np.linspace(0, 100, 256)
    y_base = np.cos(x/10) + (x/50.)**2

    sd = np.abs(np.random.randn(len(x)))
    noise = np.random.randn(len(x))/10.
    sd_noise = sd * noise

    y_noise = y_base + sd_noise

    w = sd**(-2)

    (z_sd, s, flag, wtot) = smoothn.smoothn(y_noise, sd=sd)
    (z_w, s, flag, wtot) = smoothn.smoothn(y_noise, W=w)

    # should be identical
    prec = 2e-15
    assert(np.sqrt(np.mean((z_sd - z_w)**2)) < prec)

def test_masked_array():
    (d, w) = txy_data()
    masky = ma.array(d, mask = (w == 0.))
    (s, ess, flag, wtot) = smoothn.smoothn(masky, W = w, axis=0, isrobust=True)

    target_maximum_residuals = np.array([[0.19063412, 1.63927078, 3.22016034],
                                         [2.86830946, 3.3060494 , 5.26835424],
                                         [4.74814422, 6.95324161, 4.21698349]])

    target_rms = np.array([[0.04373571, 0.31264841, 0.61608984],
                           [1.39772024, 0.99905187, 1.30683329],
                           [3.05899054, 2.85209818, 1.88332256]])

    residual_rms_assessment(s, d, target_maximum_residuals, target_rms)


# A test emulating the way smoothn is used in KaSKA
def test_time_txy():
    (d, w) = txy_data()
    (s, ess, flag, wtot) = smoothn.smoothn(d, W=w, axis=0, isrobust=True)
    
    target_maximum_residuals = np.array([[0.18979927, 1.63548774, 3.21602835],
                                         [2.87126492, 3.30231171, 5.26648978],
                                         [4.69832211, 6.88804564, 4.19266901]])

    target_rms = np.array([[0.04338182, 0.31158228, 0.61451283],
                           [1.39932148, 1.00197386, 1.306045  ],
                           [3.00532792, 2.80878405, 1.85278847]])
    
    residual_rms_assessment(s, d, target_maximum_residuals, target_rms)

def test_cauchy():
    (d, w) = txy_data()
    (s, ess, flag, wtot) = smoothn.smoothn(d, W = w, axis=0, isrobust=True, weightstr='cauchy')

    target_maximum_residuals = np.array([[0.20149125, 1.72772566, 3.42323339],
                                         [2.78516476, 3.45869778, 5.50720599],
                                         [4.28855901, 6.58983133, 4.37961953]])

    target_rms = np.array([[0.04707677, 0.33686247, 0.66873105],
                           [1.37314063, 1.03404982, 1.36475258],
                           [2.57311525, 2.46561701, 1.80441455]])

    residual_rms_assessment(s, d, target_maximum_residuals, target_rms)

def test_talworth():
    (d, w) = txy_data()
    (s, ess, flag, wtot) = smoothn.smoothn(d, W = w, axis=0, isrobust=True, weightstr='talworth')

    target_maximum_residuals = np.array([[0.14892761, 1.347204  , 2.69440801],
                                         [3.3310566 , 3.13332464, 5.24182578],
                                         [4.31627292, 7.45370604, 3.33895975]])

    target_rms = np.array([[0.02994288, 0.23521695, 0.47043391],
                           [1.44108898, 1.08753645, 1.29324771],
                           [3.1671869 , 3.03369429, 1.83437903]])

    residual_rms_assessment(s, d, target_maximum_residuals, target_rms)

def residual_rms_assessment(s, d, target_max_resid, target_rms):
    res = s - np.expand_dims(d[:, 0, :], axis=1)

    maximum_residuals = np.max(np.abs(res), axis=0)
    rms = np.sqrt(np.mean(np.square(res), axis=0))

    fudge = 2e-6
    resid_avec_fudge = target_max_resid + fudge
    rms_avec_fudge = target_rms + fudge
    
    assert(np.all(maximum_residuals <= resid_avec_fudge))
    assert(np.all(rms <= rms_avec_fudge))

def txy_data():
    
    np.random.seed(2718281828)
    
    # Creates a synthetic dataset with dimensions time, x, y position
    # Also weighting array
    nx = 3 # underlying ground truth, noisy data, cloudy (spurious and missing data) 
    ny = 3 # tropics, temperates, arctic
    nt = 360 # days in a year
    
    # Daily samples of air temperature, modelled as sinusoids
    amplitudes = np.array([5., 15, 30])
    offsets = np.array([300., 280., 268.])
    hottest_day = np.array([172., 207., 207.])
    
    t = np.arange(nt)
    data_arr = np.zeros([nt, nx, ny])
    weight_arr = np.ones([nt, nx, ny])
   
    for i in range(nx):
        for j in range(ny):
            data_arr[:, i, j] = amplitudes[j] * np.cos(np.radians(t - hottest_day[j])) + offsets[j]
            # add noise
            if i > 0: 
                data_arr[:, i, j] *= 1+data_noise(nt, 0.025)
            # Add bad data
            if i == 2:
                (transmission, weight) = cloudy_data(nt, 1)
                data_arr[:, 2, j] *= transmission
                weight_arr[:, 2, j] = weight

    return (data_arr, weight_arr)

# Something not very physical creating poor or missing data
def cloudy_data(nt, mist_threshold):
    
    optical_depth = np.random.randn(nt)
    optical_depth = np.where(optical_depth < mist_threshold, 0., optical_depth - mist_threshold)
    transmission = np.exp(-optical_depth)
    weight = np.where(optical_depth < 1, transmission**2, 0.)
    return (transmission, weight)

# a multiplicative noise value
def data_noise(nt, scale):
    return np.random.randn(nt) * scale

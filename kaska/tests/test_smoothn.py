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

@pytest.mark.parmetrize("is_robust, target, s, prec", [[False, 0.7503929639274534, None, 1e-5], # 1d unrobust
                                                       [True, 0.18943656067148762, None, 1e-5], # 1d robust
                                                       [False, 0.7503929639274534, 56.93236088601813, 1e-5], # fixed order - unrobust target/order
                                                       [True, 0.17266123460914873, 27.61712142163073, 1e-5], # fixed order - robust target/order
])
def test_general_robustness(is_robust, target, s, prec=1e-12):
    x = np.linspace(0, 100, 256)
    y_base = np.cos(x/10) + (x/50.)**2
    np.random.seed(3141592653)
    y_noise = y_base + np.random.randn(len(x))/10
    y = np.copy(y_noise)
    y[70:85:5] = (5.5, 5, 6)
    (z, s, flag, wtot) = smoothn.smoothn(y, s=s, isrobust=is_robust)

    res = z - y_base

    assert np.abs(np.max(np.abs(res)) - target) < prec


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
    rms = np.sqrt(np.mean((z_sd - z_w)**2))
    assert rms < prec


# A test emulating the way smoothn is used in KaSKA
@pytest.mark.parmetrize("weights, target_max_resid, target_rms",[['masked',
                                                                  np.array([[0.19063412, 1.63927078, 3.22016034],
                                                                            [2.86830946, 3.3060494 , 5.26835424],
                                                                            [4.74814422, 6.95324161, 4.21698349]])
                                                                  np.array([[0.04373571, 0.31264841, 0.61608984],
                                                                            [1.39772024, 0.99905187, 1.30683329],
                                                                            [3.05899054, 2.85209818, 1.88332256]])],
                                                                 ['bisquare', # time_txy
                                                                  np.array([[0.18979927, 1.63548774, 3.21602835],
                                                                            [2.87126492, 3.30231171, 5.26648978],
                                                                            [4.69832211, 6.88804564, 4.19266901]]),
                                                                  np.array([[0.04338182, 0.31158228, 0.61451283],
                                                                            [1.39932148, 1.00197386, 1.306045  ],
                                                                            [3.00532792, 2.80878405, 1.85278847]])],
                                                                 ['cauchy',
                                                                  np.array([[0.20149125, 1.72772566, 3.42323339],
                                                                            [2.78516476, 3.45869778, 5.50720599],
                                                                            [4.28855901, 6.58983133, 4.37961953]])
                                                                  np.array([[0.04707677, 0.33686247, 0.66873105],
                                                                            [1.37314063, 1.03404982, 1.36475258],
                                                                            [2.57311525, 2.46561701, 1.80441455]])],
                                                                 ['talworth',
                                                                  np.array([[0.14892761, 1.347204  , 2.69440801],
                                                                            [3.3310566 , 3.13332464, 5.24182578],
                                                                            [4.31627292, 7.45370604, 3.33895975]]),
                                                                  np.array([[0.02994288, 0.23521695, 0.47043391],
                                                                            [1.44108898, 1.08753645, 1.29324771],
                                                                            [3.1671869 , 3.03369429, 1.83437903]])],
])
def test_residual_rms_assessment(weights, target_max_resid, target_rms):

    (d, w) = txy_data()
    if weights == 'masked':
        masky = ma.array(d, mask = (w == 0.))
    else:
        masky = d
    (s, ess, flag, wtot) = smoothn.smoothn(masky, W=w, axis=0, isrobust=True, weightstr=weights)


    res = s - np.expand_dims(d[:, 0, :], axis=1)

    maximum_residuals = np.max(np.abs(res), axis=0)
    rms = np.sqrt(np.mean(np.square(res), axis=0))

    fudge = 5e-6
    resid_diff = np.abs(maximum_residuals - target_max_resid)
    rms_diff = np.abs(rms - target_rms)
    
    assert(np.all(resid_diff <= fudge))
    assert(np.all(rms_diff <= fudge))

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

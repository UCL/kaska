'''
Test the smoothn functionality, including the examples from the Matlab original

Author: Timothy Spain, t.spain@ucl.ac.uk
Date: 5th July 2019

'''

import pytest
import numpy as np

import sys

from .. import smoothn

# Make a test from the first example of the smoothn code
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
    
    #prec = 1e-12
    prec = 1e-5
    
    assert np.abs(np.max(np.abs(res)) -  unrobust_target) < prec
    assert np.abs(np.max(np.abs(resr)) -  robust_target) < prec
    
# A test emulating the way smoothn is used in KaSKA
def test_time_txy():
    (d, w) = txy_data()
    (s, ess, flag, wtot) = smoothn.smoothn(d, W = w, axis=0, isrobust=True)
    res = s - np.expand_dims(d[:, 0, :], axis=1)
    
    delta = 1e-8 # Difference between the calculated values and the copy-pasted values
    
    target_maximum_residuals = np.array([[0.18979927, 1.63548774, 3.21602835],
                                         [2.87126492, 3.30231171, 5.26648978],
                                         [4.69832211, 6.88804564, 4.19266901]]) + delta

    target_rms = np.array([[0.04338182, 0.31158228, 0.61451283],
                           [1.39932148, 1.00197386, 1.306045  ],
                           [3.00532792, 2.80878405, 1.85278847]]) + delta
    
    maximum_residuals = np.max(np.abs(res), axis=0)

    rms = np.sqrt(np.mean(np.square(res), axis=0))

    fudge = 1e-6
    
    assert(np.all(maximum_residuals <= target_maximum_residuals + fudge))
    assert(np.all(rms <= target_rms + fudge))

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

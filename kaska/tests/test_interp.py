'''
Test the interp functionality

'''

import pytest
import numpy as np

import sys

from .. import interp_fix

def test_1d():
    newx = np.arange(200)
    oldx = np.array(sorted(np.random.choice(np.arange(200), 100, replace=False)))
    oldy = np.random.rand(100)
    np_ret    = np.interp(newx, oldx, oldy)
    numba_ret = interp_fix.interp1d(newx, oldx, oldy)
    assert np.allclose(np_ret, numba_ret)

def test_2d():
    newx = np.arange(200)
    oldx = np.array(sorted(np.random.choice(np.arange(200), 100, replace=False)))
    oldy = np.random.rand(100,5)
    numba_ret = interp_fix.interp1d(newx, oldx, oldy)
    for i in range(5):
        np_ret    = np.interp(newx, oldx, oldy[:,i])
        assert np.allclose(np_ret, numba_ret[:,i])

def test_3d():
    newx = np.arange(200)
    oldx = np.array(sorted(np.random.choice(np.arange(200), 100, replace=False)))
    oldy = np.random.rand(100, 5, 10)
    numba_ret = interp_fix.interp1d(newx, oldx, oldy)
    
    close_so_far = True
    for i in range(5):
        for j in range(10):
            np_ret    = np.interp(newx, oldx, oldy[:,i, j])
            close_so_far &= np.allclose(np_ret, numba_ret[:,i, j])
            
    assert close_so_far

# def testgap():
#     newx = np.arange(200)
#     oldx = np.array(sorted(np.random.choice(np.arange(200), 100, replace=False)))
#     oldy = oldy = np.random.rand(100, 5, 10)
#     for i in range(5):
#         for j in range(10):
#             bad_pix = np.random.choice(range(100), 100)
#             gap_oldy = oldy[:,i, j].copy()
#             gap_oldy[bad_pix] = np.nan
#             numba_ret = interp_fix.interp1d(newx, oldx, gap_oldy)
#             np_ret = np.interp(newx, oldx, gap_oldy)
#             diff    = numba_ret - np_ret
#             if not np.nansum(diff)<1e-10:
#                 raise
#     print('Same result achieved and filled gaps')
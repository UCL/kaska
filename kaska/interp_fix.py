import numpy as np
from numba import jit

'''
This is a linear interpolation over 1 axis
using numpy.interp function. It uses numba 
to speed up filling gaps in the for loop.

Feng Yin
Department of Geography, UCL
ucfafyi@ucl.ac.uk
LICENSE: GNU GENERAL PUBLIC LICENSE V3 
'''

@jit(nopython=True)
def interp1d(newx, oldx, oldy):
    if oldx.shape[0] != oldy.shape[0]:
        raise ValueError('oldx and oldy must have the same shape in first axis.')
    oldy_shape = oldy.shape[1:]
    new_shape  = newx.shape[0]
    oldy_t = oldy.reshape(oldx.shape[0], -1).T
    pix_num = oldy_t.shape[0]
    newy = np.zeros((pix_num, new_shape))*np.nan
    for i in range(pix_num):
        y    = oldy_t[i]
        mask = ~np.isnan(y)
        if mask.sum()>0:
            newy[i] = np.interp(newx, oldx[mask], y[mask])
    newy_T = newy.transpose(1,0).reshape((new_shape,) + oldy_shape)
    return newy_T

def test1d():
    newx = np.arange(200)
    oldx = np.array(sorted(np.random.choice(np.arange(200), 100, replace=False)))
    oldy = oldy = np.random.rand(100)
    np_ret    = np.interp(newx, oldx, oldy)
    numba_ret = interp1d(newx, oldx, oldy)
    if not np.allclose(np_ret, numba_ret):
        raise
    print('Same result achieved')

def test2d():
    newx = np.arange(200)
    oldx = np.array(sorted(np.random.choice(np.arange(200), 100, replace=False)))
    oldy = oldy = np.random.rand(100,5)
    numba_ret = interp1d(newx, oldx, oldy)
    for i in range(5):
        np_ret    = np.interp(newx, oldx, oldy[:,i])
        if not np.allclose(np_ret, numba_ret[:,i]):
            raise
    print('Same result achieved')

def test3d():
    newx = np.arange(200)
    oldx = np.array(sorted(np.random.choice(np.arange(200), 100, replace=False)))
    oldy = oldy = np.random.rand(100, 5, 10)
    numba_ret = interp1d(newx, oldx, oldy)
    for i in range(5):
        for j in range(10):
            np_ret    = np.interp(newx, oldx, oldy[:,i, j])
            if not np.allclose(np_ret, numba_ret[:,i, j]):
                raise
    print('Same result achieved')

def testgap():
    newx = np.arange(200)
    oldx = np.array(sorted(np.random.choice(np.arange(200), 100, replace=False)))
    oldy = oldy = np.random.rand(100, 5, 10)
    for i in range(5):
        for j in range(10):
            bad_pix = np.random.choice(range(100), 100)
            gap_oldy = oldy[:,i, j].copy()
            gap_oldy[bad_pix] = np.nan
            numba_ret = interp1d(newx, oldx, gap_oldy)
            np_ret = np.interp(newx, oldx, gap_oldy)
            diff    = numba_ret - np_ret
            if not np.nansum(diff)<1e-10:
                raise
    print('Same result achieved and filled gaps')
if __name__ == '__main__':
    test1d()
    test2d() 
    test3d()
    testgap()
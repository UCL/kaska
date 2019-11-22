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
    newy = np.full((pix_num, new_shape), np.nan)
    for i in range(pix_num):
        y    = oldy_t[i]
        mask = ~np.isnan(y)
        if mask.any():
            newy[i] = np.interp(newx, oldx[mask], y[mask])
    newy_T = newy.transpose(1,0).reshape((new_shape,) + oldy_shape)
    return newy_T

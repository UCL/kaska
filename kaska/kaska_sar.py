# -*- coding: utf-8 -*-

"""Main module."""
import logging
import time  # Just for timekeeping
import datetime as dt
import numpy as np

from scipy.interpolate import interp1d

import scipy.optimize

from .s1_observations import Sentinel1Observations

from .utils import define_temporal_grid, save_output_parameters

from .watercloudmodel import cost, cost_jac, cost_hess


LOG = logging.getLogger(__name__)


def sar_inversion(s1_obs, s2_data):
    """SAR inversion"""
    # Move everything to DoY to simplify interpolation
    s2_doys = [int(dt.datetime.strftime(x, "%j"))
               for x in s2_data.f.temporal_grid]
    s1_doys = [int(dt.datetime.strftime(x, "%j"))
               for x in s1_obs.dates.keys()]
    n_sar_obs = len(s1_doys)
    s1_temporal_grid = sorted(s1_obs.dates.keys())

    # Interpolate S2 retrievals to S1 time grid
    func = interp1d(s2_doys, s2_data.f.slai, axis=0, bounds_error=False)
    lai_s1 = func(s1_doys)
    func = interp1d(s2_doys, s2_data.f.scab, axis=0, bounds_error=False)
    cab_s1 = func(s1_doys)
    func = interp1d(s2_doys, s2_data.f.scbrown, axis=0, bounds_error=False)
    # cbrown_s1 = f(s1_doys)
    # Read in S1 data
    s1_backscatter = s1_obs.read_time_series(s1_temporal_grid)

    # Wrap cost functions. NB The cab argument isn't used in these functions.
    def cost_nolai(my_xx, svh, svv, lai, cab, theta):
        n_obs = len(svh)
        return cost(np.concatenate([my_xx[:6], lai, lai, my_xx[-n_obs:]]),
                    svh, svv, theta)

    def cost_nolai_jac(my_xx, svh, svv, lai, cab, theta):
        n_obs = len(svh)
        return cost_jac(np.concatenate([my_xx[:6], lai, lai, my_xx[-n_obs:]]),
                        svh, svv, theta)

    def cost_nolai_hess(my_xx, svh, svv, lai, cab, theta):
        n_obs = len(svh)
        return cost_hess(np.concatenate([my_xx[:6], lai, lai, my_xx[-n_obs:]]),
                         svh, svv, theta)

    avv, bvv, cvv = -12, 0.05, 0.1   # Magic numbers again
    avh, bvh, cvh = -14, 0.01, 0.1
    sigma_soil0 = np.zeros(n_sar_obs)*0.2  # Say
    x0_all = \
        np.r_[avv, bvv, cvv, avh, bvh, cvh, sigma_soil0]  # ,V1,V2,sigma_soil]
    n_t, n_y, n_x = lai_s1.shape
    avv_out = np.zeros((n_y, n_x))
    bvv_out = np.zeros((n_y, n_x))
    cvv_out = np.zeros((n_y, n_x))
    avh_out = np.zeros((n_y, n_x))
    bvh_out = np.zeros((n_y, n_x))
    cvh_out = np.zeros((n_y, n_x))
    cost_f = np.zeros((n_y, n_x))
    sigma_soil_out = np.zeros((n_t, n_y, n_x))
    tic = time.time()
    n_pxls = 0
    # bounds = [
    #    [-40, -5],
    #    [1e-4, 1],
    #    [-40, -1],
    #    [-40, -5],
    #    [1e-4, 1],
    #    [-40, -1],
    #    *([[0.01, 1]]*n_sar_obs)
    # ]
    x_0 = x0_all
    for (row, col) in np.ndindex(*lai_s1[0].shape):
        lai = lai_s1[:, row, col]
        if lai.max() < 2.5:
            cost_f[row, col] = -900.  # No dynamics
            continue
        # Select one pixel
        svv = 10*np.log10(s1_backscatter.VV[:, row, col])
        svh = 10*np.log10(s1_backscatter.VH[:, row, col])
        theta = s1_backscatter.theta[:, row, col]
        sigma_soil0 = np.ones_like(svv)*0.2

        cab = cab_s1[:, row, col]
        # Might be worth defining Cxx from LAI=0 average and
        #  Axx when LAI=LAI.max()
        x_0[2] = svv[lai < 0.3].mean()
        x_0[5] = svh[lai < 0.3].mean()
        x_0[0] = svv[lai > (0.9*lai.max())].mean()
        x_0[3] = svh[lai > (0.9*lai.max())].mean()
        retval = scipy.optimize.minimize(cost_nolai, x_0,
                                         args=(svh, svh, lai, cab, theta),
                                         jac=cost_nolai_jac,
                                         hess=cost_nolai_hess,
                                         method="Newton-CG")
        avv_out[row, col] = retval.x[0]
        bvv_out[row, col] = retval.x[1]
        cvv_out[row, col] = retval.x[2]
        avh_out[row, col] = retval.x[3]
        bvh_out[row, col] = retval.x[4]
        cvh_out[row, col] = retval.x[5]
        cost_f[row, col] = retval.fun
        sigma_soil_out[:, row, col] = retval.x[6:]
        if retval.fun < 1e5:
            print(f"Good good...{row:d}, {col:d}")
            x_0 = retval.x
        else:
            # Rubbish inversion, do not use parameters!
            x_0 = x0_all

        n_pxls += 1
        if n_pxls % 1000 == 0:
            n_pxls = 0
            # LOG.info(f"Done 100 pixels in {(time.time()-tic):g}")
            LOG.info("Done 100 pixels in %g", time.time()-tic)
            tic = time.time()
    return s1_temporal_grid, sigma_soil_out


def save_s1_output(output_folder, obs, sar_data, time_grid, chunk):
    """Save the S1 output."""
    save_output_parameters(time_grid, obs, output_folder, ["sigma"],
                           [sar_data], output_format="GTiff",
                           chunk=chunk, fname_pattern="s1")


if __name__ == "__main__":
    STATE_MASK = "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/ESU.tif"
    NC_FILE = "/data/selene/ucfajlg/ELBARA_LMU/mirror_ftp/141.84.52.201" + \
              "/S1/S1_LMU_site_2017_new.nc"
    START_DATE = dt.datetime(2017, 5, 1)
    END_DATE = dt.datetime(2017, 9, 1)
    TEMPORAL_GRID_SPACE = 5
    TEMPORAL_GRID = define_temporal_grid(START_DATE, END_DATE,
                                         TEMPORAL_GRID_SPACE)
    # Define S1 observations
    S1_OBS = Sentinel1Observations(NC_FILE,
                                   STATE_MASK,
                                   time_grid=TEMPORAL_GRID)

    # Read in smoothed S2 retrievals
    S2_DATA = np.load("temporary_dump.npz")
    sar_inversion(S1_OBS, S2_DATA)

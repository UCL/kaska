# -*- coding: utf-8 -*-

"""Main module."""
import logging

import datetime as dt
import numpy as np

from scipy.interpolate import interp1d

from NNParameterInversion import NNParameterInversion

from s2_observations import Sentinel2Observations

from s1_observations import Sentinel1Observations

from smoothn import smoothn

from utils import define_temporal_grid

from watercloudmodel import cost, cost_jac, cost_hess

import scipy.optimize

LOG = logging.getLogger(__name__ + ".KaSKA")
LOG.setLevel(logging.DEBUG)
if not LOG.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - ' +
                                  '%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    LOG.addHandler(ch)
LOG.propagate = False

if __name__ == "__main__":
    state_mask = "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/ESU.tif"
    nc_file = "/data/selene/ucfajlg/ELBARA_LMU/mirror_ftp/141.84.52.201/S1/S1_LMU_site_2017_new.nc"
    start_date = dt.datetime(2017, 5, 1)
    end_date = dt.datetime(2017, 9, 1)
    temporal_grid_space = 5
    temporal_grid = define_temporal_grid(start_date, end_date,
                                        temporal_grid_space)
    # Define S1 observations
    s1_obs = Sentinel1Observations(nc_file,
                state_mask,
                time_grid=temporal_grid)
    
    # Read in smoothed S2 retrievals
    s2_data = np.load("temporary_dump.npz")
    # Move everything to DoY to simplify interpolation
    s2_doys = [int(dt.datetime.strftime(x, "%j"))
               for x in s2_data.f.temporal_grid]
    s1_doys = [int(dt.datetime.strftime(x, "%j"))
               for x in s1_obs.dates.keys()]
    n_sar_obs = len(s1_doys)
    # Interpolate S2 retrievals to S1 time grid
    f = interp1d(s2_doys, s2_data.f.slai, axis=0, bounds_error=False)
    lai_s1 = f(s1_doys)
    f = interp1d(s2_doys, s2_data.f.scab, axis=0, bounds_error=False)
    cab_s1 = f(s1_doys)
    f = interp1d(s2_doys, s2_data.f.scbrown, axis=0, bounds_error=False)
    cbrown_s1 = f(s1_doys)
    # Read in S1 data
    S1_backscatter=s1_obs.read_time_series(temporal_grid)  


    # Wrap cost functions
    def cost_nolai(xx, svh, svv, lai, cab, theta):
        n_obs = len(svh)
        return cost(np.concatenate([xx[:6], lai,lai*cab, xx[-n_obs:]]), svh,svv,theta)

    def cost_nolai_jac(xx, svh, svv, lai, cab, theta):
        n_obs = len(svh)
        return cost_jac(np.concatenate([xx[:6], lai,lai*cab, xx[-n_obs:]]), svh,svv,theta)

    def cost_nolai_hess(xx, svh, svv, lai, cab, theta):
        n_obs = len(svh)
        return cost_hess(np.concatenate([xx[:6], lai, lai*cab, xx[-n_obs:]]), svh,svv,theta)

    Avv, Bvv, Cvv, V1, V2, sigma_soil = -12,  0.05, 0.1, np.ones(2)*4, np.ones(2)*4, np.ones(2)*0.1
    Avh, Bvh, Cvh = -14, 0.01, 0.1
    
    x0 = np.r_[Avv, Bvv, Cvv, Avh, Bvh, Cvh, sigma_soil0]#, V1, V2, sigma_soil]
    
    for (row, col) in np.npindex(lai):
        # Select one pixel
        svv = S1_backscatter.VV[:, row, col]
        svh = S1_backscatter.VH[:, row, col]
        theta = S1_backscatter.theta[:, row, col]
        sigma_soil0 = np.ones_like(svv)*0.2 # 
        lai = lai_s1[:, row, col]
        cab = cab_s1[:, row, col]

        retval = scipy.optimize.minimize(cost_nolai, x0, args=(svh, svh, lai, cab, theta), 
                                 jac=cost_nolai_jac, hess=cost_nolai_hess,
                                method="Newton-CG")
        print(cost_nolai(retval.x, svh, svv, lai, cab, theta))    
        x0 = retval.x


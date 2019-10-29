# -*- coding: utf-8 -*-

"""Sentinel 1 inversion class"""
import logging

from pathlib import Path

import datetime as dt
import numpy as np

from scipy.interpolate import interp1d

from .s1_observations import Sentinel1Observations

from .utils import define_temporal_grid

from .watercloudmodel import cost, cost_jac, cost_hess

from .kaska import Sentinel2Data

import scipy.optimize


LOG = logging.getLogger(__name__)


class KaSKASAR(object):
    """A class to process Sentinel 1 SAR data using S2 data as 
    an input"""

    def __init__(
        self, time_grid, state_mask, s1_observations, s2_data, chunk=None
    ):
        """Set up processing paths and options for s1 observations
        
        Parameters
        ----------
        time_grid : iter
            A temporal grid. E.g. a list of datetimes
        state_mask : str
            The state mask file. Must be readable by GDAL and be georeferenced
        s1_observations : s1_observations
            S1 observations type
        s2_data : s2_observations
            S2 observations type
        chunk : inteter, optional
            The chunk if processing by chunks. Used for file outputs, by default None
        """
        self.time_grid = time_grid
        self.s1_observations = s1_observations
        self.state_mask = state_mask
        self.output_folder = Path(output_folder) / ("S1_outputs")
        self.chunk = chunk
        self.s2_data = self._resampling_times(s2_data)

    def _resampling_times(self, s2_data):
        """Resample S2 smoothed output to match S1 observations
        times"""
        # Move everything to DoY to simplify interpolation
        s2_doys = [
            int(dt.datetime.strftime(x, "%j")) for x in s2_data.temporal_grid
        ]
        s1_doys = [
            int(dt.datetime.strftime(x, "%j"))
            for x in self.s1_observations.dates.keys()
        ]
        n_sar_obs = len(s1_doys)
        # Interpolate S2 retrievals to S1 time grid
        f = interp1d(s2_doys, s2_data.slai, axis=0, bounds_error=False)
        lai_s1 = f(s1_doys)
        f = interp1d(s2_doys, s2_data.scab, axis=0, bounds_error=False)
        cab_s1 = f(s1_doys)
        f = interp1d(s2_doys, s2_data.scbrown, axis=0, bounds_error=False)
        cbrown_s1 = f(s1_doys)
        return Sentinel2Data(lai_s1, cab_s1, cbrown_s1)

    def sentinel1_inversion(self):
        nt, ny, nx = self.s2_data.slai.shape
        outputs = {
            param: np.zeros((nt, ny, nx))
            for param in ["Avv", "Bvv", "Cvv" "Avh", "Bvh", "Cvh"]
        }
        bounds = [
            [-40, -5],
            [1e-4, 1],
            [-40, -1],
            [-40, -5],
            [1e-4, 1],
            [-40, -1],
            *([[0.01, 1]] * nt),
        ]

        for (row, col) in np.ndindex(*self.s2_data.slai[0].shape):
            lai = self.s2_data.slai[:, row, col]
            svv = 10 * np.log10(self.s1_observations.VV[:, row, col])
            svh = 10 * np.log10(self.s1_observations.VH[:, row, col])
            theta = self.s1_observations.theta[:, row, col]
            retval = scipy.optimize.minimize(
                cost_nolai, x0, args=(svh, svh, lai, theta)
            )
            for i, raster in enumerate(outputs):
                raster[row, col] = retval.x[i]

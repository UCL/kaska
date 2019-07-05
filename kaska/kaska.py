# -*- coding: utf-8 -*-

"""Main module."""
import logging

import datetime as dt
import numpy as np

from NNParameterInversion import NNParameterInversion

from s2_observations import Sentinel2Observations

from smoothn import smoothn

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

            
def define_temporal_grid(start_date, end_date, temporal_grid_space):
    """Creates a temporal grid"""
    temporal_grid = [start_date + i*dt.timedelta(days=temporal_grid_space) 
                    for i in range(int(np.ceil(366/temporal_grid_space)))
                    if start_date + i*dt.timedelta(days=temporal_grid_space)
                                    <= end_date]
    return temporal_grid

class KaSKA(object):
    """The main KaSKA object"""

    def __init__(self, observations, time_grid, state_mask, approx_inverter,
                output_folder):
        self.time_grid = time_grid
        self.observations = observations
        self.state_mask = state_mask
        self.output_folder = output_folder
        self.inverter = NNParameterInversion(approx_inverter)

    def first_pass_inversion(self):
        """A first pass inversion. Could be anything, from a quick'n'dirty
        LUT, a regressor. As coded, we use the `self.inverter` method, which
        in this case, will call the ANN inversion."""
        S = {}
        for k in self.observations.dates:
            retval = self.inverter.invert_observations(self.observations, k)
            if retval is not None:
                S[k] = retval
        return S

    def _process_first_pass(self, first_passer_dict):
        """This methods takes the first pass estimates of surface parameters
        (stored as a dictionary) and assembles them into an
        `(n_params, n_times, nx, ny)` grid. The assumption here is the 
        dictionary is indexed by dates (e.g. datetime objects) and that for
        each date, we have a list of parameters.
        
        Parameters
        ----------
        first_passer_dict: dict
            A dictionary with first pass guesses in an irregular temporal grid
        
        """
        dates = [k for k in first_passer_dict.keys()]
        n_params, nx, ny = first_passer_dict[dates[0]].shape
        param_grid = np.zeros((n_params, len(self.time_grid), nx, ny))
        idx = np.argmin(np.abs(self.time_grid -
                        np.array(dates)[:, None]), axis=1)
        LOG.info("Re-arranging first pass solutions into an array")
        for ii, tstep in enumerate(self.time_grid):
            ## Number of observations in current time step
            #n_obs_tstep = list(idx).count(ii)
            # Keys for the current time step
            sel_keys = list(np.array(dates)[idx == ii])
            LOG.info(f"Doing timestep {str(tstep):s}")
            for k in sel_keys:
                LOG.info(f"\t {str(k):s}")
            for p in range(n_params):
                arr = np.array([first_passer_dict[k][p] for k in sel_keys])
                arr[arr < 0] = np.nan
                param_grid[p, ii, :, :] = np.nanmean(arr, axis=0)
        return dates, param_grid

    def run_retrieval(self):
        """Runs the retrieval for all time-steps. It proceeds by first 
        inverting on a observation by observation fashion, and then performs
        a per pixel smoothing/interpolation."""
        dates, retval = self._process_first_pass(self.first_pass_inversion())
        LOG.info("Burp!")
        x0 = np.zeros_like(retval)
        for param in range(retval.shape[0]):
            S = retval[param]*1
            ss = smoothn(S, isrobust=True, s=1, TolZ=1e-2, axis=0)
            x0[param, :, :] = ss[0]
        return x0



if __name__ == "__main__":
    start_date = dt.datetime(2017, 5, 1)
    end_date = dt.datetime(2017, 9, 1)
    temporal_grid_space = 5
    temporal_grid = define_temporal_grid(start_date, end_date,
                                        temporal_grid_space)
    s2_obs = Sentinel2Observations(
        "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/s2_obs/",
        "/home/ucfafyi/DATA/Prosail/prosail_2NN.npz",
        "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/ESU.tif",
        band_prob_threshold=20,
        chunk=None,
        time_grid=temporal_grid,
    )
    state_mask = "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/ESU.tif"
    approx_inverter = "/home/ucfafyi/DATA/Prosail/Prosail_5_paras.h5"
    kaska = KaSKA(s2_obs, temporal_grid, state_mask, approx_inverter,
                     "/tmp/")
    KK = kaska.run_retrieval()
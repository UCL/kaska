# -*- coding: utf-8 -*-

"""Main module."""
import logging

from pathlib import Path
import datetime as dt
import numpy as np

from scipy.interpolate import interp1d

from .NNParameterInversion import NNParameterInversion

from .s2_observations import Sentinel2Observations

from .s1_observations import Sentinel1Observations

from .smoothn import smoothn

from .utils import save_output_parameters

from .interp_fix import interp1d

LOG = logging.getLogger(__name__)
            
class KaSKA(object):
    """The main KaSKA object"""

    def __init__(self, observations, time_grid, state_mask, approx_inverter,
                output_folder,
                chunk = None):
        self.time_grid = time_grid
        self.observations = observations
        self.state_mask = state_mask
        self.output_folder = output_folder
        self.inverter = NNParameterInversion(approx_inverter)
        self.chunk = chunk
        self.save_sgl_inversion = True

    def first_pass_inversion(self):
        """A first pass inversion. Could be anything, from a quick'n'dirty
        LUT, a regressor. As coded, we use the `self.inverter` method, which
        in this case, will call the ANN inversion."""
        state_mask = self.observations.state_mask.ReadAsArray()
        state_mask = state_mask.astype(np.bool)
        LOG.info("Doing first pass inversion!")
        S = {}
        for k in self.observations.dates:
            retval = self.inverter.invert_observations(self.observations, k,
                                                       state_mask=state_mask)
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
        param_grid = np.zeros((n_params, len(dates), nx, ny))
        for i, k in enumerate(dates):
            for j in range(n_params):
                    param_grid[j, i, :, :] = first_passer_dict[k][j]
        # param_grid = np.zeros((n_params, len(self.time_grid), nx, ny))
        # idx = np.argmin(np.abs(self.time_grid -
        #                 np.array(dates)[:, None]), axis=1)
        # LOG.info("Re-arranging first pass solutions into an array")
        # for ii, tstep in enumerate(self.time_grid):
        #     ## Number of observations in current time step
        #     #n_obs_tstep = list(idx).count(ii)
        #     # Keys for the current time step
        #     sel_keys = list(np.array(dates)[idx == ii])
        #     LOG.info(f"Doing timestep {str(tstep):s}")
        #     for k in sel_keys:
        #         LOG.info(f"\t {str(k):s}")
        #     for p in range(n_params):
        #         arr = np.array([first_passer_dict[k][p] for k in sel_keys])
        #         arr[arr < 0] = np.nan
        #         param_grid[p, ii, :, :] = np.nanmean(arr, axis=0)
        return dates, param_grid

    def run_retrieval(self):
        """Runs the retrieval for all time-steps. It proceeds by first 
        inverting on a observation by observation fashion, and then performs
        a per pixel smoothing/interpolation."""
        dates, retval = self._process_first_pass(self.first_pass_inversion())
        LOG.info("Burp! Now doing temporal smoothing")
        return self._run_smoother(dates, retval)
        #x0 = np.zeros_like(retval)
        #for param in range(retval.shape[0]):
        #    S = retval[param]*1
        #    ss = smoothn(S, isrobust=True, s=1, TolZ=1e-2, axis=0)
        #    x0[param, :, :] = ss[0]
        #return x0

    def _run_smoother(self, dates, parameter_block):
        """Very specific method that applies some parameter transformations
        to the data in a very unrobust way."""
        # This needs to be abstracted up...
        # Note that in general, we don't know what parameters we are dealing
        # with. We probably want a data structure here with the parameter list,
        # transformation function, as well as boundaries, which could be
        # associated with the NN
        lai = -2 * np.log(parameter_block[-2, :, :, :])
        cab = -100*np.log(parameter_block[1, :, :, :])
        cbrown = parameter_block[2, :, :, :]
        if self.save_sgl_inversion is True:
            save_output_parameters(dates, self.observations, 
                self.output_folder/"single_imgs/", ["lai", "cab", "cbrown"],
                           [lai, cab, cbrown], output_format="GTiff",
                           chunk=self.chunk, fname_pattern="s2_sgl",
                           options=['COMPRESS=DEFLATE',
                                    'BIGTIFF=YES',
                                    'PREDICTOR=1',
                                    'TILED=YES'])
               
        # Basically, remove weird values outside of boundaries, nans and stuff
        # Could be done simply with the previously stated data structure, as
        # this is a bit of an adhoc piece of code.
        lai[~np.isfinite(lai)] = np.nan
        cab[~np.isfinite(cab)] = np.nan
        cbrown[~np.isfinite(cbrown)] = np.nan
        lai[lai < 0] = np.nan
        cab[cab < 0] = np.nan
        cbrown[cbrown < 0] = np.nan
        # Create a mask where we have no (LAI) data
        mask = np.all(lai == 0, axis=(0))
        
        # Time axes in days of year
        doys = np.array([int(x.strftime('%j')) for x in dates])
        doy_grid = np.array([int(x.strftime('%j')) for x in self.time_grid])
        # Do a linear interpolation for missing values in the observations
        laii = interp1d(doys, doys, lai)
        cabi = interp1d(doys, doys, cab)
        cbrowni = interp1d(doys, doys, cbrown)
        # There might be some NaNs around, set to 0
        laii[np.isnan(laii)] = 0
        cabi[np.isnan(cabi)] = 0
        cbrowni[np.isnan(cbrowni)] = 0
        # Smooth on observations grid
        slai = smoothn(np.array(laii), W=2*np.array(laii), isrobust=True, s=0.05,
                       TolZ=1e-6, axis=0)[0]
        slai[slai < 0] = 0
        scab = smoothn(np.array(cabi), W=slai, isrobust=True, s=0.5,
                        TolZ=1e-6, axis=0)[0]
        scbrown = smoothn(np.array(cbrowni), W=slai, isrobust=True, s=0.5,
                        TolZ=1e-6, axis=0)[0]
        # Interpolate to state grid
        laii = interp1d(doy_grid, doys, slai)
        cabi = interp1d(doy_grid, doys, scab)
        cbrowni =  interp1d(doy_grid, doys, scbrown)
        return (["lai", "cab", "cbrown"], [laii, cabi, cbrowni])

    def save_s2_output(self, parameter_names, output_data,
                       time_grid=None, output_format="GTiff"):
        if time_grid is None:
            time_grid = self.time_grid
        save_output_parameters(time_grid, self.observations,
                               self.output_folder,
                               parameter_names, output_data,
                               output_format=output_format,
                               chunk=self.chunk)

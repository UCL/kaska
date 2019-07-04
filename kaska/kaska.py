# -*- coding: utf-8 -*-

"""Main module."""
import datetime as dt
import numpy as np

from NNParameterInversion import NNParameterInversion

from s2_observations import Sentinel2Observations

from smoothn import smoothn

from TwoNN import Two_NN

# This stuff should go on its own file....

            
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
                emulator, output_folder):
        self.time_grid = time_grid
        self.observations = observations
        self.state_mask = state_mask
        self.output_folder = output_folder
        self.inverter = NNParameterInversion(approx_inverter)
        self.emulator = emulator

    def first_pass_inversion(self):
        """A first pass inversion. Could be anything, from a quick'n'dirty
        LUT, a regressor, ..."""
        S = {}
        for k in self.observations.dates:
            retval = self.inverter.invert_observations(self.observations, k)
            if retval is not None:
                S[k] = retval
        return S

    def _process_first_pass(self, first_passer_dict):
        """Aim of this method is to get the first pass estimates into a regular
        grid, and (possibly!), interpolate and smooth them out. Like a boss."""
        dates = [k for k in first_passer_dict.keys()]
        n_params, nx, ny = first_passer_dict[dates[0]].shape
        param_grid = np.zeros((n_params, len(self.time_grid), nx, ny))
        idx = np.argmin(np.abs(self.time_grid -
                        np.array(dates)[:, None]), axis=1)
        
        for ii, tstep in enumerate(self.time_grid):
            ## Number of observations in current time step
            #n_obs_tstep = list(idx).count(ii)
            # Keys for the current time step
            sel_keys = list(np.array(dates)[idx == ii])
            print(f"Doing timestep {str(tstep):s}")
            for k in sel_keys:
                print(f"\t {str(k):s}")
            for p in range(n_params):
                arr = np.array([first_passer_dict[k][p] for k in sel_keys])
                arr[arr < 0] = np.nan
                param_grid[p, ii, :, :] = np.nanmean(arr, axis=0)
        return dates, param_grid
                



    def run_smoother(self, ):
        # Questions here are how to solve things...
        # Two main options:
        # 1. NL solver using TRMs -> Should be feasible, but not there yet
        # 2. Linearised solver using sparse matrices
        # (3). NL solver (this probably first implementation as a test)
        # Other things to worry about is chunking... We can probably solve  
        # a bunch of pixels together (e.g. a 9x9 or something)
        # It might be a good idea to explore pre-conditioning, as we
        # expect contighous pixels to evolve similarly...
        dates, retval = self._process_first_pass(self.first_pass_inversion())
        x0 = retval * 1.0
        np.savez_compressed("data_dumper.npz", tsteps=self.time_grid,
                            parameters=retval)
        print("Burp!")
        for param in range(retval.shape[0]):
            S = retval[param]*1
            ss = smoothn(S, isrobust=True, s=1, TolZ=1e-6, axis=0)
            x0[param, :, :] = ss[0]
        return x0

def read_emulator(emulator_file="/home/ucfafyi/DATA/Prosail/prosail_2NN.npz"):
    f = np.load(str(emulator_file))
    emulator = Two_NN(Hidden_Layers=f.f.Hidden_Layers,
                      Output_Layers=f.f.Output_Layers)
    return emulator

if __name__ == "__main__":
    start_date = dt.datetime(2017, 5, 1)
    end_date = dt.datetime(2017, 9, 1)
    temporal_grid_space = 10
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
    emulator = read_emulator() # Read default emulator
    kaska = KaSKA(s2_obs, temporal_grid, state_mask, approx_inverter, emulator,
                 "/tmp/")
    KK = kaska.run_smoother()
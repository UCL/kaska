# -*- coding: utf-8 -*-

"""Main module."""
import datetime as dt
import numpy as np
import tensorflow as tf

from s2_observations import Sentinel2Observations

from smoothn import smoothn

from TwoNN import Two_NN

# This stuff should go on its own file....
class NNParameterInversion(object):
    def __init__(self, NN_file):
        self.inverse_param_model = tf.keras.models.load_model(NN_file)
        self.bands = [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B10",
            "B11",
            "B12",
        ]
        self.b_ind = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        

    def invert_observations(self, data, date):
        """Main method to invert observations using a NN inverter."""
        # The solved parameters are stored indexed by time step.
        # On each time step, one will get a bunch of parameters
        # This is inverter dependent.
        # We can also restrict the time period if it makes sense
        # We extract the reflectance and relevant metadata etc
        print(f"Extracting {str(date):s}")
        rho, mask, sza, vza, raa, rho_unc = data.read_granule(date)
        if rho is not None:
            rho = np.array(rho)[self.b_ind]
            nbands, ny, nx = rho.shape
            X = rho[:, mask]
            n_clr_pxls = mask.sum()
            # Stack the input of the inverter
            X = np.vstack(
                [rho[:, mask],
                np.ones(n_clr_pxls) * sza,
                np.ones(n_clr_pxls) * vza,
                np.ones(n_clr_pxls) * raa],
            )
            # Run the inversion, probably returns a tuple
            print("\tInverting...")
            retval = self.inverse_param_model.predict(X.T)
            
            n_cells, n_params = retval.shape
            params = np.zeros((n_params, ny, nx))
            for i in range(n_params):
                params[i, mask] = retval[:, i]
            return params
        else:
            return None  # No clear pixels!
            
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
        # Should read first and pass pointers to refletance around
        # refactor: go into its own function
        for date in dates:
            print(f"Extracting {str(date):s}")
            rhoi, maski, szai, vzai, raai, rho_unci = \
                                           self.observations.read_granule(date)
        
        S2_obs = s2_obs(np.array(obs_doy),
                    np.ones_like(obs_doy, dtype=np.bool)*True,
                    np.array(s2_refl).mean(axis=(2,3)),
                    np.array(sza), np.array(vza), np.array(raa),
                    np.array(s2_refl).mean(axis=(2,3))*0.005)

        cost_wrapper = CostWrapper(time_grid, S2_obs,
                                   [0, 500000, 0, 100000, 0, 0, 1000, 0,
                                   100000,100000], emu, mu_prior, A)

        retval = minimize(cost_wrapper.calc_cost, mu_prior,  jac=True,
                          method="L-BFGS-B", bounds=bounds,
                          options={"iprint":1, "maxcor":400, "maxiter":500,})

        return retval


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
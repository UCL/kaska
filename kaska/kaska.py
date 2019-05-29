# -*- coding: utf-8 -*-

"""Main module."""
import datetime as dt
import numpy as np
import tensorflow as tf

from .s2_observations import Sentinel2Observations


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

    def invert_observations(self, data, sel_dates=None):
        """Main method to invert observations using a NN inverter."""
        # The solved parameters are stored indexed by time step.
        # On each time step, one will get a bunch of parameters
        # This is inverter dependent.
        # We can also restrict the time period if it makes sense
        solved_params = {}
        for istep, date in enumerate(data.time):
            if (sel_dates is not None) and (date in sel_dates):
                # Current date not in selection
                continue
            # We extract the reflectance and relevant metadata etc
            rho = data.observations[istep][self.band_ind]
            sza, vza, raa = data.metadata[istep]
            mask = data.mask[istep]
            X = rho[:, mask]
            n_clr_pxls = mask.sum()
            # Stack the input of the inverter
            X = np.vstack(
                rho[:, mask],
                np.ones(n_clr_pxls) * sza,
                np.ones(n_clr_pxls) * vza,
                np.ones(n_clr_pxls) * raa,
            )
            # Run the inversion, probably returns a tuple
            retval = self.inverse_param_model.predict(X)
            solved_params[date] = retval


def define_temporal_grid(start_date, end_date, temporal_grid_space):
    """Creates a temporal grid"""
    temporal_grid = [start_date + i*dt.timedelta(days=temporal_grid_space) 
                    for i in range(int(ceil(366/temporal_grid_space)))
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

    def _run_approx_inversion(self, sel_dates=None):
        # Make sure we pass a list fo the inverter...
        if sel_dates is not None and not isinstance(sel_dates, list):
            sel_dates = list(sel_dates)
        retval = self.inverter.invert_observations(self.observations,
                                                   sel_dates=sel_dates)
        return retval  # Probably needs jiggery pokery

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
        pass


if __name__ == "__main__":
    start_date = dt.datetime(2017, 5, 1)
    end_date = dt.datetime(2017, 9, 1)
    temporal_grid_space = 10
    temporal_grid = define_temporal_grid (start_date, end_date, temporal_grid_space)
    s2_obs = Sentinel2Observations(
        "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/s2_obs/",
        "/home/ucfafyi/DATA/Prosail/prosail_2NN.npz",
        "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/ESU.tif",
        band_prob_threshold=20,
        chunk=None,
    )
    state_mask = "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/ESU.tif"
    approx_inverter = "/home/ucfafyi/DATA/Prosail/Prosail_5_paras.h5"
    kaska = KaSKA(observations, time_grid, state_mask, approx_inverter,
                 "/tmp/")
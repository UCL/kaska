#!/usr/bin/env python

import logging

from pathlib import Path

import numpy as np

import tensorflow as tf

LOG = logging.getLogger(__name__)

"""Neural Network Parameter inversion for Sentinel-2"""

class NNParameterInversion(object):
    """A class for inverint parameters from Sentinel2 data using a neural net.
    Code uses Tensorflow.keras to load up a pre-trained model. The user shold
    be able to select what band(s) get used for the inversion. By default,
    the VIS/NIR ones are used (e.g. no SWIR yet)."""
    def __init__(self, NN_file):
        """Set up NN parameter inversion.

        Parameters:
        NN_file: str
            An already trained emulator that can be read in by tf.keras.
        """
        path = Path(NN_file)
        if not path.exists():
            LOG.error(f"File {str(path):s} does not exist in the system")
            raise IOError(f"File {str(path):s} does not exist in the system")
        LOG.info(f"Using inverter file {NN_file:s}")

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
        ]  # Band names
        # Bands to be used:
        self.b_ind = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        
        

    def invert_observations(self, data, date):
        """Main method to invert observations using a NN inverter. Takes a 
        date and a data object. The data object could be one defined in 
        s2_observations.py, for example, where we have a `read_granule` method
        that when called with a data with observations returns the reflectance,
        mask, angles and uncertainty."""
        LOG.info(f"Extracting data for {str(date):s}...")
        # Read in the data and return a bunch of numpy arrays
        rho, mask, sza, vza, raa, rho_unc = data.read_granule(date)
        # rho will be None if there are no data available.
        
        if rho is not None:
            LOG.info(f"{mask.sum():d} pixels to be processed")
            # Subset the bands that we want to use for our inversion
            rho = np.array(rho)[self.b_ind]
            # Get some shapes of pixels
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
            LOG.info("\tInverting...")
            retval = self.inverse_param_model.predict(X.T)
            # OK, so we have the parameters. Re-arrange them on a 3D array
            # (params, ny, nx)
            n_cells, n_params = retval.shape
            params = np.zeros((n_params, ny, nx))
            for i in range(n_params):
                params[i, mask] = retval[:, i]
            return params
        else:
            LOG.info(f"No clear pixels")
            return None  # No clear pixels!

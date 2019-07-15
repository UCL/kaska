# -*- coding: utf-8 -*-

"""Writing output to disk."""
import logging
import os

from osgeo import gdal
from osgeo import osr

import numpy as np

from parmap import parmap

class KaskaOutput(object):
    """A very simple class to output the state."""
    def __init__(self, parameter_list, geotransform, projection, folder,
                 state_mask, prefix=None,
                 fmt="GTiff"):
        """The inference engine works on tiles, so we get the tilewidth
        (we assume the tiles are square), the GDAL-friendly geotransform
        and projection, as well as the destination directory and the
        format (as a string that GDAL can understand)."""
        self.geotransform = geotransform
        self.projection = projection
        self.folder = folder
        self.fmt = fmt
        self.parameter_list = parameter_list
        self.n_params = len(parameter_list)
        self.prefix = prefix
        self.state_mask = state_mask
        LOG.info("Setting up TIFF writer")
        LOG.info(f"\tOutput folder: {self.folder:s}")
        LOG.info(f"\tOutput format: {self.fmt:s}")
        LOG.info(f"\tOutput prefix: {self.prefix:s}")

    def dump_data(self, timegrid, x_solution,
                  state_mask, n_params):
        
        def write_data(inx):
            ii, param = inx
            for jj, timestep in enumerate(timegrid):
                drv = gdal.GetDriverByName(self.fmt)
                if self.prefix is None:
                    fname = os.path.join(self.folder, "%s_%s.tif" %
                                        (param, timestep.strftime("A%Y%j")))
                else:
                    fname = os.path.join(self.folder, "%s_%s_%s.tif" %
                                        (param, timestep.strftime("A%Y%j"),
                                        self.prefix))
                dst_ds = drv.Create(fname, state_mask.shape[1],
                                    state_mask.shape[0], 1,
                                    gdal.GDT_Float32, ['COMPRESS=DEFLATE',
                                                    'BIGTIFF=YES',
                                                    'PREDICTOR=1',
                                                    'TILED=YES'])
                dst_ds.SetProjection(self.projection)
                dst_ds.SetGeoTransform(self.geotransform)
                A = np.zeros(state_mask.shape, dtype=np.float32)
                A[state_mask] = x_solution[ii, jj, :, :]
                dst_ds.GetRasterBand(1).WriteArray(A)
        LOG.info("Writing outputs...")
        par_list = [(ii, param) 
                            for (ii, param) in enumerate(self.parameter_list)]
        retval = list(parmap(write_data, par_list, Nt=len(self.parameter_list)))
        bothered = False
        if bothered:
            LOG.info("Not saving uncertainty. Yet.")
            ############################################################
            ############################################################
            #### NOT WORKING, NOT FIGURED OUT UNCERTAINTY YET!!!!!!!!!!!
            ############################################################
            ############################################################
            # # # LOG.info("Saving posterior inverse covariance matrix")
            # # # # Proabably also save the state...
            # # # # And the propagation matrix, unless it's just feasible to
            # # # # reconstruct it elsewhere
            # # # sp.save_npz(os.path.join(
            # # #             self.folder,
            # # #             f"P_analysis_inv_{timestep.strftime('A%Y%j'):s}.npz"),
            # # #             P_analysis_inv)
            # # # np.savez_compressed(os.path.join(
            # # #             self.folder,
            # # #             f"x_analysis_{timestep.strftime('A%Y%j'):s}.npz"),
            # # #             x_analysis=x_analysis)
            # # # # Probably need to save state mask and other things to "unwrap"
            # # # # the matrix, such as parameters and so on
            # # # bothered = False
            # # # LOG.info("Not saving uncertainties")
            # # # if bothered:
            # # #     def write_unc(inx):
            # # #         ii, param = inx
            # # #         if self.prefix is None:
            # # #             fname = os.path.join(self.folder, "%s_%s_unc.tif" %
            # # #                                 (param, timestep.strftime("A%Y%j")))
            # # #         else:
            # # #             fname = os.path.join(self.folder, "%s_%s_%s_unc.tif" %
            # # #                                 (param, timestep.strftime("A%Y%j"),
            # # #                                 self.prefix))
            # # #         dst_ds = drv.Create(fname, state_mask.shape[1],
            # # #                             state_mask.shape[0], 1,
            # # #                             gdal.GDT_Float32, ['COMPRESS=DEFLATE',
            # # #                                             'BIGTIFF=YES',
            # # #                                             'PREDICTOR=1', 'TILED=YES'])
            # # #         dst_ds.SetProjection(self.projection)
            # # #         dst_ds.SetGeoTransform(self.geotransform)
            # # #         A = np.zeros(state_mask.shape, dtype=np.float32)
            # # #         A[state_mask] = 1./np.sqrt(P_analysis_inv.diagonal()[ii::n_params])
            # # #         dst_ds.GetRasterBand(1).WriteArray(A)
            # # #     retval = list(parmap(write_unc, par_list, Nt=len(self.parameter_list)))

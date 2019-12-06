#!/usr/bin/env python
"""Dealing with Sentinel 1 observations"""


import datetime as dt
import logging
from collections import namedtuple
from pathlib import Path

import gdal
import numpy as np

from .utils import reproject_data, define_temporal_grid

gdal.UseExceptions()

LOG = logging.getLogger(__name__)

# A SIAC data storage type
S1data = namedtuple(
    # "S1data", "time observations uncertainty mask metadata emulator"
    "S1data", "time VV VH theta VV_unc VH_unc"
)


LAYERS = [
    "sigma0_vv_norm_multi_db",
    "sigma0_vh_norm_multi_db",
    "localIncidenceAngle"
    ]


def get_s1_dates(s1_file):
    """Gets the dates from a LMU processed netCDF Sentinel 1 file"""
    times = [float(s1_file.GetRasterBand(b+1).GetMetadata()['NETCDF_DIM_time'])
             for b in range(s1_file.RasterCount)]
    times = [dt.datetime(1970, 1, 1) + dt.timedelta(days=x) for x in times]
    # LOG.info(f"Sentinel 1 First obs: {times[0].strftime('%Y-%m-%d'):s}")
    LOG.info("Sentinel 1 First obs: %s", times[0].strftime('%Y-%m-%d'))
    # LOG.info(f"Sentinel 1 Last obs: {times[-1].strftime('%Y-%m-%d'):s}")
    LOG.info("Sentinel 1 Last obs: %s", times[-1].strftime('%Y-%m-%d'))
    return times


class Sentinel1Observations:
    """Class for dealing with Sentinel 1 observations"""
    def __init__(
            self,
            netCDF_file,
            state_mask,
            chunk=None,
            time_grid=None,
            nc_layers=None
            ):
        if nc_layers is None:
            nc_layers = {"VV": "sigma0_vv_norm_multi_db",
                         "VH": "sigma0_vh_norm_multi_db",
                         "theta": "localIncidenceAngle"}
        self.time_grid = time_grid
        self.state_mask = state_mask
        self.nc_file = Path(netCDF_file)
        self.nc_layers = nc_layers
        self._match_to_mask()

    def apply_roi(self, ulx, uly, lrx, lry):
        """Apply a region of interest"""
        # self.ulx = ulx
        # self.uly = uly
        # self.lrx = lrx
        # self.lry = lry
        width = lrx - ulx
        height = uly - lry

        self.state_mask = gdal.Translate(
            "",
            self.original_mask,  # Error: class member does not exist.
            srcWin=[ulx, uly, width, abs(height)],
            format="MEM",
        )
        LOG.info(f"Applied ROI ulx, uly = ({ulx:d}, {uly:d}," +
                 f" w,h {width:d}, {height:d}")
        self._match_to_mask()

    def define_output(self):
        """Define the output array shapes to be consistent with the state
        mask. You get the projection and geotransform, that should be
        enough to define an ouput dataset that conforms to the state mask.

        Returns
        -------
        tuple
            The first element is the projection string (WKT probably?), and
            the second element is the geotransform.
        """
        try:
            dataset = gdal.Open(self.state_mask)
            proj = dataset.GetProjection()
            geo_transform = np.array(dataset.GetGeoTransform())
            num_x = dataset.RasterXSize
            num_y = dataset.RasterYSize

        except RuntimeError:
            proj = self.state_mask.GetProjection()
            geo_transform = np.array(self.state_mask.GetGeoTransform())
            num_x = self.state_mask.RasterXSize
            num_y = self.state_mask.RasterYSize

        # new_geoT = geoT*1.
        # new_geoT[0] = new_geoT[0] + self.ulx*new_geoT[1]
        # new_geoT[3] = new_geoT[3] + self.uly*new_geoT[5]
        return proj, geo_transform.tolist(), num_x, num_y  # new_geoT.tolist()

    def _match_to_mask(self):
        """Matches the observations to the state mask.
        """
        self.s1_data_ptr = {}
        last_layer = None
        for layer, layer_name in self.nc_layers.items():
            fname = f'NETCDF:"{self.nc_file.as_posix():s}":{layer_name:s}'
            self.s1_data_ptr[layer] = \
                reproject_data(fname,
                               output_format="VRT",
                               src_srs="EPSG:4326",
                               target_img=self.state_mask)
            last_layer = layer
        s1_dates = get_s1_dates(self.s1_data_ptr[last_layer])
        self.dates = {x: (i+1)
                      for i, x in enumerate(s1_dates)
                      if self.time_grid[0] <= x <= self.time_grid[-1]
                      }

    def read_time_series(self, time_grid):
        """Reads a time series of observations. Uses the time grid to provide
        a min/max times.

        Parameters
        ----------
        time_grid : list of datetimes
            List of datetimes

        Returns
        -------
        S1data
            An object with arrays containing the VV, VH and theta, as well
            as uncertainties...
        """
        early = time_grid[0]
        late = time_grid[-1]

        sel_dates = [k for k, v in self.dates.items()
                     if early <= k <= late]
        sel_bands = [v for k, v in self.dates.items()
                     if early <= k <= late]
        obs = {}
        for j, layer in enumerate(self.s1_data_ptr.keys()):
            obs[layer] = np.array([self.s1_data_ptr[
                layer].GetRasterBand(i).ReadAsArray()
                                   for i in sel_bands])
        the_obs = S1data(sel_dates, obs['VV'], obs['VH'],
                         obs['theta'], 0.5, 0.5)
        return the_obs


if __name__ == "__main__":
    START_DATE = dt.datetime(2017, 3, 1)
    END_DATE = dt.datetime(2017, 9, 1)
    TEMPORAL_GRID_SPACE = 5
    TEMPORAL_GRID = define_temporal_grid(START_DATE, END_DATE,
                                         TEMPORAL_GRID_SPACE)
    NC_FILE = "/data/selene/ucfajlg/ELBARA_LMU/mirror_ftp/" + \
              "141.84.52.201/S1/S1_LMU_site_2017_new.nc"
    S1_OBS = Sentinel1Observations(
                NC_FILE,
                "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/ESU.tif",
                time_grid=TEMPORAL_GRID)
    VV = S1_OBS.read_time_series(TEMPORAL_GRID[:5])

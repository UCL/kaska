#!/usr/bin/env python
"""Dealing with S2 observations"""


import datetime as dt
import logging
from collections import namedtuple
from pathlib import Path

import gdal
import numpy as np

from .TwoNN import Two_NN

from .utils import reproject_data

gdal.UseExceptions()

LOG = logging.getLogger(__name__)
# A SIAC data storage type
S2MSIdata = namedtuple(
    "S2MSIdata", "time observations uncertainty mask metadata emulator"
)


class Sentinel2Observations(object):
    def __init__(
        self,
        parent_folder,
        emulator,
        state_mask,
        band_prob_threshold=5,
        chunk=None,
        time_grid=None,
    ):
        self.band_prob_threshold = band_prob_threshold
        parent_folder = Path(parent_folder)
        if not parent_folder.exists():
            LOG.info(f"S2 data folder: {parent_folder}")
            raise IOError("S2 data folder doesn't exist")
        self.band_map = [
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
        # self.band_map = ['05', '08']

        self.parent = parent_folder
        self.original_mask = state_mask
        self.state_mask = state_mask

        f = np.load(emulator, allow_pickle=True)
        self.emulator = Two_NN(
            Hidden_Layers=f.f.Hidden_Layers, Output_Layers=f.f.Output_Layers
        )
        LOG.debug("Read emulator in")
        LOG.debug("Searching for files....")
        self._find_granules(self.parent, time_grid)
        self.chunk = chunk

    def apply_roi(self, ulx, uly, lrx, lry):
        """Applies a region of interest (ROI) window to the state mask, which is
        then used to subset the data spatially. Useful for spatial windowing/
        chunking
        
        Parameters
        ----------
        ulx : integer
            The Upper Left corner of the state mask (in pixels). Easting.
        uly : integer
            The Upper Left corner of the state mask (in pixels). Northing.
        lrx : integer
            The Lower Right corner of the state mask (in pixels). Easting.
        lry : integer
            The Lower Right corner of the state mask (in pixels). Northing.
        Returns
        -------
        None
        Doesn't return anything, but changes `self.state_mask`
        """
        self.ulx = ulx
        self.uly = uly
        self.lrx = lrx
        self.lry = lry
        width = lrx - ulx
        height = uly - lry

        self.state_mask = gdal.Translate(
            "",
            self.original_mask,
            srcWin=[ulx, uly, width, abs(height)],
            format="MEM",
        )

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
            g = gdal.Open(self.state_mask)
            proj = g.GetProjection()
            geoT = np.array(g.GetGeoTransform())
            nx = g.RasterXSize
            ny = g.RasterYSize
        except RuntimeError:
            proj = self.state_mask.GetProjection()
            geoT = np.array(self.state_mask.GetGeoTransform())
            nx = self.state_mask.RasterXSize
            ny = self.state_mask.RasterYSize
        # new_geoT = geoT*1.
        # new_geoT[0] = new_geoT[0] + self.ulx*new_geoT[1]
        # new_geoT[3] = new_geoT[3] + self.uly*new_geoT[5]
        return proj, geoT.tolist(), nx, ny  # new_geoT.tolist()

    def _find_granules(self, parent_folder, time_grid=None):
        """Finds granules. Currently does so by checking for
        Feng's AOT file."""
        # this is needed to follow symlinks
        test_files = [
            x for f in parent_folder.iterdir() for x in f.rglob("**/*_aot.tif")
        ]
        try:
            dates = [
                dt.datetime(*(list(map(int, f.parts[-5:-2]))))
                for f in test_files
            ]
        except ValueError:
            dates = [
                dt.datetime.strptime(
                    f.parts[-1].split("_")[1], "%Y%m%dT%H%M%S"
                )
                for f in test_files
            ]
        # Sort dates by time, as currently S2A/S2B will be part of ordering

        # test_files = sorted(test_files, key=lambda x:dates[test_files.index(x)])
        # dates = sorted(dates)
        if time_grid is not None:
            start_date = time_grid[0]
            end_date = time_grid[-1]
            self.dates = [
                d.replace(hour=0, minute=0, second=0)
                for d in dates
                if (start_date <= d <= end_date)
            ]
            test_files = [
                test_files[i]
                for i, d in enumerate(dates)
                if (start_date <= d <= end_date)
            ]
        else:
            self.dates = [x.replace(hour=0, minute=0, second=0) for x in dates]
        temp_dict = dict(zip(self.dates, [f.parent for f in test_files]))
        dates = sorted(self.dates)
        self.date_data = {k: temp_dict[k] for k in dates}
        self.dates = dates

        # self.date_data = dict(zip(self.dates, [f.parent for f in test_files]))
        self.bands_per_observation = {}
        LOG.info(f"Found {len(test_files):d} S2 granules")
        LOG.info(
            f"First granule: "
            + f"{sorted(self.dates)[0].strftime('%Y-%m-%d'):s}"
        )
        LOG.info(
            f"Last granule: "
            + f"{sorted(self.dates)[-1].strftime('%Y-%m-%d'):s}"
        )

        for the_date in self.dates:
            self.bands_per_observation[the_date] = len(self.band_map)

    def read_time_series(self, time_grid):
        """Reads a time series of S2 data
        Arguments:
            time_grid  -- A list of dates
        Returns:
             A list of S2MSIdata objects
        """
        start_time = min(time_grid)
        end_time = max(time_grid)
        obs_dates = [
            date
            for date in self.dates
            if ((date >= start_time) & (date <= end_time))
        ]
        data = [self.read_granule(date) for date in obs_dates]
        observations = [x[0] for x in data if x[1] is not None]
        obs_dates = [
            obs_dates[i] for i in range(len(data)) if data[i][1] is not None
        ]
        mask = [x[1] for x in data if x[1] is not None]
        metadata = [[x[2], x[3], x[4]] for x in data if x[1] is not None]
        uncertainty = [x[5] for x in data if x[1] is not None]
        s2_obs = S2MSIdata(
            obs_dates, observations, uncertainty, mask, metadata, self.emulator
        )
        return s2_obs

    def read_granule(self, timestep):
        """Reads data granule for a given `timestep`. Returns all relevant 
        bits and bobs (surface reflectrance, angles, cloud mask, uncertainty).
        The mask is true for OK pixels. If there are no suitable pixels, the
        returned tuple is a collection of `None`
        
        
        Parameters
        ----------
        timestep : datetime
            The datetime object
        
        Returns
        -------
        tuple
            rho_surface, mask, sza, vza, raa, rho_unc
        """

        assert timestep in self.date_data, f"{str(timestep):s} not available!"
        # NOTE: Currently reads in sequentially. It's better to gather
        # all the filenames and read them in parallel using parmap.py"""
        current_folder = self.date_data[timestep]
        fname_prefix = [
            f.name.split("B02")[0] for f in current_folder.glob("*B02_sur.tif")
        ][0]
        # Find cloud mask
        cloud_mask = current_folder.parent / f"cloud.tif"
        cloud_mask = reproject_data(
            str(cloud_mask), target_img=self.state_mask
        ).ReadAsArray()
        # cloud mask is probabilty of cloud
        # OK pixels have a probability of cloud below `band_prob_threshold`
        mask = cloud_mask <= self.band_prob_threshold
        # If we have no unmasked pixels, bail out.
        if mask.sum() == 0:
            # No pixels! Pointless to carry on reading!
            LOG.info("No clear observations")
            return None, None, None, None, None, None
        # Read in surface reflectance and associated uncertainty
        rho_surface = []
        rho_unc = []
        for the_band in self.band_map:
            original_s2_file = current_folder / (
                f"{fname_prefix:s}" + f"{the_band:s}_sur.tif"
            )
            LOG.debug(f"Original file {str(original_s2_file):s}")
            rho = reproject_data(
                str(original_s2_file), target_img=self.state_mask
            ).ReadAsArray()

            rho_surface.append(rho)
            original_s2_file = current_folder / (
                f"{fname_prefix:s}" + f"{the_band:s}_sur_unc.tif"
            )
            # LOG.debug(f"Uncertainty file {str(original_s2_file):s}")
            # unc = reproject_data(
            #    str(original_s2_file), target_img=self.state_mask
            # ).ReadAsArray()
            rho_unc.append(np.ones_like(rho) * 0.005)
        # For reference...
        # bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
        #         'B8A', 'B09', 'B10','B11', 'B12']
        # b_ind = np.array([1, 2, 3, 4, 5, 6, 7, 8])

        rho_surface = np.array(rho_surface)
        # Now, ensure all surface reflectance pixels have values above
        # 0 & aren't cloudy.
        # So valid pixels if all refl > 0 AND mask is True
        # Array of the desired bands. Not necessarily contiguous. 
        sel_bands = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        mask1 = np.logical_and(
            np.all(rho_surface[sel_bands] > 0, axis=0), mask
        )
        mask = mask1
        if mask.sum() == 0:
            LOG.info(f"{str(timestep):s} -> No clear observations")
            return None, None, None, None, None, None
        LOG.info(
            f"{str(timestep):s} -> Total of {mask.sum():d} clear pixels "
            + f"({100.*mask.sum()/np.prod(mask.shape):f}%)"
        )

        rho_surface = rho_surface / 10000.0

        rho_unc = np.array(rho_unc) / 10000.0
        rho_unc[:, ~mask] = np.nan
        # Average uncertainty over the image
        rho_unc = np.nanmean(rho_unc, axis=(1, 2))
        # Set missing pixels to NaN
        rho_surface[:, ~mask] = np.nan
        # Now read angles
        sun_angles = reproject_data(
            str(current_folder.parent / "ANG_DATA/SAA_SZA.tif"),
            target_img=self.state_mask,
            xRes=20,
            yRes=20,
            resample=0,
        ).ReadAsArray()
        view_angles = reproject_data(
            str(current_folder.parent / "ANG_DATA/VAA_VZA_B05.tif"),
            target_img=self.state_mask,
            xRes=20,
            yRes=20,
            resample=0,
        ).ReadAsArray()
        sza = np.cos(np.deg2rad(sun_angles[1].mean() / 100.0))
        vza = np.cos(np.deg2rad(view_angles[1].mean() / 100.0))
        saa = sun_angles[0].mean() / 100.0
        vaa = view_angles[0].mean() / 100.0
        raa = np.cos(np.deg2rad(vaa - saa))
        return rho_surface, mask, sza, vza, raa, rho_unc


if __name__ == "__main__":
    time_grid = []
    today = dt.datetime(2017, 1, 1)
    while today <= dt.datetime(2017, 12, 31):
        time_grid.append(today)
        today += dt.timedelta(days=5)

    s2_obs = Sentinel2Observations(
        "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/s2_obs/",
        "/home/ucfafyi/DATA/Prosail/prosail_2NN.npz",
        "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/ESU.tif",
        band_prob_threshold=20,
        chunk=None,
        time_grid=time_grid,
    )
    retval = s2_obs.read_time_series(
        [dt.datetime(2017, 1, 1), dt.datetime(2017, 12, 31)]
    )

#!/usr/bin/env python
"""Dealing with S2 observations"""


import datetime as dt
import logging
from collections import namedtuple
from pathlib import Path
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

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
        band_prob_threshold=20,
        chunk=None,
        time_grid=None
    ):
        """
        Initialise the Sentinel2Observations object.

        Class members initialised in __init__ :
        parent : the path to the parent folder containing all the data.
        emulator :
        original_mask : the mask to be applied to the data (tif file)
                        originally given as input argument.
        state_mask : the mask to be applied to the data (tif file) after
                     possible alterations.
        band_prob_threshold : (optional)
        band_map : the list of available bands - hardcoded
        chunk : the size of the chunk/tile for parallel processing (optional)

        Class members initialised in _find_granules :
        dates : list of dates to run the analysis on (data files available
                for those dates)
        date_data : dictionary of date-folder pairs, ordered in date
        bands_per_observation : dictionary of date-band_map pairs, ordered
                                in date
        """
        parent_folder = Path(parent_folder)
        if not parent_folder.exists():
            LOG.info(f"S2 data folder: {parent_folder}")
            raise IOError("S2 data folder doesn't exist")
        self.parent = parent_folder

        f = np.load(emulator, allow_pickle=True)
        self.emulator = Two_NN(
            Hidden_Layers=f.f.Hidden_Layers, Output_Layers=f.f.Output_Layers
        )
        LOG.debug("Read emulator in")

        self.original_mask = state_mask
        self.state_mask = state_mask

        self.band_prob_threshold = band_prob_threshold
        self.band_map = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                         'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

        self.chunk = chunk

        LOG.debug("Searching for files....")
        self._find_granules(time_grid)

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

    def _find_granules(self, time_grid=None):
        """Finds granules within the given time grid if given.
        Currently does so by checking for Feng's AOT file.
        """
        folders = sorted([x for f in self.parent.iterdir()
                          for x in f.rglob("*.SAFE")
                          if x.name.find("MSIL1C") >= 0])

        # Apply multithreaded
        dates = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            for params in executor.map(
                    lambda x: self._process_xml(x/"MTD_MSIL1C.xml"), folders):
                dates.append(params)

        self.dates = [x[0].replace(hour=0, minute=0, second=0, microsecond=0)
                      for x in dates]
        self.date_data = dict(zip(self.dates, folders))
        #self.date_data = dict(dates)

        LOG.info(f"Found {len(dates):d} S2 granules")
        LOG.info(
            f"First granule: "
            + f"{sorted(self.dates)[0].strftime('%Y-%m-%d'):s}"
        )
        LOG.info(
            f"Last granule: "
            + f"{sorted(self.dates)[-1].strftime('%Y-%m-%d'):s}"
        )

        # Fill bands_per_observation dictionary with sorted dates-band_map data
        self.bands_per_observation = {}
        for the_date in self.dates:
            self.bands_per_observation[the_date] = len(self.band_map)

    def _process_xml(self, metadata_file):
        tree = ET.parse(metadata_file.as_posix())
        root = tree.getroot()
        acq_time = [time.text for time in root.iter("PRODUCT_START_TIME")]
        date = dt.datetime.strptime(acq_time[0], "%Y-%m-%dT%H:%M:%S.%fZ")
        refl_files = [metadata_file.parent/f"{granule.text:s}_sur.tif"
                      for granule in root.iter('IMAGE_FILE')
                      if not granule.text.endswith("TCI")]
        # Can check that all refl_files exist
        # Can also figure out where the cloud mask is (e.g. 'mask.tif'
        # one folder up from the tif files)
        # Angles are also one folder up in folder ANG_DATA
        return date, refl_files

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
        """NOTE: Currently reads in sequentially. It's better to gather
        all the filenames and read them in parallel using parmap.py"""
        current_folder = self.date_data[timestep]

        fname_prefix = [
            f.name.split("B02")[0] for f in current_folder.glob("*B02_sur.tif")
        ][0]
        cloud_mask = current_folder.parent / f"cloud.tif"
        
        cloud_mask = reproject_data(
            str(cloud_mask), target_img=self.state_mask
        ).ReadAsArray()
        mask = cloud_mask <= self.band_prob_threshold
        if mask.sum() == 0:
            # No pixels! Pointless to carry on reading!
            LOG.info("No clear observations")
            return None, None, None, None, None, None

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
            LOG.debug(f"Uncertainty file {str(original_s2_file):s}")
            unc = reproject_data(
                str(original_s2_file), target_img=self.state_mask
            ).ReadAsArray()
            rho_unc.append(unc)
        # For reference...
        #bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
        #         'B8A', 'B09', 'B10','B11', 'B12']
        # b_ind = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        
        rho_surface = np.array(rho_surface)
        
        
        mask1 = np.all(rho_surface[[1, 2, 3, 4, 5, 6, 7, 8, ]] > 0,
                       axis=0) & (~mask)
        mask = ~mask1
        rho_surface = rho_surface / 10000.0
        
        #mask = np.all(rho_surface > 0, axis=0) & (~mask)

        rho_unc = np.array(rho_unc) / 10000.0
        rho_unc[:, mask] = np.nan
        # Average uncertainty over the image
        rho_unc = np.nanmean(rho_unc, axis=(1, 2))
        rho_surface[:, mask1] = np.nan
        if mask.sum() == 0:
            LOG.info(f"{str(timestep):s} -> No clear observations")
            return None, None, None, None, None, None
        LOG.info(
            f"{str(timestep):s} -> Total of {mask.sum():d} clear pixels "
            + f"({100.*mask.sum()/np.prod(mask.shape):f}%)"
        )
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
    # import pickle

    time_grid = []
    today = dt.datetime(2017,1,1)
    while (today <= dt.datetime(2017, 12, 31)):
        time_grid.append(today)
        today += dt.timedelta(days=5)

    s2_obs = Sentinel2Observations(
        #        "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/s2_obs/",
        #        "/home/ucfafyi/DATA/Prosail/prosail_2NN.npz",
        #        "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/ESU.tif",
        "/home/ilektra/kaska/kaska/tests/data/s2_data",
        "/home/ilektra/kaska/kaska/inverters/prosail_2NN.npz",
        "/home/ilektra/kaska/kaska/tests/data/ESU.tif",
        band_prob_threshold=20,
        chunk=None,
        time_grid=time_grid
    )
    retval = s2_obs.read_time_series([dt.datetime(2017, 1, 1),
                                      dt.datetime(2017,12,31)])

    # pickle.dumps(retval)

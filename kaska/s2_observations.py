#!/usr/bin/env python
"""Dealing with S2 observations"""


import datetime as dt
import logging
from collections import namedtuple
from pathlib import Path
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import itertools

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


class Sentinel2Observations():
    """Class for dealing with Sentinel 2 observations"""
    def __init__(
            self,
            parent_folder,
            emulator,
            state_mask,
            band_prob_threshold=5,
            chunk=None,
            time_grid=None,
    ):
        """
        Initialise the Sentinel2Observations object.

        Parameters
        ----------
        parent_folder : str
            path of the top folder that contains the data
        emulator      : str
            the s2 emulator filename
        state_mask    : str
            an existing spatial raster with the binary mask detailing which
            pixels to process
        band_prob_threshold : int, optional
            threshold for xy pixel to be accepted
        chunk         : int, optional
            index of the xy chunk being read/processed. If not given, the whole
            area will be read in.
        time_grid     : list, optional
            list of datetime objects corresponding to the timesteps to be read.
            If not given, all the timesteps will be read in.

        Returns
        -------
        None

        Attributes
        ----------
        **Initialised in __init__:**
        parent        : path
            the path to the parent folder containing all the data
        emulator      : two_nn object
            the s2 emulator
        original_mask : tif file
            the mask to be applied to the data without alterations
        state_mask    : tif file
            the mask to be applied to the data after possible alterations
        band_prob_threshold : int
            threshold for xy pixel to be accepted
        band_map      : list
            the list of available bands - hardcoded
        chunk         : int
            index of the xy chunk being read/processed

        **Initialised in _find_granules:**
        date_data : dict
            dictionary of {date:[reflectivity files]}, ordered in date
        """
        parent_folder = Path(parent_folder)
        if not parent_folder.exists():
            LOG.info(f"S2 data folder: {parent_folder}")
            raise IOError("S2 data folder doesn't exist")
        self.parent = parent_folder

        my_file = np.load(emulator, allow_pickle=True)
        self.emulator = Two_NN(
            Hidden_Layers=my_file.f.Hidden_Layers,
            Output_Layers=my_file.f.Output_Layers
        )
        LOG.debug("Read emulator in")

        self.original_mask = state_mask
        self.state_mask = state_mask

        self.band_prob_threshold = band_prob_threshold
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
        self.chunk = chunk

        LOG.debug("Searching for files....")
        self._find_granules()

    def apply_roi(self, ulx, uly, lrx, lry):
        """Applies a region of interest (ROI) window to the state mask, which
        is then used to subset the data spatially. Useful for spatial
        windowing/chunking

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

        return proj, geo_transform.tolist(), num_x, num_y

    def _find_granules(self, max_workers=10):
        """Finds granules within the given time grid if given.

        Parameters
        ----------
        max_workers : int, optional
            the maximum number of worker threads used to read xml files
            concurrently

        Returns
        -------
        None
        Doesn't return anything, but changes `self.date_data`
        """
        # This is a list of the granule parent folders. The xml file in each
        # of those folders, contains the relative paths to all the reflectivity
        # data files for that granule.
        folders = sorted([x for x in self.parent.rglob("*/*MSIL1C*.SAFE")
                          if x.is_dir()])

        # Apply multithreaded
        date_files = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for params in executor.map(
                    lambda x: self._process_xml(x/"MTD_MSIL1C.xml"), folders):
                date_files.append(params)

        dates = [x[0].replace(hour=0, minute=0, second=0, microsecond=0)
                 for x in date_files]
        self.date_data = dict(zip(dates, [x[1] for x in date_files]))
        # TODO: Could do here instead of in read_granule():
        # - Check that all refl_files exist
        # - Figure out where the cloud mask is (e.g. 'cloud.tif'
        #   one folder up from the tif files)
        # - Angles are also one folder up in folder ANG_DATA

        LOG.info(f"Found {len(dates):d} S2 granules")
        LOG.info(
            f"First granule: {min(dates):%Y-%m-%d}"
        )
        LOG.info(
            f"Last granule: {max(dates):%Y-%m-%d}"
        )

    def _process_xml(self, metadata_file):
        """Processes the input xml file to fish out time series and file names.

        Parameters
        ----------
        metadata_file : path
            the xml file
        Returns
        -------
        tuple
            The first element is the datetime, and the second element is a list
            of the data files for this datetime.
        """
        tree = ET.parse(str(metadata_file))
        root = tree.getroot()
        acq_time = [time.text for time in root.iter("PRODUCT_START_TIME")]
        date = dt.datetime.strptime(acq_time[0], "%Y-%m-%dT%H:%M:%S.%fZ")
        refl_files = [[metadata_file.parent/f"{granule.text:s}_sur.tif",
                       metadata_file.parent/f"{granule.text:s}_sur_unc.tif"]
                      for granule in root.iter('IMAGE_FILE')
                      if not granule.text.endswith("TCI")]
        refl_files = list(itertools.chain.from_iterable(refl_files))

        return date, refl_files

    def read_time_series(self, time_grid):
        """Reads a time series of S2 data

        Parameters
        ----------
        time_grid : list
            the list of dates to read
        Returns
        -------
        list
            a list of S2MSIdata objects corresponding to the requested dates
        """
        start_time = min(time_grid)
        end_time = max(time_grid)
        obs_dates = [
            date
            for date in self.date_data
            if (start_time <= date <= end_time)
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
        """Reads the data for the given timestep. Returns all relevant
        quantities (surface reflectrance, angles, cloud mask, uncertainty).
        The mask is true for OK pixels. If there are no suitable pixels, the
        returned tuple is a collection of `None`

        Parameters
        ----------
        timestep : datetime
            datetime of granule to read
        Returns
        -------
        tuple
            contains the rho_surface, mask, sza, vza, raa, rho_unc

        .. note:: Currently reads in sequentially. It's better to gather
        all the filenames and read them in parallel using parmap.py
        """
        assert timestep in self.date_data, f"{timestep} not available!"

        # This is the parent folder for this granule: the path from the current
        # location all the way down to the one containing ".SAFE"
        current_folder = Path()
        for part in self.date_data[timestep][0].parts:
            current_folder /= part
            if ".SAFE" in part:
                break
        assert current_folder is not Path(), f"""Parent folder for granule
         {timestep:%Y-%m-%d} does not follow expected pattern: should end in
         '.SAFE'"""

        # Read in cloud mask and reproject it on state mask.
        # The cloud mask is the probabilty of cloud. OK pixels have
        # a probability of cloud below `band_prob_threshold`.
        # Stop processing if cloud mask file doesn't exist,
        # or if no clear pixels exist.
        try:
            cloud_mask_file = next(current_folder.glob("**/cloud.tif"))
        except StopIteration:
            LOG.info(f"Cloud file cloud.tif does not exist.")
            return (None,) * 6
        cloud_mask = reproject_data(
            str(cloud_mask_file), target_img=self.state_mask).ReadAsArray()
        mask = cloud_mask <= self.band_prob_threshold
        # If we have no unmasked pixels, bail out.
        if mask.sum() == 0:
            # No pixels! Pointless to carry on reading!
            LOG.info("No clear observations")
            return (None,) * 6

        # Read in surface reflectance and associated uncertainty per band
        rho_surface = []
        rho_unc = []
        s2_files = self.date_data[timestep]
        LOG.info(f"Reading data for granule {timestep:%Y-%m-%d}.")
        for the_band in self.band_map:
            rho = self._read_band_data('rho', the_band, s2_files)
            if rho is None:
                return (None,) * 6
            rho_surface.append(rho)

            unc = self._read_band_data('unc', the_band, s2_files)
            if unc is None:
                return (None,) * 6
            rho_unc.append(unc)
        # For reference...
        # bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08',
        #         'B8A', 'B09', 'B10','B11', 'B12']
        # b_ind = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        rho_surface = np.array(rho_surface)

        # Ensure all surface reflectance pixels have values above 0 and
        # aren't cloudy. So pixels are valid if all refl > 0 AND mask is True
        # Array of the desired bands. Not necessarily contiguous.
        sel_bands = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        mask1 = np.logical_and(
            np.all(rho_surface[sel_bands] > 0, axis=0), mask
        )
        mask = mask1
        if mask.sum() == 0:
            LOG.info(f"{timestep} -> No clear observations")
            return (None,) * 6
        LOG.info(
            f"{timestep} -> Total of {mask.sum():d} clear pixels "
            f"({100.*mask.sum()/np.prod(mask.shape):f}%)"
        )

        # Rescaling and setting missing pixels to NaN
        # reflectivity
        rho_surface = rho_surface / 10000.0
        rho_surface[:, ~mask] = np.nan
        # uncertainty
        rho_unc = np.array(rho_unc) / 10000.0
        rho_unc[:, ~mask] = np.nan
        # Average uncertainty over the image
        rho_unc = np.nanmean(rho_unc, axis=(1, 2))

        # Read in angles
        try:
            sun_angles = self._read_angle_data('SAA_SZA', current_folder)
            view_angles = self._read_angle_data('VAA_VZA_B05', current_folder)
        except ValueError as err:
            LOG.info(f"Sun or View angle {err}")
            return (None,) * 6
        raa = np.cos(np.deg2rad(view_angles[1] - sun_angles[1]))

        return rho_surface, mask, sun_angles[0], view_angles[0], raa, rho_unc

    def _read_band_data(self, type_data, band, s2_files):
        """
        Reads different types of data for the given band from the list of
        input files.

        Parameters
        ----------
        type_data : str
            the type of data to read. Currently 'rho' for reflectivity or
            'unc' for uncertainty
        band : str
            the name of the band
        s2_files : list
            a list of files (paths) containing the data for all bands
            of one timestep.
        Returns
        -------
        numpy array
            contains the (reflectivity or uncertainty) data for one band
        """

        meta_dict = {'rho': ('', ''),
                     'unc': ('_unc', 'uncertainty')}

        try:
            original_s2_file = next(
                f for f in s2_files
                if f.name.endswith(f"{band}_sur{meta_dict[type_data][0]}.tif")
            )
        except KeyError:
            LOG.info(f"Only reflectivity ('rho') or uncertainty ('unc') data "
                     f"read by _read_band_data. Calling argument {type_data} "
                     f"not recognised.")
            raise
        except StopIteration:
            LOG.info(f"Reflectivity {meta_dict[type_data][1]} file name for "
                     f"band {band} does not exist in the granule xml file.")
            return None
        if not original_s2_file.exists():
            LOG.info(f"Reflectivity {meta_dict[type_data][1]} file for band "
                     f"{band} does not exist.")
            return None
        LOG.debug(f"Original {meta_dict[type_data][1]} file "
                  f"{original_s2_file}")
        data = reproject_data(
            str(original_s2_file), target_img=self.state_mask
        ).ReadAsArray()

        return data

    def _read_angle_data(self, file_type, current_folder):
        """
        Reads different types of angle data contained somewhere in the file
        tree below current_folder.

        Parameters
        ----------
        file_type : str
            the type of angle data to read. This is the file name without the
            'tif' extension.
        current_folder : path
            the path to where all the data is contained
        Returns
        -------
        numpy arrays
            two arrays, containing the zenith and azimuth angle data
        """

        try:
            angle_file = next(current_folder.glob(f"**/{file_type}.tif"))
        except StopIteration:
            raise ValueError(f"file {file_type}.tif does not exist.")

        angles = reproject_data(
            str(angle_file),
            target_img=self.state_mask,
            x_res=20,
            y_res=20,
            resample=0,
        ).ReadAsArray()
        # zenith (90 - altitude)
        za = np.cos(np.deg2rad(angles[1].mean() / 100.0))
        # azimuth
        aa = angles[0].mean() / 100.0

        return za, aa

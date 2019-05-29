#!/usr/bin/env python
"""Dealing with S2 observations"""


import datetime as dt
import logging
from collections import namedtuple
from pathlib import Path

import gdal
import numpy as np
import osr

from TwoNN import Two_NN

gdal.UseExceptions()

LOG = logging.getLogger(__name__ + ".Sentinel2_Observations")

# A SIAC data storage type
S2MSIdata = namedtuple(
    "S2MSIdata", "time observations uncertainty mask metadata emulator"
)


class reproject_data(object):
    """
    A class that uses a source and a target images to
    reproject & clip the source image to match the extent,
    projection and resolution of the target image.

    """

    def __init__(
        self,
        source_img,
        target_img=None,
        dstSRS=None,
        srcNodata=np.nan,
        dstNodata=np.nan,
        outputType=None,
        verbose=False,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        xRes=None,
        yRes=None,
        xSize=None,
        ySize=None,
        resample=1,
    ):

        self.source_img = source_img
        self.target_img = target_img
        self.verbose = verbose
        self.dstSRS = dstSRS
        self.srcNodata = srcNodata
        self.dstNodata = dstNodata
        self.outputType = (
            gdal.GDT_Unknown if outputType is None else outputType
        )
        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax
        self.xRes = xRes
        self.yRes = yRes
        self.xSize = xSize
        self.ySize = ySize
        self.resample = resample
        if self.srcNodata is None:
            try:
                self.srcNodata = " ".join(
                    [
                        i.split("=")[1]
                        for i in gdal.Info(self.source_img).split("\n")
                        if " NoData" in i
                    ]
                )
            except RuntimeError:
                self.srcNodata = None
        if (self.target_img is None) & (self.dstSRS is None):
            raise IOError(
                "Projection should be specified ether from "
                + "a file or a projection code."
            )
        elif self.target_img is not None:
            try:
                g = gdal.Open(self.target_img)
            except RuntimeError:
                g = target_img
            geo_t = g.GetGeoTransform()
            x_size, y_size = g.RasterXSize, g.RasterYSize

            if self.xRes is None:
                self.xRes = abs(geo_t[1])
            if self.yRes is None:
                self.yRes = abs(geo_t[5])

            if self.xSize is not None:
                x_size = 1.0 * self.xSize * self.xRes / abs(geo_t[1])
            if self.ySize is not None:
                y_size = 1.0 * self.ySize * self.yRes / abs(geo_t[5])

            xmin, xmax = (
                min(geo_t[0], geo_t[0] + x_size * geo_t[1]),
                max(geo_t[0], geo_t[0] + x_size * geo_t[1]),
            )
            ymin, ymax = (
                min(geo_t[3], geo_t[3] + y_size * geo_t[5]),
                max(geo_t[3], geo_t[3] + y_size * geo_t[5]),
            )
            dstSRS = osr.SpatialReference()
            raster_wkt = g.GetProjection()
            dstSRS.ImportFromWkt(raster_wkt)
            self.g = gdal.Warp(
                "",
                self.source_img,
                format="MEM",
                outputBounds=[xmin, ymin, xmax, ymax],
                dstNodata=self.dstNodata,
                warpOptions=["NUM_THREADS=ALL_CPUS"],
                xRes=self.xRes,
                yRes=self.yRes,
                dstSRS=dstSRS,
                outputType=self.outputType,
                srcNodata=self.srcNodata,
                resampleAlg=self.resample,
            )

        else:
            self.g = gdal.Warp(
                "",
                self.source_img,
                format="MEM",
                outputBounds=[self.xmin, self.ymin, self.xmax, self.ymax],
                xRes=self.xRes,
                yRes=self.yRes,
                dstSRS=self.dstSRS,
                warpOptions=["NUM_THREADS=ALL_CPUS"],
                copyMetadata=True,
                outputType=self.outputType,
                dstNodata=self.dstNodata,
                srcNodata=self.srcNodata,
                resampleAlg=self.resample,
            )
        if self.g.RasterCount <= 3:
            self.data = self.g.ReadAsArray()
            # return self.data
        elif self.verbose:
            print(
                "There are %d bands in this file, use "
                + "g.GetRasterBand(<band>) to avoid reading the whole file."
                % self.g.RasterCount
            )


class Sentinel2Observations(object):
    def __init__(
        self,
        parent_folder,
        emulator_file,
        state_mask,
        band_prob_threshold=20,
        chunk=None,
        time_grid=None
    ):
        self.band_prob_threshold = band_prob_threshold
        parent_folder = Path(parent_folder)
        emulator_file = Path(emulator_file)
        if not parent_folder.exists():
            LOG.info(f"S2 data folder: {parent_folder}")
            raise IOError("S2 data folder doesn't exist")

        if not emulator_file.exists():
            LOG.info(f"Emulator file: {emulator_file}")
            raise IOError("Emulator file doesn't exist")
        self.band_map = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                        'B08', 'B8A', 'B09', 'B10','B11', 'B12']
        # self.band_map = ['05', '08']

        self.parent = parent_folder
        self.emulator_folder = emulator_file
        self.original_mask = state_mask
        self.state_mask = state_mask
        self._find_granules(self.parent, time_grid)
        f = np.load(str(emulator_file))
        self.emulator = Two_NN(
            Hidden_Layers=f.f.Hidden_Layers, Output_Layers=f.f.Output_Layers
        )
        self.chunk = chunk

    def apply_roi(self, ulx, uly, lrx, lry):
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
        try:
            g = gdal.Open(self.state_mask)
            proj = g.GetProjection()
            geoT = np.array(g.GetGeoTransform())
        except RuntimeError:
            proj = self.state_mask.GetProjection()
            geoT = np.array(self.state_mask.GetGeoTransform())

        # new_geoT = geoT*1.
        # new_geoT[0] = new_geoT[0] + self.ulx*new_geoT[1]
        # new_geoT[3] = new_geoT[3] + self.uly*new_geoT[5]
        return proj, geoT.tolist()  # new_geoT.tolist()

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
        dates = sorted(dates)
        if time_grid is not None:
            start_date = time_grid[0]
            end_date = time_grid[-1]
            self.dates = [d.replace(hour=0, minute=0, second=0) for d in dates
                            if (d >= start_date) and (d <= end_date)] 
        else:
            self.dates = [x.replace(hour=0,minute=0, second=0) for x in dates]
        self.date_data = dict(zip(self.dates, [f.parent for f in test_files]))
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
        """NOTE: Currently reads in sequentially. It's better to gather
        all the filenames and read them in parallel using parmap.py"""
        current_folder = self.date_data[timestep]

        fname_prefix = [
            f.name.split("B02")[0] for f in current_folder.glob("*B02_sur.tif")
        ][0]
        cloud_mask = current_folder.parent / f"cloud.tif"
        cloud_mask = reproject_data(
            str(cloud_mask), target_img=self.state_mask
        ).data
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
            ).data
            mask = mask * (rho > 0)
            rho_surface.append(rho)
            original_s2_file = current_folder / (
                f"{fname_prefix:s}" + f"{the_band:s}_sur_unc.tif"
            )
            LOG.debug(f"Uncertainty file {str(original_s2_file):s}")
            unc = reproject_data(
                str(original_s2_file), target_img=self.state_mask
            ).data
            rho_unc.append(unc)

        rho_unc = np.array(rho_unc) / 10000.0
        rho_unc[:, ~mask] = np.nan
        # Average uncertainty over the image
        rho_unc = np.nanmean(rho_unc, axis=(1, 2))
        rho_surface = np.array(rho_surface) / 10000.0
        rho_surface[:, ~mask] = np.nan
        if mask.sum() == 0:
            LOG.info("No clear observations")
            return None, None, None, None, None, None
        LOG.info(
            f"Total of {mask.sum():d} clear pixels "
            + f"({100.*mask.sum()/np.prod(mask.shape):f}%)"
        )
        # Now read angles
        sun_angles = reproject_data(
            str(current_folder.parent / "ANG_DATA/SAA_SZA.tif"),
            target_img=self.state_mask,
            xRes=20,
            yRes=20,
            resample=0,
        ).data
        view_angles = reproject_data(
            str(current_folder.parent / "ANG_DATA/VAA_VZA_B05.tif"),
            target_img=self.state_mask,
            xRes=20,
            yRes=20,
            resample=0,
        ).data
        sza = np.cos(np.deg2rad(sun_angles[1].mean() / 100.0))
        vza = np.cos(np.deg2rad(view_angles[1].mean() / 100.0))
        saa = sun_angles[0].mean() / 100.0
        vaa = view_angles[0].mean() / 100.0
        raa = np.cos(np.deg2rad(vaa - saa))

        return rho_surface, mask, sza, vza, raa, rho_unc


if __name__ == "__main__":
    time_grid = []
    today = dt.datetime(2017,1,1)
    while (today <= dt.datetime(2017, 12, 31)):
        time_grid.append(today)
        today += dt.timedelta(days=5)

    s2_obs = Sentinel2Observations(
        "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/s2_obs/",
        "/home/ucfafyi/DATA/Prosail/prosail_2NN.npz",
        "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/ESU.tif",
        band_prob_threshold=20,
        chunk=None,
        time_grid=time_grid
    )
    retval = s2_obs.read_time_series([dt.datetime(2017, 1, 1),
                                      dt.datetime(2017,12,31)])

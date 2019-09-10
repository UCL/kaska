#!/usr/bin/env python
"""The main Kaska external interface. This module provides a way to
configure and execute Kaska, including spreading the processing by
spatial tiles and distributing over some dask-aware cluster.
"""

import copy
import logging
import shutil
from copy import deepcopy
import datetime as dt
from functools import partial
from pathlib import Path
from collections import namedtuple

import numpy as np

from osgeo import gdal

from .utils import get_chunks
from .s2_observations import Sentinel2Observations
from .kaska import KaSKA
from .kaska import define_temporal_grid

Config = namedtuple(
    "Config", "s2_obs temporal_grid state_mask inverter output_folder"
)

LOG = logging.getLogger(__name__)


def stitch_outputs(output_folder, parameter_list):
    """Given a folder with some contiguous files in GeoTIFF format, this
    function will stitch them together using their georeference information.
    There's a fairly strong assumption that these tiles have already been
    reasonably created by e.g. `process_tile` below. It is not a problem to
    have missing tiles that weren't processed (this sometimes happens if
    the state mask shows no useable pixels). We assume the relevant files
    have the following filename structure 
    `[parameter]_A[year][DoY]_0x[tile].tif`, where `parameter` is the 
    parameter name, `year` is e.g. `2019` `DoY` is day of year and tile
    is a hexadecimal number with the tile number.
    
    Parameters
    ----------
    output_folder : str
        A folder
    parameter_list : list
        A list of parameters. This is used to search individual tiles by
        filename (e.g. `LAI_A2019135_0xfd.tif`).
    
    Returns
    -------
    list
        A list of the stitched up files.
    """
    # Get the output folder
    p = Path(output_folder)
    # Loop over parameters and find all the files for all the
    # chunks and dates
    output_tiffs = {}
    for parameter in parameter_list:
        files = sorted([fich for fich in p.glob(f"*{parameter:s}_A*_0x*.tif")])
        dates = sorted(
            list(
                set(
                    [
                        fich.stem.split(parameter)[1].split("_")[1]
                        for fich in files
                    ]
                )
            )
        )
        fnames = []
        # Now for each data, stitch up all the chunks for that parameter
        for date in dates:

            sel_files = [
                fich.as_posix() for fich in files if fich.stem.find(date) >= 0
            ]
            dst_ds = gdal.BuildVRT(
                (p / f"{parameter:s}_{date:s}.vrt").as_posix(), sel_files
            )
            fnames.append(dst_ds.GetDescription())
            dst_ds = None

        # Potentially, create a multiband VRT/GTiff with all the dates?
        dst_ds = gdal.BuildVRT(
            (p / f"{parameter:s}.vrt").as_posix(),
            fnames,
            options=gdal.BuildVRTOptions(separate=True),
        )
        dst_ds = None
        dst_ds = gdal.Translate(
            (p / f"{parameter:s}.tif").as_posix(),
            (p / f"{parameter:s}.vrt").as_posix(),
            options=gdal.TranslateOptions(
                format="GTiff",
                creationOptions=[
                    "TILED=YES",
                    "INTERLEAVE=BAND",
                    "COMPRESS=LZW",
                    "COPY_SRC_OVERVIEWS=YES",
                ],
            ),
        )
        for band in range(1, dst_ds.RasterCount + 1):
            dst_ds.GetRasterBand(band).SetMetadata({"DoY": dates[band - 1][1:]})
        output_tiffs[parameter] = dst_ds.GetDescription()
        dst_ds = None
        g = gdal.Open((p / f"{parameter:s}.tif").as_posix(), gdal.GA_Update)
        g.BuildOverviews("average", np.power(2, np.arange(8)))
        g = None
        g = gdal.Translate(
            (p / "temporary.tif").as_posix(),
            (p / f"{parameter:s}.tif").as_posix(),
            format="GTiff",
            creationOptions=[
                "TILED=YES",
                "INTERLEAVE=BAND",
                "COMPRESS=LZW",
                "COPY_SRC_OVERVIEWS=YES",
            ],
        )
        shutil.move(p / "temporary.tif", (p / f"{parameter:s}.tif").as_posix())

        LOG.info(f"Saved {parameter:s} file as {output_tiffs[parameter]:s}")
    return output_tiffs


def process_tile(the_chunk, config):
    """A function to process a single spatial tile. The function
    receives a `chunk` object, and a configuration object.
    
    Parameters
    ----------
    the_chunk : iter
        A list, tuple or whatever with the top left X and Y coordinates
        in pixels, the X and Y number of pixels of the tile, and the
        tile number.
    config : Config
        A configuration object.
    
    Returns
    -------
    list
        A list of retrieved parameters.
    """
    # Unpack chunck object with UL pixel coordinates,
    # number of pixels in tile and chunk number
    this_X, this_Y, nx_valid, ny_valid, chunk_no = the_chunk
    ulx = this_X
    uly = this_Y
    lrx = this_X + nx_valid
    lry = this_Y + ny_valid
    # copy the observations in case we have issues with
    # references hanging around...
    # Apply the region of interest to the observations
    s2_obs = copy.copy(config.s2_obs)
    s2_obs.apply_roi(ulx, uly, lrx, lry)

    # Define KaSKA object with windowed observations.
    kaska = KaSKA(
        s2_obs,
        config.temporal_grid,
        config.state_mask,
        config.inverter,
        config.output_folder,
        chunk=hex(chunk_no),
    )
    parameter_names, parameter_data = kaska.run_retrieval()
    kaska.save_s2_output(parameter_names, parameter_data)
    return parameter_names


def kaska_runner(
    start_date,
    end_date,
    temporal_grid_space,
    state_mask,
    s2_folder,
    approx_inverter,
    s2_emulator,
    output_folder,
    dask_client=None,
):
    """Runs a KaSKA problem for S2 producing parameter estimates between
    `start_date` and `end_date` with a temporal spacing `temporal_grid_space`.
    
    
    Parameters
    ----------
    start_date : datetime object
        Starting date for the inference
    end_date : datetime object
        End date for the inference
    temporal_grid_space : datetime object
        Temporal resolution of the inference (in days).
    state_mask : str
        An existing spatial raster with the state mask (binary mask detailing
        which pixels to process).
    s2_folder : str
        Folder where the Sentinel2 data reside.
    approx_inverter : str
        The inverter filename
    s2_emulator : str
        The emulator filename
    output_folder : str
        A folder where the output files will be dumped.
    dask_client : dask, optional
        Allows the distribution of the processing using a dask distributed
        cluster. If this is None, then the processing is run tiled but 
        sequentially.
    
    Returns
    -------
    list
        A list of the processed parameters files.
    """
    temporal_grid = define_temporal_grid(
        start_date, end_date, temporal_grid_space
    )
    s2_obs = Sentinel2Observations(
        s2_folder,
        s2_emulator,
        state_mask,
        band_prob_threshold=20,
        time_grid=temporal_grid,
    )

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    # "s2_obs temporal_grid state_mask inverter output_folder"
    config = Config(
        s2_obs, temporal_grid, state_mask, approx_inverter, output_folder
    )
    # Avoid reading mask in memory in case we fill it up
    g = gdal.Open(state_mask)
    ny, nx = g.RasterYSize, g.RasterXSize

    # Do the splitting
    them_chunks = (the_chunk for the_chunk in get_chunks(nx, ny))

    wrapper = partial(process_tile, config=config)
    if dask_client is None:
        retval = list(map(wrapper, them_chunks))
    else:
        A = dask_client.map(wrapper, them_chunks)
        retval = dask_client.gather(A)

    parameter_names = retval[0]

    return stitch_outputs(output_folder, parameter_names)

#!/usr/bin/env python
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

from .utils import  get_chunks
from .s2_observations import Sentinel2Observations
from .kaska import KaSKA, get_chunks
from .kaska import define_temporal_grid

Config = namedtuple("Config",
                    "s2_obs temporal_grid state_mask inverter output_folder")

LOG = logging.getLogger(__name__ + ".KaSKA")
LOG.setLevel(logging.DEBUG)
if not LOG.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - ' +
                                  '%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    LOG.addHandler(ch)
LOG.propagate = False

def stitch_outputs(output_folder, parameter_list):
    """Stitch outputs"""
    # Get the output folder
    p = Path(output_folder)
    # Loop over parameters and find all the files for all the 
    # chunks and dates
    output_tiffs = {}
    for parameter in parameter_list:
        files = sorted([fich for fich in p.glob(f"*{parameter:s}_A*_0x*.tif")])
        dates = sorted(list(set([fich.stem.split(parameter)[1].split("_")[1] 
                 for fich in files])))
        fnames = []
        # Now for each data, stitch up all the chunks for that parameter
        for date in dates:
            
            sel_files = [fich.as_posix() 
                         for fich in files if fich.stem.find(date) >= 0 ]
            dst_ds = gdal.BuildVRT((p/f"{parameter:s}_{date:s}.vrt").as_posix(),
                                sel_files)
            fnames.append(dst_ds.GetDescription())
            dst_ds = None
        
        # Potentially, create a multiband VRT/GTiff with all the dates?
        dst_ds = gdal.BuildVRT((p/f"{parameter:s}.vrt").as_posix(),
                               fnames,options=gdal.BuildVRTOptions(separate=True))
        dst_ds = None
        dst_ds = gdal.Translate((p/f"{parameter:s}.tif").as_posix(),
                                (p/f"{parameter:s}.vrt").as_posix(),
                                options=gdal.TranslateOptions(format="GTiff",
                                                             creationOptions=["TILED=YES",
                                                                            "INTERLEAVE=BAND",
                                                                            "COMPRESS=LZW",
                                                                            "COPY_SRC_OVERVIEWS=YES"
                                                                            ]))
        for band in range(1, dst_ds.RasterCount + 1):
            dst_ds.GetRasterBand(band).SetMetadata({"DoY":dates[band-1][1:]})
        output_tiffs[parameter] = dst_ds.GetDescription()
        dst_ds = None
        g = gdal.Open((p/f"{parameter:s}.tif").as_posix(),gdal.GA_Update )
        g.BuildOverviews("average", np.power(2, np.arange(8)))
        g = None
        g = gdal.Translate((p/"temporary.tif").as_posix(),
                           (p/f"{parameter:s}.tif").as_posix(), 
                            format="GTiff", creationOptions=["TILED=YES",
                                                            "INTERLEAVE=BAND",
                                                            "COMPRESS=LZW",
                                                            "COPY_SRC_OVERVIEWS=YES"])
        shutil.move(p/"temporary.tif", (p/f"{parameter:s}.tif").as_posix())
                            
        LOG.info(f"Saved {parameter:s} file as {output_tiffs[parameter]:s}")
    return output_tiffs


def process_tile(the_chunk,config):
    this_X, this_Y, nx_valid, ny_valid, chunk_no = the_chunk
    ulx = this_X
    uly = this_Y
    lrx = this_X + nx_valid
    lry = this_Y + ny_valid
    # copy the observations in case we have issues with
    # references hanging around...
    s2_obs = copy.copy(config.s2_obs)
    s2_obs.apply_roi(ulx, uly, lrx, lry)
    
    kaska = KaSKA(s2_obs, config.temporal_grid, config.state_mask,
                  config.inverter, config.output_folder, 
                  chunk=hex(chunk_no))
    parameter_names, parameter_data = kaska.run_retrieval()
    kaska.save_s2_output(parameter_names, parameter_data)
    return parameter_names




def kaska_setup(start_date, end_date, temporal_grid_space, state_mask,
                s2_folder, approx_inverter, s2_emulator, output_folder,
                dask_client=None):

    temporal_grid = define_temporal_grid(start_date, end_date,
                                            temporal_grid_space)
    s2_obs = Sentinel2Observations(
        s2_folder,
        s2_emulator,
        state_mask,
        band_prob_threshold=20,
        time_grid=temporal_grid,
    )

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    #"s2_obs temporal_grid state_mask inverter output_folder"
    config = Config(s2_obs, temporal_grid, state_mask, 
                    approx_inverter, output_folder)
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

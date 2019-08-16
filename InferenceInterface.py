#!/usr/bin/env python
import copy
import logging
import shutil
from copy import deepcopy
from functools import partial
from pathlib import Path
from collections import namedtuple

from osgeo import gdal


#from .utils import  get_chunks

LOG = logging.getLogger(__name__+".InferenceInterface")

Config = namedtuple("Config",
                    "s2_obs temporal_grid state_mask inverter output_folder")

import numpy as np
import gdal
from pathlib import Path
import logging

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


def chunk_inference(roi, prefix, current_mask, configuration):

    [ulx, uly, lrx, lry] = roi
    
    configuration.observations.apply_roi(ulx, uly, lrx, lry)
    projection, geotransform = configuration.observations.define_output()
    output = KafkaOutput(configuration.parameter_list, 
                         geotransform, projection,
                         configuration.output_folder,
                         prefix=prefix)

    #Q = np.array([100., 1e5, 1e2, 100., 1e5, 1e2, 100.])
    ## state_propagator = IdentityPropagator(Q, 7, mask)
    the_prior = configuration.prior(configuration.parameter_list,
                                    current_mask)
    state_propagator = deepcopy(configuration.propagator)
    state_propagator.mask = current_mask
    state_propagator.n_elements = current_mask.sum()
    kf = LinearKalman(configuration.observations, 
                      output, current_mask, 
                      configuration.observation_operator_creator,
                      configuration.parameter_list,
                      state_propagation=state_propagator.get_matrices,
                      prior=the_prior, 
                      band_mapper=configuration.band_mapper,
                      linear=False, upper_bound=configuration.upper_bounds,
                      lower_bound=configuration.lower_bounds)

    # Get starting state... We can request the prior object for this
    x_forecast, P_forecast_inv = the_prior.process_prior(None)
        
    
    kf.run(configuration.time_grid, x_forecast, None, P_forecast_inv,
           iter_obs_op=True, is_robust=False)


def chunk_wrapper(the_chunk, config):
    """[summary]
    
    Parameters
    ----------
    the_chunk : [type]
        [description]
    config : [type]
        [description]
    
    Returns
    -------
    [type]
        [description]
    """
    this_X, this_Y, nx_valid, ny_valid, chunk = the_chunk
    ulx = this_X
    uly = this_Y
    lrx = this_X + nx_valid
    lry = this_Y + ny_valid
    
    roi = [ulx, uly, lrx, lry]

    if config.mask[this_Y:(this_Y+ny_valid), this_X:(this_X+nx_valid)].sum() > 0:   
        print("Running chunk %s" % ( hex(chunk)))
        chunk_inference(roi, hex(chunk), 
                        config.mask[this_Y:(this_Y+ny_valid),
                                    this_X:(this_X+nx_valid)],
                        config)
        return hex(chunk)
            


def kafka_inference(mask, time_grid, parameter_list,
                    observations, prior, propagator,
                    output_folder, band_mapper, dask_client,
                    observation_operator_creator,
                    chunk_size=[64, 64], upper_bounds=None,
                    lower_bounds=None):
    
    # First, put the configuration in its own object to minimise
    # variable transport
    
    Config = namedtuple("Config", ["mask", "time_grid", "parameter_list",
                                   "observations", 
                                   "observation_operator_creator",
                                   "prior", "propagator",
                                   "output_folder", "band_mapper",
                                   "upper_bounds", "lower_bounds"])    
    config = Config(mask, time_grid, parameter_list, observations,
                    observation_operator_creator,
                    prior, propagator, output_folder, band_mapper,
                    upper_bounds, lower_bounds)
    ny, nx= mask.shape
    
    them_chunks = [the_chunk for the_chunk in qhunks(nx, ny,
                    block_size= chunk_size)]
    
    wrapper = partial(chunk_wrapper, config=config)

    if dask_client is None:
        chunk_names = list(map(wrapper, them_chunks))
    else:
        A = dask_client.map (wrapper, them_chunks)
        retval = dask_client.gather(A)
    
    return stitch_outputs(output_folder, parameter_list)



def process_tile(this_X, this_Y, nx_valid, ny_valid, chunk_no,
                 config):
    ulx = this_X
    uly = this_Y
    lrx = this_X + nx_valid
    lry = this_Y + ny_valid
    s2_obs = copy.copy(config.s2_obs)
    s2_obs.apply_roi(ulx, uly, lrx, lry)
    kaska = KaSKA(s2_obs, config.temporal_grid, config.state_mask,
                  config.inverter, config.output_folder, 
                  chunk=hex(chunk_no))
    parameter_names, parameter_data = kaska.run_retrieval()
    kaska.save_s2_output(parameter_names, parameter_data)


if __name__ == "__main__":
    import datetime as dt
    from kaska import Sentinel2Observations
    from kaska import KaSKA, get_chunks
    from kaska import define_temporal_grid

def kaska_setup(start_date, end_date, temporal_grid_space,
                s2_folder, approx_inverter, emulator, output_folder):

    temporal_grid = define_temporal_grid(start_date, end_date,
                                            temporal_grid_space)
    s2_obs = Sentinel2Observations(
        s2_folder,
        s2_emulator,
        state_mask,
        band_prob_threshold=20,
        time_grid=temporal_grid,
    )

    state_mask = "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/ESU.tif"
    approx_inverter = "/home/ucfafyi/DATA/Prosail/Prosail_5_paras.h5"
    output_folder = Path(output_folder).mkdir(parents=True, exist_ok=True)
    config = Config(s2_obs, temporal_grid, state_mask, 
                    approx_inverter, output_folder)
    # Avoid reading mask in memory in case we fill it up 
    g = gdal.Open(state_mask)
    ny, nx = g.RasterYSize, g.RasterXSize

    them_chunks = [the_chunk for the_chunk in get_chunks(nx, ny)]

    for [this_X, this_Y, nx_valid, ny_valid, chunk_no] in get_chunks(nx, ny):
    stitch_outputs("/tmp/", ["lai", "cab", "cbrown"])


    
    

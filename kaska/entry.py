#!/usr/bin/env python
import datetime as dt
from .logger import create_logger
from .inference_runner import kaska_runner
from .inverters import get_emulator, get_inverter


def run_process(start_date, end_date, temporal_grid_space, s2_folder,
                s1_ncfile, state_mask, prior_dist, output_folder, debug=True,
                logfile=None, dask_client=None, block_size=[256, 256],
                chunk=None):
    """This is the entry point function that should be called by any
    script wishing to run KaSKA. It runs a KaSKA problem for S2 producing
    parameter estimates between `start_date` and `end_date` with a temporal
    spacing `temporal_grid_space`.

    Parameters
    ----------
    start_date : datetime object
        Starting date for the inference
    end_date : datetime object
        End date for the inference
    temporal_grid_space : datetime object
        Temporal resolution of the inference (in days).
    s2_folder : str
        Folder where the Sentinel2 data reside.
    s1_ncfile: str
        NetCDF file containing the Sentinel 1 data
    state_mask : str
        An existing spatial raster with the state mask (binary mask detailing
        which pixels to process).
    output_folder : str
        A folder where the output files will be dumped.
    debug : bool, optional
        Flag for controlling debug logging.
    logfile : str, optional
        The name of the log file.
    dask_client : dask, optional
        Allows the distribution of the processing using a dask distributed
        cluster. If this is None, then the processing is run tiled but
        sequentially.
    block_size : int list[2], optional
        The size of the tile to break the image into (in pixels).
    chunk: int, optional
        If a single chunk is expected to be processed, pass its number here.

    """

    # Setup logger and log run info
    if logfile is None:
        logfile = f"KaSKA_{dt.datetime.now():%Y%M%d_%H%M}.log"
    LOG = create_logger(debug=debug, fname=logfile)
    LOG.info("Running KaSKA with arguments:")
    LOG.info("start date : "+start_date.strftime('%Y%m%d'))
    LOG.info("end date : "+end_date.strftime('%Y%m%d'))
    LOG.info("temporal grid spacing (days): "+str(temporal_grid_space))
    LOG.info("data folder : "+s2_folder)
    LOG.info("state mask : "+state_mask)
    LOG.info("output folder : "+output_folder)
    LOG.info("debug logging : "+("ON" if debug else "OFF"))
    LOG.info("block size : "+str(block_size[0])+"x"+str(block_size[1]))

    # Prepare arguments needed to run main kaska executable
    s2_emulator = get_emulator("prosail", "Sentinel2")
    approx_inverter = get_inverter("prosail_5paras", "Sentinel2")

    kaska_runner(start_date, end_date, temporal_grid_space, state_mask,
                 s2_folder, approx_inverter, s2_emulator,
                 s1_ncfile,
                 prior_dist,
                 output_folder,
                 dask_client=dask_client, block_size=block_size,
                 chunk=chunk)

#!/usr/bin/env python
"""
Entry point for running the Kaska code.
"""

import datetime as dt
from .logger import create_logger
from .inference_runner import kaska_runner
from .inverters import get_emulator, get_inverter


# pylint: disable-msg=too-many-arguments
# I don't think it's worthwhile refactoring this entry code.
# If necessary, we can create a new object class to store the values,
# and then pass that as the sole arg.

def run_process(start_date, end_date, temporal_grid_space, s2_folder,
                s1_ncfile, state_mask, output_folder, debug=True,
                logfile=None, dask_client=None,
                block_size=None,
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
    # Fix pylint "W0102: Dangerous default value [] as argument" warning
    if block_size is None:
        block_size = [256, 256]

    # Setup logger and log run info
    if logfile is None:
        logfile = f"KaSKA_{dt.datetime.now():%Y%M%d_%H%M}.log"
    log = create_logger(debug=debug, fname=logfile)
    log.info("Running KaSKA with arguments:")
    log.info("start date : %s", start_date.strftime('%Y%m%d'))
    log.info("end date : %s", end_date.strftime('%Y%m%d'))
    log.info("temporal grid spacing (days): %f", temporal_grid_space)
    log.info("data folder : %s", s2_folder)
    log.info("state mask : %s", state_mask)
    log.info("output folder : %s", output_folder)
    log.info("debug logging : %s", "ON" if debug else "OFF")
    log.info("block size : %dx%d", block_size[0], block_size[1])

    # Prepare arguments needed to run main ßkaska executable
    s2_emulator = get_emulator("prosail", "Sentinel2")
    approx_inverter = get_inverter("prosail_5paras", "Sentinel2")

    kaska_runner(start_date, end_date, temporal_grid_space, state_mask,
                 s2_folder, approx_inverter, s2_emulator,
                 s1_ncfile,
                 output_folder,
                 dask_client=dask_client, block_size=block_size,
                 chunk=chunk)
# pylint: enable-msg=too-many-arguments

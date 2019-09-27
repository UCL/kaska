#!/usr/bin/env python
import datetime as dt
from .inference_runner import kaska_runner
from .logger import create_logger
from .inverters import get_emulator, get_inverter


def run_process(start_date, end_date, temporal_grid_space, s2_folder,
                state_mask, output_folder, debug=True, logfile=None,
                dask_client=None, block_size=[256, 256]):

    if logfile is None:
        logfile = f"KaSKA_{dt.datetime.now():%Y%M%d_%H%M}.log"
    LOG = create_logger(debug=debug, fname=logfile)
    LOG.info("Running KaSKA...")
    s2_emulator = get_emulator("prosail", "Sentinel2")
    approx_inverter = get_inverter("prosail_5paras", "Sentinel2")

    kaska_runner(start_date, end_date, temporal_grid_space, state_mask,
                 s2_folder, approx_inverter, s2_emulator, output_folder,
                 dask_client=dask_client, block_size=block_size)

#!/usr/bin/env python
import logging
import datetime as dt
from pathlib import Path

from kaska import kaska_setup
from kaska import define_temporal_grid
from kaska import get_emulators, get_emulator
from kaska import get_inverters, get_inverter


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

if __name__ == "__main__":

    start_date = dt.datetime(2017,5, 1)
    end_date = dt.datetime(2017,7, 1)
    temporal_grid_space = 5
    s2_folder = "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/s2_obs/"
    state_mask = "/home/ucfajlg/Data/python/KaFKA_Validation/LMU/carto/ESU.tif"
    output_folder = "/tmp/"
    s2_emulator = get_emulator("prosail", "Sentinel2")
    approx_inverter = get_inverter("prosail_5paras", "Sentinel2")
    
    kaska_setup(start_date, end_date, temporal_grid_space, state_mask,
                s2_folder, approx_inverter, s2_emulator, output_folder)
    

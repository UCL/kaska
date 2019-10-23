#!/usr/bin/env python
"""Test reading s2_observations data"""
import os

import sys
#sys.path.append("../")

import pytest
import datetime as dt
from pathlib import Path
#import gdal
#import numpy as np


TEST_PATH = os.path.dirname(__file__)

from ..s2_observations import Sentinel2Observations

def test_s2_data():
    """Test the reading of s2 time series data.
    """
    import pickle

    time_grid = []
    today = dt.datetime(2017,1,1)
    while (today <= dt.datetime(2017, 2, 1)):
        time_grid.append(today)
        today += dt.timedelta(days=5)

    parent = TEST_PATH
    emulator = TEST_PATH + "/../inverters/prosail_2NN.npz"
    mask = TEST_PATH + "/data/ESU.tif"
    #source = TEST_PATH +  "/data/s2_test_file.tif"
    s2_obs = Sentinel2Observations(
        parent,
        emulator,
        mask,
        band_prob_threshold=20,
        chunk=None,
        time_grid=time_grid
    )

    ref_dates = [
        dt.datetime(2017, 1, 4, 0, 0),
        dt.datetime(2017, 1, 7, 0, 0),
        dt.datetime(2017, 1, 14, 0, 0),
        dt.datetime(2017, 1, 24, 0, 0),
        dt.datetime(2017, 1, 27, 0, 0)
    ]
    ref_files = [
        Path(parent+"/data/s2_data/S2A_MSIL1C_20170104.SAFE"),
        Path(parent+"/data/s2_data/S2A_MSIL1C_20170107.SAFE"),
        Path(parent+"/data/s2_data/S2A_MSIL1C_20170114.SAFE"),
        Path(parent+"/data/s2_data/S2A_MSIL1C_20170124.SAFE"),
        Path(parent+"/data/s2_data/S2A_MSIL1C_20170127.SAFE")
    ]
    assert s2_obs.dates == ref_dates
    assert [s2_obs.date_data[d] for d in s2_obs.dates] == ref_files

    #retval = s2_obs.read_time_series([dt.datetime(2017, 1, 1),
    #                                  dt.datetime(2017,12,31)])
    #pickle.dumps(retval)

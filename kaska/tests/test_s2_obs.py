#!/usr/bin/env python
"""Test reading s2_observations data"""
import os

import sys

import pytest
import datetime as dt
from pathlib import Path
import numpy as np

from ..s2_observations import Sentinel2Observations

TEST_PATH = os.path.dirname(__file__)


def test_s2_data():
    """Test the reading of s2 time series data.
    """
    import pickle

    time_grid = []
    today = dt.datetime(2017, 1, 1)
    while (today <= dt.datetime(2017, 2, 1)):
        time_grid.append(today)
        today += dt.timedelta(days=5)

    parent = TEST_PATH
    emulator = TEST_PATH + "/../inverters/prosail_2NN.npz"
    mask = TEST_PATH + "/data/ESU.tif"
    # source = TEST_PATH +  "/data/s2_test_file.tif"
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
    ]
    ref_files = [
        Path(parent+"/data/s2_data/S2A_MSIL1C_20170104.SAFE/GRANULE/IMG_DATA"),
        Path(parent+"/data/s2_data/S2A_MSIL1C_20170107.SAFE/GRANULE/IMG_DATA"),
    ]
    assert sorted(s2_obs.date_data) == sorted(ref_dates)
    assert [s2_obs.date_data[d][0].parent for d in sorted(s2_obs.date_data)] == ref_files

    for i, d in enumerate(ref_dates):
        rho_surface, mask, sza, vza, raa, rho_unc = s2_obs.read_granule(d)
        # np.savez_compressed(f"s2_obs_{i}_ref", rho_surface=rho_surface,
        #                    mask=mask, sza=sza, vza=vza, raa=raa,
        #                    rho_unc=rho_unc)
        ref_data = np.load(Path(parent+f'/data/s2_obs_{i}_ref.npz'),
                           allow_pickle=True)
        np.testing.assert_equal(ref_data['rho_surface'], rho_surface)
        np.testing.assert_equal(ref_data['mask'], mask)
        np.testing.assert_equal(ref_data['sza'], sza)
        np.testing.assert_equal(ref_data['vza'], vza)
        np.testing.assert_equal(ref_data['raa'], raa)
        np.testing.assert_equal(ref_data['rho_unc'], rho_unc)

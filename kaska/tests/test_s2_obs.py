#!/usr/bin/env python
"""Test reading s2_observations data"""
import os

import sys
import shutil

import pytest
import datetime as dt
from pathlib import Path
import numpy as np

from ..s2_observations import Sentinel2Observations

TEST_PATH = os.path.dirname(__file__)


def test_s2_data():
    """Test the reading of s2 time series data.
    """

    time_grid = []
    today = dt.datetime(2017, 1, 1)
    while (today <= dt.datetime(2017, 2, 1)):
        time_grid.append(today)
        today += dt.timedelta(days=5)

    parent = TEST_PATH
    emulator = TEST_PATH + "/../inverters/prosail_2NN.npz"
    mask = TEST_PATH + "/data/ESU.tif"
    # Test Sentinel2Observations constructor
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

    # Test read_granule when all data files are available
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

    # Test read_granule for missing cloud mask file
    shutil.move(ref_files[1].parent / f"cloud.tif",
                ref_files[1].parent / f"cloud.tif.bac")
    rho_surface, mask, sza, vza, raa, rho_unc = s2_obs.read_granule(ref_dates[0])
    assert rho_surface is None
    assert rho_unc is None
    assert mask is None
    assert sza is None
    assert vza is None
    assert raa is None
    shutil.move(ref_files[1].parent / f"cloud.tif.bac",
                ref_files[1].parent / f"cloud.tif")

    # Test read_granule for reflectivity file missing in the dictionary
    file_list = list(s2_obs.date_data[ref_dates[1]])
    file0 = s2_obs.date_data[ref_dates[1]].pop(0)
    rho_surface, mask, sza, vza, raa, rho_unc = s2_obs.read_granule(ref_dates[1])
    assert rho_surface is None
    assert rho_unc is None
    assert mask is None
    assert sza is None
    assert vza is None
    assert raa is None
    s2_obs.date_data[ref_dates[1]] = file_list
    # Test read_granule for reflectivity file missing in the folder
    shutil.move(file0, file0.with_suffix(".bac"))
    rho_surface, mask, sza, vza, raa, rho_unc = s2_obs.read_granule(ref_dates[1])
    assert rho_surface is None
    assert rho_unc is None
    assert mask is None
    assert sza is None
    assert vza is None
    assert raa is None
    shutil.move(file0.with_suffix(".bac"), file0)

    # Test read_granule for reflectivity uncertainty file missing in the dictionary
    file1 = s2_obs.date_data[ref_dates[1]].pop(1)
    rho_surface, mask, sza, vza, raa, rho_unc = s2_obs.read_granule(ref_dates[1])
    assert rho_surface is None
    assert rho_unc is None
    assert mask is None
    assert sza is None
    assert vza is None
    assert raa is None
    s2_obs.date_data[ref_dates[1]] = file_list
   # Test read_granule for reflectivity uncertainty file missing in the folder
    shutil.move(file1, file1.with_suffix(".bac"))
    rho_surface, mask, sza, vza, raa, rho_unc = s2_obs.read_granule(ref_dates[1])
    assert rho_surface is None
    assert rho_unc is None
    assert mask is None
    assert sza is None
    assert vza is None
    assert raa is None
    shutil.move(file1.with_suffix(".bac"), file1)

   # Test read_granule for sun angle file missing
    shutil.move(ref_files[1].parent / "ANG_DATA/SAA_SZA.tif",
                ref_files[1].parent / "ANG_DATA/SAA_SZA.tif.bac")
    rho_surface, mask, sza, vza, raa, rho_unc = s2_obs.read_granule(ref_dates[1])
    assert rho_surface is None
    assert rho_unc is None
    assert mask is None
    assert sza is None
    assert vza is None
    assert raa is None
    shutil.move(ref_files[1].parent / "ANG_DATA/SAA_SZA.tif.bac",
                ref_files[1].parent / "ANG_DATA/SAA_SZA.tif")
   # Test read_granule for view angle file missing
    shutil.move(ref_files[1].parent / "ANG_DATA/VAA_VZA_B05.tif",
                ref_files[1].parent / "ANG_DATA/VAA_VZA_B05.tif.bac")
    rho_surface, mask, sza, vza, raa, rho_unc = s2_obs.read_granule(ref_dates[1])
    assert rho_surface is None
    assert rho_unc is None
    assert mask is None
    assert sza is None
    assert vza is None
    assert raa is None
    shutil.move(ref_files[1].parent / "ANG_DATA/VAA_VZA_B05.tif.bac",
                ref_files[1].parent / "ANG_DATA/VAA_VZA_B05.tif")

    #assert False

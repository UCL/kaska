#!/usr/bin/env python
"""Test reading s2_observations data"""
import datetime as dt
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from ..inverters import get_emulator
from ..s2_observations import Sentinel2Observations

TEST_PATH = Path(os.path.dirname(__file__))


def test_s2_data():
    """Test the reading of s2 time series data.
    """

    today = dt.datetime(2017, 1, 1)
    time_grid = [today + dt.timedelta(days=n) for n in range(0, 31, 5)]

    parent = TEST_PATH
    emulator = Path(get_emulator("prosail", "Sentinel2"))
    mask = TEST_PATH / "data" / "ESU.tif"
    # Test Sentinel2Observations constructor
    s2_obs = Sentinel2Observations(
        str(parent),
        emulator,
        str(mask),
        band_prob_threshold=20,
        chunk=None,
        time_grid=time_grid
    )

    ref_dates = [
        dt.datetime(2017, 1, 4, 0, 0),
        dt.datetime(2017, 1, 7, 0, 0),
    ]
    ref_files = [
        parent / "data" / "s2_data" / f"S2A_MSIL1C_{date:%Y%m%d}.SAFE" /
        "GRANULE" / "IMG_DATA" for date in ref_dates
    ]
    assert list(s2_obs.date_data) == ref_dates
    assert [obs[0].parent for _, obs in s2_obs.date_data.items()] == ref_files

    # Test read_granule when all data files are available
    output_names = ['rho_surface', 'mask', 'sza', 'vza', 'raa', 'rho_unc']
    for i, d in enumerate(ref_dates):
        output_data = dict(zip(output_names, s2_obs.read_granule(d)))
        # np.savez_compressed(f"s2_obs_{i}_ref", rho_surface=rho_surface,
        #                    mask=mask, sza=sza, vza=vza, raa=raa,
        #                    rho_unc=rho_unc)
        ref_data = np.load(parent / 'data' / f's2_obs_{i}_ref.npz',
                           allow_pickle=True)
        for datum in output_names:
            np.testing.assert_equal(ref_data[datum], output_data[datum])

    # Test read_granule for missing cloud mask file
    shutil.move(ref_files[1].parent / f"cloud.tif",
                ref_files[1].parent / f"cloud.tif.bac")
    output_data = dict(zip(output_names, s2_obs.read_granule(ref_dates[1])))
    assert list(output_data.values()) == [None] * len(output_names)
    shutil.move(ref_files[1].parent / f"cloud.tif.bac",
                ref_files[1].parent / f"cloud.tif")

    # Test read_granule for reflectivity file missing in the dictionary
    file_list = list(s2_obs.date_data[ref_dates[1]])
    file0 = s2_obs.date_data[ref_dates[1]].pop(0)
    output_data = dict(zip(output_names, s2_obs.read_granule(ref_dates[1])))
    assert list(output_data.values()) == [None] * len(output_names)
    s2_obs.date_data[ref_dates[1]] = list(file_list)
    # Test read_granule for reflectivity file missing in the folder
    shutil.move(file0, file0.with_suffix(".bac"))
    output_data = dict(zip(output_names, s2_obs.read_granule(ref_dates[1])))
    assert list(output_data.values()) == [None] * len(output_names)
    shutil.move(file0.with_suffix(".bac"), file0)

    # Test read_granule for reflectivity uncertainty file missing in the
    # dictionary
    file1 = s2_obs.date_data[ref_dates[1]].pop(1)
    output_data = dict(zip(output_names, s2_obs.read_granule(ref_dates[1])))
    assert list(output_data.values()) == [None] * len(output_names)
    s2_obs.date_data[ref_dates[1]] = list(file_list)
    # Test read_granule for reflectivity uncertainty file missing in the folder
    shutil.move(file1, file1.with_suffix(".bac"))
    output_data = dict(zip(output_names, s2_obs.read_granule(ref_dates[1])))
    assert list(output_data.values()) == [None] * len(output_names)
    shutil.move(file1.with_suffix(".bac"), file1)

    # Test read_granule for sun angle file missing
    shutil.move(ref_files[1].parent / "ANG_DATA" / "SAA_SZA.tif",
                ref_files[1].parent / "ANG_DATA" / "SAA_SZA.tif.bac")
    output_data = dict(zip(output_names, s2_obs.read_granule(ref_dates[1])))
    assert list(output_data.values()) == [None] * len(output_names)
    shutil.move(ref_files[1].parent / "ANG_DATA" / "SAA_SZA.tif.bac",
                ref_files[1].parent / "ANG_DATA" / "SAA_SZA.tif")
    # Test read_granule for view angle file missing
    shutil.move(ref_files[1].parent / "ANG_DATA" / "VAA_VZA_B05.tif",
                ref_files[1].parent / "ANG_DATA" / "VAA_VZA_B05.tif.bac")
    output_data = dict(zip(output_names, s2_obs.read_granule(ref_dates[1])))
    assert list(output_data.values()) == [None] * len(output_names)
    shutil.move(ref_files[1].parent / "ANG_DATA" / "VAA_VZA_B05.tif.bac",
                ref_files[1].parent / "ANG_DATA" / "VAA_VZA_B05.tif")

    # assert False

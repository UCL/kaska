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
    output_names = ['rho_surface', 'mask', 'sza', 'vza', 'raa', 'rho_unc']

    def test_missing_file_in_dict(date, pop_index):
        popped_file = s2_obs.date_data[date].pop(pop_index)
        output_data = dict(zip(output_names, s2_obs.read_granule(date)))
        assert list(output_data.values()) == [None] * len(output_names)
        s2_obs.date_data[date] = list(file_list_orig)

        return popped_file

    def test_missing_file(filepath, date):
        shutil.move(filepath, filepath.with_suffix(".bac"))
        output_data = dict(zip(output_names, s2_obs.read_granule(date)))
        assert list(output_data.values()) == [None] * len(output_names)
        shutil.move(filepath.with_suffix(".bac"), filepath)


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

    # Test read_granule for missing cloud mask file
    test_missing_file(ref_files[1].parent / f"cloud.tif", ref_dates[1])

    # Test read_granule for reflectivity file missing in the dictionary
    file_list_orig = list(s2_obs.date_data[ref_dates[1]])
    file_miss = test_missing_file_in_dict(ref_dates[1], 0)
    # Test read_granule for reflectivity file missing in the folder
    test_missing_file(file_miss, ref_dates[1])

    # Test read_granule for reflectivity uncertainty file missing in the
    # dictionary
    file_miss = test_missing_file_in_dict(ref_dates[1], 1)
    # Test read_granule for reflectivity uncertainty file missing in the folder
    test_missing_file(file_miss, ref_dates[1])

    # Test read_granule for sun angle file missing
    test_missing_file(ref_files[1].parent / "ANG_DATA" / "SAA_SZA.tif",
                      ref_dates[1])
    # Test read_granule for view angle file missing
    test_missing_file(ref_files[1].parent / "ANG_DATA" / "VAA_VZA_B05.tif",
                      ref_dates[1])

    # Test read_granule when all data files are available
    for i, d in enumerate(ref_dates):
        output_data = dict(zip(output_names, s2_obs.read_granule(d)))
        # np.savez_compressed(f"s2_obs_{i}_ref",
        #                    rho_surface=output_data['rho_surface'],
        #                    mask=output_data['mask'], sza=output_data['sza'],
        #                    vza=output_data['vza'], raa=output_data['raa'],
        #                    rho_unc=output_data['rho_unc'])
        ref_data = np.load(parent / 'data' / f's2_obs_{i}_ref.npz',
                           allow_pickle=True)
        for datum in output_names:
            np.testing.assert_equal(ref_data[datum], output_data[datum])

    # assert False

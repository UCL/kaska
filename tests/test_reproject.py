#!/usr/bin/env python
"""Test rerpoject_data"""

import pytest
import gdal
import numpy as np

import sys
sys.path.append("../kaska")

from kaska.utils import reproject_data

def test_reproject_data():
    """Test than when reprojecting a file to match another,
    the output has the same extent and size as the "target"
    one."""
    target = "tests/data/ESU.tif"
    source = "tests/data/s2_test_file.tif"
    gg = reproject_data(source, target)
    g = gdal.Open(target)
    assert g.RasterXSize == gg.RasterXSize
    assert g.RasterYSize == gg.RasterYSize
    # Not sure about this one... Should be tested...
    assert np.allclose(gg.GetGeoTransform(), g.GetGeoTransform())

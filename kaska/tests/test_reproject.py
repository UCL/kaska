#!/usr/bin/env python
"""Test reproject data"""

import os
import sys
import numpy as np
import pytest
import gdal
from ..utils import reproject_data

DATA_PATH = os.path.dirname(__file__)


def test_reproject_data():
    """Test than when reprojecting a file to match another,
    the output has the same extent and size as the "target"
    one."""
    target = DATA_PATH + "/data/ESU.tif"
    source = DATA_PATH + "/data/s2_test_file.tif"
    gg = reproject_data(source, target)
    g = gdal.Open(target)
    assert g.RasterXSize == gg.RasterXSize
    assert g.RasterYSize == gg.RasterYSize
    # Not sure about this one... Should be tested...
    assert np.allclose(gg.GetGeoTransform(), g.GetGeoTransform())

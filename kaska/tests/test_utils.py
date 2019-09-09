"""
Test some utility functions


"""

import pytest
import numpy as np


from ..utils import get_chunks
from ..utils import rasterise_vector


def test_get_chunks():
    nx = 512
    ny = 512
    chunker = np.array([x for x in get_chunks(nx, ny)])
    target = np.array(
        [
            [0, 0, 256, 256, 1],
            [0, 256, 256, 256, 2],
            [256, 0, 256, 256, 3],
            [256, 256, 256, 256, 4],
        ]
    )
    np.allclose(chunker, target)


def test_get_chunks_over_x():
    # Check overflow in x
    nx = 525
    ny = 512
    chunker = np.array([x for x in get_chunks(nx, ny)])
    target = np.array(
        [
            [0, 0, 256, 256, 1],
            [0, 256, 256, 256, 2],
            [256, 0, 256, 256, 3],
            [256, 256, 256, 256, 4],
            [512, 0, 13, 256, 5],
            [512, 256, 13, 256, 6],
        ]
    )
    np.allclose(chunker, target)


def test_rasterise_vector():
    mask = rasterise_vector(
        "/vsicurl/http://www2.geog.ucl.ac.uk/"
        + "~ucfajlg/Estonia/Jarsvelja_ROI.geojson",
        sample_f="/vsicurl/"
        + "http://www2.geog.ucl.ac.uk/~ucfajlg/Estonia/"
        + "S2A_MSIL1C_20170105T094402_N0204_R036_T35VNE_20170105T094358.SAFE/"
        + "GRANULE/L1C_T35VNE_A008041_20170105T094358/"
        + "IMG_DATA/T35VNE_20170105T094402_B06_sur.tif",
    ).ReadAsArray()
    assert mask.sum() == 412451

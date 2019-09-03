'''
Test some utility functions


'''

import pytest
import numpy as np


from ..utils import get_chunks


def test_get_chunks():
    nx = 512
    ny = 512
    chunker = np.array([x for x in get_chunks(nx, ny)])
    target = np.array([[  0,   0, 256, 256,   1],
                       [  0, 256, 256, 256,   2],
                       [256,   0, 256, 256,   3],
                       [256, 256, 256, 256,   4]])
    np.allclose(chunker, target)


def test_get_chunks_over_x():
    # Check overflow in x
    nx = 525
    ny = 512
    chunker = np.array([x for x in get_chunks(nx, ny)])
    target = np.array([[  0,   0, 256, 256,   1],
                        [  0, 256, 256, 256,   2],
                        [256,   0, 256, 256,   3],
                        [256, 256, 256, 256,   4],
                        [512,   0,  13, 256,   5],
                        [512, 256,  13, 256,   6]])
    np.allclose(chunker, target)
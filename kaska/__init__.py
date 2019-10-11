# -*- coding: utf-8 -*-

"""Top-level package for KaSKA."""

__author__ = """Jose Gomez-Dans"""
__email__ = 'j.gomez-dans@ucl.ac.uk'

from .version import __version__
from .TwoNN import Two_NN
from .NNParameterInversion import NNParameterInversion
from .inverters import get_emulators, get_emulator
from .inverters import get_inverters, get_inverter
from .kaska import KaSKA
from .s2_observations import Sentinel2Observations
from .utils import get_chunks, define_temporal_grid
from .inference_runner import kaska_runner

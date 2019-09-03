#!/usr/bin/env python
"""Test reproject data"""
import os

import sys
sys.path.append("../")

import pytest
import gdal
import numpy as np


DATA_PATH = os.path.dirname(__file__)

from ..inverters import get_emulators, get_emulator
from ..inverters import get_inverters, get_inverter

def test_getinverters():
    avail_inverters = get_inverters()
    assert avail_inverters[0]=='prosail_5paras'

def test_getemulators():
    avail_emulators = get_emulators()
    assert avail_emulators[0] == 'prosail'

def test_get_emulator_wrong_sensor():
    with pytest.raises(AssertionError):
        retval = get_emulator("prosail_5paras", "Sentinel34")

def test_get_emulator_wrong_model():
    with pytest.raises(AssertionError):
        retval = get_emulator("parasol", "Sentinel2")

def test_get_emulator_wrong_sensor_model():
    with pytest.raises(AssertionError):
        retval = get_emulator("prosail_5paras", "Sentinel3/OLCI")

def test_get_inverter_wrong_sensor():
    with pytest.raises(AssertionError):
        retval = get_inverter("prosail", "Sentinel34")

def test_get_inverter_wrong_model():
    with pytest.raises(AssertionError):
        retval = get_inverter("parasol", "Sentinel2")

def test_get_inverter_wrong_sensor_model():
    with pytest.raises(AssertionError):
        retval = get_inverter("prosail", "Sentinel3/OLCI")

#def test_get_inverter_ok():
#    retval = get_inverter("prosail_5paras", "Sentinel2")
#    assert retval == "/data/netapp_3/ucfajlg/python/KaSKA/kaska/inverters/Prosail_5_paras.h5"

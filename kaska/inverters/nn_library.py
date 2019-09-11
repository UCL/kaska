#!/usr/bin/env python

"""Functionality to (i) distribute and (ii) access both emulators
and inverters of observation operators.
Some naming conventions:
* An **emulator** is a fast surrogate model for a costly computational
code. The code usually predicts observations (e.g. reflectance, backscatter)
as a function of parameters of interest.
* An **inverter** is a fast model that implements the inverse of the model
the emulator emulates: it predicts parameters of interest *from observations*.

Both emulators and inverters have been trained with a particular sensor 
spectral characteristics in mind.
"""
import logging
import sys
import os
import pkgutil

from io import BytesIO

from enum import Enum

LOG = logging.getLogger(__name__)


valid_sensors = [
    "Sentinel2",
    "Landsat8",
    "Sentinel3/OLCI",
    "Sentinel3/SLSTR",
    "MODIS",
    "Sentinel1",
]

# The NN come in a variety of file formats:
# 1. npz: numpy npz
# 2. h5: HDF5 (tensorflow)
# 3. csv: At some point, someone will bring a CSV
class Format(Enum):
    npz = 1
    h5 = 2
    csv = 3


Inverters = {
    "prosail_5paras": {
        "Sentinel2": {
            "fname": "inverters/Prosail_5_paras.h5",
            "format": Format.h5,
        }
    }
}


Emulators = {
    "prosail": {
        "Sentinel2": {
            "fname": "inverters/prosail_2NN.npz",
            "format": Format.npz,
        }
    }
}


def get_filename(package, resource):
    """Rewrite of pkgutil.get_data() that return the file path.
    Nicked from https://github.com/DLR-RY/pando-core/raw/master/pando/pkg.py
    """
    loader = pkgutil.get_loader(package)
    if loader is None or not hasattr(loader, "get_data"):
        return None
    mod = sys.modules.get(package) or loader.load_module(package)
    if mod is None or not hasattr(mod, "__file__"):
        return None

    # Modify the resource name to be compatible with the loader.get_data
    # signature - an os.path format "filename" starting with the dirname of
    # the package's __file__
    parts = resource.split("/")
    parts.insert(0, os.path.dirname(mod.__file__))
    resource_name = os.path.normpath(os.path.join(*parts))

    return resource_name


def get_inverters():
    """Returns the list of available inverters as distributed with the 
    package.
    
    Returns
    -------
    list
        List of inverter names
    """
    return [x for x in Inverters.keys()]


def get_emulators():
    """Returns a list of available emulators distributed with the package.
    
    Returns
    -------
    lits
        A list of emulator names
    """
    return [x for x in Emulators.keys()]


def get_emulator(name, sensor, fname=True):
    """Retrieve an emulator (`name`) for sensor `sensor`. The function
    returns either the file contents or the file name.
    
    Parameters
    ----------
    name : str
        Emulator name. Use `get_emulators` to get a list of available
        emulators.
    sensor : str
        A sensor list. Currently, it's only Sentinel2
    fname : bool, optional
        Whether to return the filename (True) or the file contents (False)
    
    Returns
    -------
    Emulator file content or filename.
    """
    assert sensor in valid_sensors, f"{sensor:s} is not a valid sensor!"
    assert name in Emulators, f"No such emulator available! {name:s}"
    assert sensor in Emulators[name], (
        f"No {sensor:s} set-up for" + f" emulator {name:s} available"
    )

    if fname:
        emulator = get_filename(
            "kaska", f"{Emulators[name][sensor]['fname']:s}"
        )
    else:
        emulator = pkgutil.get_data(
            "kaska", f"{Emulators[name][sensor]['fname']:s}"
        )
    return emulator


def get_inverter(name, sensor, fname=True):
    """Retrieve an inverter (`name`) for sensor `sensor`. The function
    returns either the file contents or the file name.
    
    Parameters
    ----------
    name : str
        Inverter name. Use `get_emulators` to get a list of available
        emulators.
    sensor : str
        A sensor list. Currently, it's only Sentinel2
    fname : bool, optional
        Whether to return the filename (True) or the file contents (False)
    
    Returns
    -------
    Inverter file content or filename.
    """

    assert sensor in valid_sensors, f"{sensor:s} is not a valid sensor!"
    assert name in Inverters, f"No such emulator available! {name:s}"
    assert sensor in Inverters[name], (
        f"No {sensor:s} set-up for" + f" inverter {name:s} available"
    )
    LOG.debug(f"{Inverters[name][sensor]['fname']:s}")
    if fname:
        inverter = get_filename(
            "kaska", f"{Inverters[name][sensor]['fname']:s}"
        )
    else:
        inverter = pkgutil.get_data(
            "kaska", f"{Inverters[name][sensor]['fname']:s}"
        )

    return inverter

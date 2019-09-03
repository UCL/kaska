#!/usr/bin/env python

"""Some library to find the KaSKA-distributed NN files.
"""
import sys
import os
import pkgutil

from io import BytesIO

from enum import Enum

valid_sensors = [ "Sentinel2",
                 "Landsat8",
                 "Sentinel3/OLCI",
                 "Sentinel3/SLSTR",
                 "MODIS",
                 "Sentinel1"
]

class Format(Enum):
    npz = 1
    h5 = 2
    csv = 3



Inverters = {'prosail_5paras' : {"Sentinel2":{
                                "fname":"inverters/Prosail_5_paras.h5",
                                'format':Format.h5}}}


Emulators = {'prosail' : {"Sentinel2": {
                                "fname":"inverters/prosail_2NN.npz",
                                'format':Format.npz}}}



def get_filename(package, resource):
    """Rewrite of pkgutil.get_data() that return the file path.
    Nicked from https://github.com/DLR-RY/pando-core/raw/master/pando/pkg.py
    """
    loader = pkgutil.get_loader(package)
    if loader is None or not hasattr(loader, 'get_data'):
        return None
    mod = sys.modules.get(package) or loader.load_module(package)
    if mod is None or not hasattr(mod, '__file__'):
        return None

    # Modify the resource name to be compatible with the loader.get_data
    # signature - an os.path format "filename" starting with the dirname of
    # the package's __file__
    parts = resource.split('/')
    parts.insert(0, os.path.dirname(mod.__file__))
    resource_name = os.path.normpath(os.path.join(*parts))

    return resource_name 



def get_inverters():
    return [x for x in Inverters.keys()]


def get_emulators():
    return [x for x in Emulators.keys()]

def get_emulator(name, sensor, fname=True):
    assert sensor in valid_sensors, f"{sensor:s} is not a valid sensor!"
    assert name in Emulators, f"No such emulator available! {name:s}"
    assert sensor in Emulators[name], f"No {sensor:s} set-up for" + \
                            f" emulator {name:s} available"

    if fname:
        emulator = get_filename("kaska",
                f"{Emulators[name][sensor]['fname']:s}")
    else:
        emulator = pkgutil.get_data("kaska",
                f"{Emulators[name][sensor]['fname']:s}")
    return emulator
    
def get_inverter(name, sensor, fname=True):
    assert sensor in valid_sensors, f"{sensor:s} is not a valid sensor!"
    assert name in Inverters, f"No such emulator available! {name:s}"
    assert sensor in Inverters[name], f"No {sensor:s} set-up for" + \
                            f" inverter {name:s} available"
    print(f"{Inverters[name][sensor]['fname']:s}")            
    if fname:
        inverter = get_filename("kaska",
                                f"{Inverters[name][sensor]['fname']:s}")
    else:
        inverter  = pkgutil.get_data("kaska",
                                    f"{Inverters[name][sensor]['fname']:s}")

    return inverter
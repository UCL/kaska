# Minimal setup.py for setuptools
from distutils import log
from setuptools import setup
from setuptools.config import read_configuration

try:
    import gdal
except ImportError:
    log.warn("Gdal is not insalled in your sysem "
    "we recommend to install it through conda "
    "\n    `conda install -c conda-forge gdal`")

extras = read_configuration("setup.cfg")['options']['extras_require']
# Dev is everything
extras['dev'] = list(extras.values())

setup(extras_require=extras)

# Minimal setup.py for setuptools
from setuptools import setup
from setuptools.config import read_configuration


extras = read_configuration("setup.cfg")['options']['extras_require']
# Dev is everything
extras['dev'] = list(extras.values())

setup(extras_require=extras)

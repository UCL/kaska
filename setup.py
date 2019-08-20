# Minimal setup.py for setuptools

from setuptools import setup, find_packages

setup(name="kaska",
      version="0.0.1",
      description = "An efficient smoother",
      url = "https://github.com/jgomezdans/kaska",
      author = "Jose Gomez-Dans",
      author_email = "j.gomez-dans@ucl.ac.uk",
      license = "GNU General Public License v3",
#      packages = find_packages(include=['kaska']),
      packages = ['kaska'], # , 'kaska.TwoNN', 'kaska.NNParameterInversion'],
      scripts= ['scripts/run_kaska'],
      zip_safe=False
      )

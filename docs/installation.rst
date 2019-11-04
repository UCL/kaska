.. highlight:: shell

============
Installation
============

Getting the dependencies
------------------------

We suggest you use `anaconda`_ (or `miniconda`_) with Python 3.7 for the
installation. However, you can also get the dependencies through your prefered
package manager.

Using `conda` you can create an isolated environment for KaSKA.

.. code-block:: console

   $ conda config --add channels conda-forge
   $ conda create -n kaska python=3.7 tensorflow numba gdal
   $ conda activate kaska

Then you can follow the instructions below. You can also install other
dependencies using `conda install package`. If you don't they will downloaded
from pypi when installing KaSKA.


To enter and exit from this environment you will need:

.. code-block:: console

   $ conda activate kaska
   $ ...code...code...code...
   $ conda deactivate



Stable release
--------------

.. warning:: A stable release is still not available via pypi. Read the installation insructions `from sources`_ to know how to install kaska.

To install KaSKA, run this command in your terminal:

.. code-block:: console

    $ pip install kaska

This is the preferred method to install KaSKA, as it will always install the most recent stable release.

If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for KaSKA can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/UCL/kaska

Or download the `tarball`_:

.. code-block:: console

    $ curl  -OL https://github.com/UCL/kaska/tarball/master

Once you have a copy of the source, you can install it, from within the
downloaded directory, with:

.. code-block:: console

    $ pip install .


.. _Github repo: https://github.com/UCL/kaska
.. _tarball: https://github.com/UCL/kaska/tarball/master
.. _anaconda: https://www.anaconda.com/distribution/#download-section
.. _miniconda: https://docs.conda.io/en/latest/miniconda.html

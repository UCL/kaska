#!/usr/bin/env python
"""Generic logging functionality"""
import os
import logging


from .version import __version__


def create_logger(fname=None):
    """Creates a python logger
    
    Parameters
    ----------
    fname : str, optional
        If you pass a filename to this, it'll also output logging info to
        a file.
    
    Returns
    -------
    logging
        A log object.
    """
    logger = logging.getLogger(f"KaSKA-{__version__:s}->")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s -" " %(levelname)s - %(message)s"
    )

    fh = logging.StreamHandler()
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if fname is not None:
        fh = logging.FileHandler(fname)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

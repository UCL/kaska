#!/usr/bin/env python
"""Generic logging functionality"""
import os
import logging


from .version import __version__


def create_logger(debug=True, fname=None):
    """Creates a python logger
    
    Parameters
    ----------
    debug: bool, optional
        Whether to set default logger level to debug
    fname : str, optional
        If you pass a filename to this, it'll also output logging info to
        a file.
    
    Returns
    -------
    logging
        A log object.
    """
    logger = logging.getLogger(f"KaSKA-{__version__:s}->")
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s -" " %(levelname)s - %(message)s"
    )

    ch = logging.StreamHandler()
    if debug:
        ch.setLevel(logging.DEBUG)
    else:
        ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if fname is not None:
        fh = logging.FileHandler(fname)
        if debug:
            fh.setLevel(logging.DEBUG)
         else:
            fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

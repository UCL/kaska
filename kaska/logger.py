#!/usr/bin/env python
"""Generic logging functionality"""

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
    logger
        A log object.
    """
    logger = logging.getLogger()
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s -" " %(levelname)s - %(message)s"
    )

    stream_handle = logging.StreamHandler()
    if debug:
        stream_handle.setLevel(logging.DEBUG)
    else:
        stream_handle.setLevel(logging.INFO)
    stream_handle.setFormatter(formatter)
    logger.addHandler(stream_handle)

    if fname is not None:
        file_handle = logging.FileHandler(fname)
        if debug:
            file_handle.setLevel(logging.DEBUG)
        else:
            file_handle.setLevel(logging.INFO)
        file_handle.setFormatter(formatter)
        logger.addHandler(file_handle)
        # logger.info(f"Logging to {fname:s}")
        logger.info("Logging to %s", fname)
    logger.propagate = True

    return logger

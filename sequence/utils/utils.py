
"""Utils for training"""

import json
import logging
import pathlib
import time


def init_logger(path):
    """Initializes logger.

    Set stream and file handler with specified format.

    Parameters
    ----------
    path : str
        Path to logging file directory

    Returns
    -------
    logger : logging.Logger
        Logger
    """

    log_dir = pathlib.Path(path)
    if not log_dir.exists():
        log_dir.mkdir()

    log_fn = "training_{}.log".format(time.strftime("%Y%m%d"))
    log_path = log_dir.joinpath(log_fn)

    # Initialize logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Set stream handler (console)
    sh = logging.StreamHandler()
    sh_fmt = logging.Formatter(
        "%(asctime)s - %(module)s.%(funcName)s - %(levelname)s : %(message)s")
    sh.setFormatter(sh_fmt)
    logger.addHandler(sh)

    # Set file handler (log file)
    fh = logging.FileHandler(filename=log_path)
    fh_fmt = logging.Formatter(
        "%(asctime)s - %(module)s.%(funcName)s - %(levelname)s : %(message)s")
    fh.setFormatter(fh_fmt)
    logger.addHandler(fh)

    return logger


def load_config(path):

    with pathlib.Path(path).open() as f:
        config = json.load(f)

    return config

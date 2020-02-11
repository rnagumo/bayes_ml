
"""Utils for training"""

import logging
import pathlib
import time

import torch


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


def plot_image_from_latent(generate_from_prior, decoder, z_dim, t_max, device):

    x = []
    z_prev = torch.zeros(1, z_dim).to(device)
    with torch.no_grad():
        for _ in range(t_max):
            # Sample
            samples = generate_from_prior.sample({"z_prev": z_prev})
            x_t = decoder.sample_mean({"z": samples["z"]})

            # Update
            z_prev = samples["z"]
            x.append(x_t[None, :])

        x = torch.cat(x, dim=0).transpose(0, 1)
        return x[:, None]

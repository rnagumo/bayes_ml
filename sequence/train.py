
"""Training method"""

import argparse
import pathlib

import tqdm

import torch
from torch.utils import tensorboard

from data.polydata import init_poly_dataloader
from model.dmm import load_dmm_model
from model.vrnn import load_vrnn_model
from utils.utils import init_logger, load_config


def data_loop(loader, model, device, args, config, train_mode=True):

    # Returned values
    total_loss = 0
    total_len = 0

    # Train with mini-batch
    for x, seq_len in tqdm.tqdm(loader):
        # Input dimension must be (timestep_size, batch_size, feature_size)
        x = x.transpose(0, 1).to(device)

        # Mask for sequencial data
        mask = torch.zeros(x.size(0), x.size(1)).to(device)
        for i, v in enumerate(seq_len):
            mask[:v, i] += 1

        # Initial latent variable
        minibatch_size = x.size(1)
        if args.model == "dmm":
            var_name = "z_prev"
            var = torch.zeros(
                minibatch_size, config["dmm_params"]["z_dim"]).to(device)
        elif args.model == "vrnn":
            var_name = "h_prev"
            var = torch.zeros(
                minibatch_size, config["vrnn_params"]["h_dim"]).to(device)

        # Train / test
        if train_mode:
            _loss = model.train({"x": x, var_name: var}, mask=mask)
        else:
            _loss = model.test({"x": x, var_name: var}, mask=mask)

        # Add training results
        total_loss += _loss * minibatch_size
        total_len += seq_len.sum()

    return total_loss / total_len


def plot_image_from_latent(generate_from_prior, decoder, t_max, device, args,
                           config):

    if args.model == "dmm":
        var_name = "z"
        var = torch.zeros(1, config["dmm_params"]["z_dim"]).to(device)
        input_keys = ["z"]
    elif args.model == "vrnn":
        var_name = "h"
        var = torch.zeros(1, config["vrnn_params"]["h_dim"]).to(device)
        input_keys = ["z", "h_prev"]

    x = []
    with torch.no_grad():
        for _ in range(t_max):
            # Sample
            samples = generate_from_prior.sample({var_name + "_prev": var})
            x_t = decoder.sample_mean({k: samples[k] for k in input_keys})

            # Update
            var = samples[var_name]
            x.append(x_t[None, :])

        x = torch.cat(x, dim=0).transpose(0, 1)
        return x[:, None]


def train(args, logger, config, load_model):

    # -------------------------------------------------------------------------
    # 1. Settings
    # -------------------------------------------------------------------------

    # Settings
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(args.seed)

    # Tensorboard writer
    writer = tensorboard.SummaryWriter(args.logdir)

    # -------------------------------------------------------------------------
    # 2. Data
    # -------------------------------------------------------------------------

    # Loader
    batch_size = args.batch_size
    path = pathlib.Path(args.root, args.filename)
    train_loader, valid_loader, test_loader = init_poly_dataloader(
        path, use_cuda, batch_size)

    logger.info(f"Train data size: {train_loader.dataset.data.size()}")
    logger.info(f"Valid data size: {valid_loader.dataset.data.size()}")
    logger.info(f"Test data size: {test_loader.dataset.data.size()}")

    # Data dimension (seq_len, batch_size, input_size)
    x_dim = train_loader.dataset.data.size(2)
    t_max = train_loader.dataset.data.size(0)

    # -------------------------------------------------------------------------
    # 3. Model
    # -------------------------------------------------------------------------

    model, generate_from_prior, decoder = load_model(
        x_dim, t_max, device, config)

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        logger.info(f"--- Epoch {epoch} ---")

        # Training
        train_loss = data_loop(train_loader, model, device, args, config,
                               train_mode=True)
        valid_loss = data_loop(valid_loader, model, device, args, config,
                               train_mode=False)
        test_loss = data_loop(test_loader, model, device, args, config,
                              train_mode=False)

        # Sample data
        sample = plot_image_from_latent(
            generate_from_prior, decoder, t_max, device, args, config)

        # Log
        writer.add_scalar("train_loss", train_loss.item(), epoch)
        writer.add_scalar("valid_loss", valid_loss.item(), epoch)
        writer.add_scalar("test_loss", test_loss.item(), epoch)
        writer.add_images("image_from_latent", sample, epoch)

        logger.info(f"Train loss = {train_loss.item()}")
        logger.info(f"Valid loss = {valid_loss.item()}")
        logger.info(f"Test loss = {test_loss.item()}")

    writer.close()


def init_args():
    parser = argparse.ArgumentParser(description="Polyphonic data training")

    # Direcotry settings
    parser.add_argument("--logdir", type=str, default="../logs/seq/tmp/")
    parser.add_argument("--root", type=str, default="../data/poly/")
    parser.add_argument("--filename", type=str, default="JSB_Chorales.pickle")
    parser.add_argument("--config", type=str, default="./config.json")
    parser.add_argument("--model", type=str, default="dmm")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)

    return parser.parse_args()


def main():
    # Args
    args = init_args()

    # Logger
    logger = init_logger(args.logdir)
    logger.info("Start logger")
    logger.info(f"Commant line args: {args}")

    # Config
    config = load_config(args.config)

    # Select model
    if args.model == "dmm":
        load_func = load_dmm_model
    elif args.model == "vrnn":
        load_func = load_vrnn_model
    else:
        raise NotImplementedError("Selected model is not implemented")

    try:
        train(args, logger, config, load_func)
    except Exception as e:
        logger.exception(f"Run function error: {e}")

    logger.info("End logger")


if __name__ == "__main__":
    main()

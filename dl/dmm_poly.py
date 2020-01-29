
"""DMM experiments by Pixyz

* Model by pixyz
https://github.com/masa-su/pixyzoo/tree/master/DMM

* Data
http://www-etud.iro.umontreal.ca/~boulanni/icml2012

* Example by pyro
http://pyro.ai/examples/dmm.html
"""

import argparse
import pathlib
import pickle

import tqdm

import torch
from torch import optim
from torch.utils import tensorboard

import pixyz.losses as pxl
import pixyz.models as pxm

from dmm import RNN, Generator, Inference, Prior
from utils import init_logger


class PolyphonicDataset(torch.utils.data.Dataset):
    def __init__(self, data, total_length, note_range=88, bias=20):
        super().__init__()

        # Data size of (seq_len, batch_size, input_size)
        data, seq_len = self._preprocess(data, total_length, note_range, bias)
        self.data = data
        self.seq_len = seq_len

    def __getitem__(self, idx):
        return self.data[:, idx], self.seq_len[idx]

    def __len__(self):
        return self.data.size(1)

    @staticmethod
    def _preprocess(data, total_length, note_range, bias):
        res = []

        # For each sequence
        for seq in range(len(data)):
            seq_len = len(data[seq])
            sequence = torch.zeros((seq_len, note_range))

            # For each time step
            for t in range(seq_len):
                note_slice = torch.tensor(data[seq][t]) - bias
                slice_len = note_slice.size(0)
                if slice_len > 0:
                    # Convert index list to one-hot vector
                    sequence[t, note_slice] = torch.ones(slice_len)

            # Append to list
            res.append(sequence)

        # Pack sequences
        pack = torch.nn.utils.rnn.pack_sequence(res, enforce_sorted=False)

        # Pad packed sequences with given total length
        data, seq_len = torch.nn.utils.rnn.pad_packed_sequence(
            pack, total_length=total_length)

        return data, seq_len


def init_poly_dataloader(path, cuda=False, batch_size=20):
    # Load data from pickle file
    with pathlib.Path(path).open("rb") as f:
        data = pickle.load(f)

    # Max length of all sequences
    max_len = max(len(l) for key in data for l in data[key])

    # Kwargs for data loader
    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else{}

    # Instantiate data loader
    train_loader = torch.utils.data.DataLoader(
        PolyphonicDataset(data["train"], max_len),
        batch_size=batch_size, shuffle=True, **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        PolyphonicDataset(data["valid"], max_len),
        batch_size=batch_size, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        PolyphonicDataset(data["test"], max_len),
        batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, valid_loader, test_loader


def data_loop(loader, model, z_dim, device, train_mode=True):

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
        z_prev = torch.zeros(minibatch_size, z_dim).to(device)

        # Train / test
        if train_mode:
            _loss = model.train({"x": x, "z_prev": z_prev}, mask=mask)
        else:
            _loss = model.test({"x": x, "z_prev": z_prev}, mask=mask)

        # Add training results
        total_loss += _loss * minibatch_size
        total_len += seq_len.sum()

    return total_loss / total_len


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


def load_dmm_model(x_dim, t_max, device, args):

    # Latent dimensions
    h_dim = args.h_dim
    hidden_dim = args.hidden_dim
    z_dim = args.z_dim

    # Distributions
    prior = Prior(z_dim, hidden_dim).to(device)
    decoder = Generator(z_dim, hidden_dim, x_dim).to(device)
    encoder = Inference(z_dim, h_dim).to(device)
    rnn = RNN(x_dim, h_dim).to(device)

    # Sampler
    generate_from_prior = prior * decoder

    # Loss
    ce = pxl.CrossEntropy(encoder, decoder)
    kl = pxl.KullbackLeibler(encoder, prior)
    step_loss = ce + kl
    _loss = pxl.IterativeLoss(step_loss, max_iter=t_max, series_var=["x", "h"],
                              update_value={"z": "z_prev"})
    loss = _loss.expectation(rnn).mean()

    # Model
    dmm = pxm.Model(loss, distributions=[rnn, encoder, decoder, prior],
                    optimizer=optim.Adam,
                    optimizer_params={"lr": args.learning_rate,
                                      "betas": (args.beta1, args.beta2),
                                      "weight_decay": args.weight_decay},
                    clip_grad_norm=args.clip_grad_norm)

    return dmm, generate_from_prior, decoder


def init_args():
    parser = argparse.ArgumentParser(description="DMM Polyphonic")

    # Direcotry settings
    group_1 = parser.add_argument_group("Directory settings")
    group_1.add_argument("--logdir", type=str, default="../logs/tmp/")
    group_1.add_argument("--root", type=str, default="../data/poly/")
    group_1.add_argument("--filename", type=str, default="JSB_Chorales.pickle")

    # Model parameters
    group_2 = parser.add_argument_group("Model parameters")
    group_2.add_argument("--h-dim", type=int, default=32)
    group_2.add_argument("--hidden-dim", type=int, default=32)
    group_2.add_argument("--z-dim", type=int, default=64)

    # Training parameters
    group_3 = parser.add_argument_group("Training parameters")
    group_3.add_argument("--test", action="store_true")
    group_3.add_argument("--cuda", action="store_true")
    group_3.add_argument("--seed", type=int, default=1)
    group_3.add_argument("--batch-size", type=int, default=20)
    group_3.add_argument("--epochs", type=int, default=5)

    # Adam parameters
    group_3 = parser.add_argument_group("Adam parameters")
    group_3.add_argument("--learning-rate", type=float, default=0.0008)
    group_3.add_argument("--beta1", type=float, default=0.96)
    group_3.add_argument("--beta2", type=float, default=0.999)
    group_3.add_argument("--clip-grad-norm", type=float, default=10.0)
    group_3.add_argument("--weight-decay", type=float, default=2.0)

    return parser.parse_args()


def run(args, logger):

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

    model, generate_from_prior, decoder = load_dmm_model(
        x_dim, t_max, device, args)

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        logger.info(f"--- Epoch {epoch} ---")

        # Training
        train_loss = data_loop(train_loader, model, args.z_dim, device,
                               train_mode=True)
        valid_loss = data_loop(valid_loader, model, args.z_dim, device,
                               train_mode=False)

        # Sample data
        sample = plot_image_from_latent(generate_from_prior, decoder,
                                        args.z_dim, t_max, device)

        # Log
        writer.add_scalar("train_loss", train_loss.item(), epoch)
        writer.add_scalar("valid_loss", valid_loss.item(), epoch)
        writer.add_images("image_from_latent", sample, epoch)

        logger.info(f"Train loss = {train_loss.item()}")
        logger.info(f"Valid loss = {valid_loss.item()}")

        # Test
        if args.test:
            test_loss = data_loop(test_loader, model, args.z_dim, device,
                                  train_mode=False)
            writer.add_scalar("test_loss", test_loss.item(), epoch)
            logger.info(f"Test loss = {test_loss.item()}")

    writer.close()


def main():
    # Args
    args = init_args()

    # Logger
    logger = init_logger(args.logdir)
    logger.info("Start logger")
    logger.info(f"Commant line args: {args}")

    try:
        run(args, logger)
    except Exception as e:
        logger.exception(f"Run function error: {e}")

    logger.info("End logger")


if __name__ == "__main__":
    main()

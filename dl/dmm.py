
"""DMM sapmle code by Pixyz

https://github.com/masa-su/pixyzoo/blob/master/DMM/dmm.ipynb
"""

import argparse

import tqdm

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import tensorboard
from torchvision import datasets, transforms

import pixyz.distributions as pxd
import pixyz.losses as pxl
import pixyz.models as pxm


class RNN(pxd.Deterministic):
    def __init__(self, x_dim, h_dim):
        super().__init__(cond_var=["x"], var=["h"])

        self.rnn = nn.GRU(x_dim, h_dim, bidirectional=True)
        self.h0 = nn.Parameter(torch.zeros(2, 1, h_dim))
        self.h_dim = h_dim

    def forward(self, x):
        h0 = self.h0.expand(2, x.size(1), self.h_dim).contiguous()
        h, _ = self.rnn(x, h0)
        return {"h": h}


class Generator(pxd.Bernoulli):
    def __init__(self, z_dim, hidden_dim, x_dim):
        super().__init__(cond_var=["z"], var=["x"])

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, x_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return {"probs": torch.sigmoid(self.fc2(h))}


class Inference(pxd.Normal):
    def __init__(self, z_dim, h_dim):
        super().__init__(cond_var=["h", "z_prev"], var=["z"])

        self.fc1 = nn.Linear(z_dim, h_dim * 2)
        self.fc21 = nn.Linear(h_dim * 2, z_dim)
        self.fc22 = nn.Linear(h_dim * 2, z_dim)

    def forward(self, h, z_prev):
        h_z = torch.tanh(self.fc1(z_prev))
        h = 0.5 * (h + h_z)
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}


class Prior(pxd.Normal):
    def __init__(self, z_dim, hidden_dim):
        super().__init__(cond_var=["z_prev"], var=["z"])

        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)

    def forward(self, z_prev):
        h = F.relu(self.fc1(z_prev))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}


def data_loop(loader, model, z_dim, device, train_mode=True):
    mean_loss = 0
    for data, _ in tqdm.tqdm(loader):
        x = data.transpose(0, 1).to(device)
        batch_size = x.shape[1]
        z_prev = torch.zeros(batch_size, z_dim).to(device)

        if train_mode:
            mean_loss += model.train({"x": x, "z_prev": z_prev})
        else:
            mean_loss += model.test({"x": x, "z_prev": z_prev})

        mean_loss *= batch_size

    return mean_loss / len(loader.dataset)


def plot_image_from_latent(generate_from_prior, decoder, batch_size, z_dim,
                           t_max, device):

    x = []
    z_prev = torch.zeros(batch_size, z_dim).to(device)

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


def init_dataloader(root, cuda=False, batch_size=128):
    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambd=lambda x: x[0]),
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root, train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True, **kwargs,
    )

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root, train=False, transform=transform),
        batch_size=batch_size, shuffle=False, **kwargs,
    )

    return train_loader, test_loader


def init_args():
    parser = argparse.ArgumentParser(description="VAE MNIST")
    parser.add_argument("--logdir", type=str, default="../logs/tmp/")
    parser.add_argument("--data-root", type=str, default="../data/")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--h-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--z-dim", type=int, default=64)

    return parser.parse_args()


def main():
    # -------------------------------------------------------------------------
    # 1. Settings
    # -------------------------------------------------------------------------

    # Args
    args = init_args()

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
    train_loader, test_loader = init_dataloader(
        root=args.data_root, cuda=use_cuda, batch_size=batch_size)

    # Data dimension
    x_dim = train_loader.dataset.data.shape[1]
    t_max = train_loader.dataset.data.shape[2]

    # -------------------------------------------------------------------------
    # 3. Model
    # -------------------------------------------------------------------------

    # Latent dimension
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
    vrnn = pxm.Model(loss, distributions=[rnn, encoder, decoder, prior],
                     optimizer=optim.Adam, optimizer_params={"lr": 1e-3},
                     clip_grad_value=10)

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        # Training
        train_loss = data_loop(train_loader, vrnn, z_dim, device,
                               train_mode=True)
        test_loss = data_loop(test_loader, vrnn, z_dim, device,
                              train_mode=False)

        # Sample data
        sample = plot_image_from_latent(generate_from_prior, decoder,
                                        batch_size, z_dim, t_max, device)

        # Log
        writer.add_scalar("train_loss", train_loss.item(), epoch)
        writer.add_scalar("test_loss", test_loss.item(), epoch)
        writer.add_images("image_from_latent", sample, epoch)

    writer.close()


if __name__ == "__main__":
    main()

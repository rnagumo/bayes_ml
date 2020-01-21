
"""VRNN sapmle code by Pixyz

https://github.com/masa-su/pixyzoo/blob/master/VRNN/vrnn.ipynb
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


class Phi_x(nn.Module):
    def __init__(self, x_dim, h_dim):
        super().__init__()
        self.fc0 = nn.Linear(x_dim, h_dim)

    def forward(self, x):
        return F.relu(self.fc0(x))


class Phi_z(nn.Module):
    def __init__(self, z_dim, h_dim):
        super().__init__()
        self.fc0 = nn.Linear(z_dim, h_dim)

    def forward(self, z):
        return F.relu(self.fc0(z))


class Generator(pxd.Bernoulli):
    def __init__(self, h_dim, z_dim, x_dim, f_phi_z):
        super().__init__(cond_var=["z", "h_prev"], var=["x"])

        self.fc1 = nn.Linear(h_dim * 2, h_dim)
        self.fc2 = nn.Linear(h_dim, h_dim)
        self.fc3 = nn.Linear(h_dim, x_dim)
        self.f_phi_z = f_phi_z

    def forward(self, z, h_prev):
        h = torch.cat([self.f_phi_z(z), h_prev], dim=1)
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        return {"probs": torch.sigmoid(self.fc3(h))}


class Prior(pxd.Normal):
    def __init__(self, h_dim, z_dim):
        super().__init__(cond_var=["h_prev"], var=["z"])

        self.fc1 = nn.Linear(h_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)

    def forward(self, h_prev):
        h = F.relu(self.fc1(h_prev))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}


class Recurrence(pxd.Deterministic):
    def __init__(self, h_dim, f_phi_x, f_phi_z):
        super().__init__(cond_var=["x", "z", "h_prev"], var=["h"])

        self.rnn_cell = nn.GRUCell(h_dim * 2, h_dim)
        self.f_phi_x = f_phi_x
        self.f_phi_z = f_phi_z

    def forward(self, x, z, h_prev):
        h_next = self.rnn_cell(
            torch.cat([self.f_phi_z(z), self.f_phi_x(x)], dim=-1), h_prev)
        return {"h": h_next}


class Inference(pxd.Normal):
    def __init__(self, h_dim, z_dim, f_phi_x):
        super().__init__(cond_var=["x", "h_prev"], var=["z"], name="q")

        self.fc1 = nn.Linear(h_dim * 2, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
        self.f_phi_x = f_phi_x

    def forward(self, x, h_prev):
        h = torch.cat([self.f_phi_x(x), h_prev], dim=-1)
        h = F.relu(self.fc1(h))
        return {"loc": self.fc21(h), "scale": F.softplus(self.fc22(h))}


def data_loop(loader, model, h_dim, device, train_mode=True):
    mean_loss = 0
    for data, _ in tqdm.tqdm(loader):
        x = data.transpose(0, 1).to(device)
        batch_size = x.shape[1]
        h_prev = torch.zeros(batch_size, h_dim).to(device)

        if train_mode:
            mean_loss += model.train({"x": x, "h_prev": h_prev})
        else:
            mean_loss += model.test({"x": x, "h_prev": h_prev})

        mean_loss *= batch_size

    return mean_loss / len(loader.dataset)


def plot_image_from_latent(generate_from_prior, decoder, batch_size, h_dim,
                           t_max, device):

    x = []
    h_prev = torch.zeros(batch_size, h_dim).to(device)

    with torch.no_grad():
        for _ in range(t_max):
            # Sample
            samples = generate_from_prior.sample({"h_prev": h_prev})
            x_t = decoder.sample_mean({"z": samples["z"],
                                       "h_prev": samples["h_prev"]})

            # Update
            h_prev = samples["h"]
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
    parser = argparse.ArgumentParser(description="VRNN MNIST")
    parser.add_argument("--logdir", type=str, default="../logs/tmp/")
    parser.add_argument("--data-root", type=str, default="../data/")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--h-dim", type=int, default=100)
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
    z_dim = args.z_dim

    # Functions
    f_phi_x = Phi_x(x_dim, h_dim).to(device)
    f_phi_z = Phi_z(z_dim, h_dim).to(device)

    # Distributions
    prior = Prior(h_dim, z_dim).to(device)
    decoder = Generator(h_dim, z_dim, x_dim, f_phi_z).to(device)
    encoder = Inference(h_dim, z_dim, f_phi_x).to(device)
    recurrence = Recurrence(h_dim, f_phi_x, f_phi_z).to(device)

    # Sampler
    encoder_with_recurrence = encoder * recurrence
    generate_from_prior = prior * decoder * recurrence

    # Loss
    reconst = pxl.StochasticReconstructionLoss(encoder_with_recurrence,
                                               decoder)
    kl = pxl.KullbackLeibler(encoder, prior)
    step_loss = (reconst + kl).mean()
    loss = pxl.IterativeLoss(step_loss, max_iter=t_max, series_var=["x"],
                             update_value={"h": "h_prev"})

    # Model
    vrnn = pxm.Model(loss, distributions=[encoder, decoder, prior, recurrence],
                     optimizer=optim.Adam, optimizer_params={"lr": 1e-3})

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        # Training
        train_loss = data_loop(train_loader, vrnn, h_dim, device,
                               train_mode=True)
        test_loss = data_loop(test_loader, vrnn, h_dim, device,
                              train_mode=False)

        # Sample data
        sample = plot_image_from_latent(generate_from_prior, decoder,
                                        batch_size, h_dim, t_max, device)

        # Log
        writer.add_scalar("train_loss", train_loss.item(), epoch)
        writer.add_scalar("test_loss", test_loss.item(), epoch)
        writer.add_images("image_from_latent", sample, epoch)

    writer.close()


if __name__ == "__main__":
    main()

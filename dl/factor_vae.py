
"""FactorVAE sapmle code by Pixyz.

https://github.com/masa-su/pixyzoo/tree/master/FactorVAE
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


class Inference(pxd.Normal):
    def __init__(self, x_dim, z_dim):
        super().__init__(cond_var=["x"], var=["z"], name="q")

        self.fc1 = nn.Linear(x_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc31 = nn.Linear(512, z_dim)
        self.fc32 = nn.Linear(512, z_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return {"loc": self.fc31(h), "scale": F.softplus(self.fc32(h))}


class Generator(pxd.Bernoulli):
    def __init__(self, z_dim, x_dim):
        super().__init__(cond_var=["z"], var=["x"], name="p")

        self.fc1 = nn.Linear(z_dim, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, x_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return {"probs": torch.sigmoid(self.fc3(h))}


class InferenceShuffleDim(pxd.Deterministic):
    def __init__(self, q):
        super().__init__(cond_var=["x_shf"], var=["z"], name="q_shf")

        self.q = q

    def forward(self, x_shf):
        z = self.q.sample({"x": x_shf}, return_all=False)["z"]
        return {"z": z[:, torch.randperm(z.shape[1])]}


class Discriminator(pxd.Deterministic):
    def __init__(self, z_dim):
        super().__init__(cond_var=["z"], var=["t"], name="d")

        self.model = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        t = self.model(z)
        return {"t": t}


def data_loop(loader, model, tc, device, train_mode):
    loss = 0
    d_loss = 0
    for x, _ in tqdm.tqdm(loader):
        x = x.to(device)
        len_x = x.shape[0] // 2

        if train_mode:
            loss += model.train({"x": x[:len_x], "x_shf": x[len_x:]})
            d_loss += tc.train({"x": x[:len_x], "x_shf": x[len_x:]})
        else:
            loss += model.test({"x": x[:len_x], "x_shf": x[len_x:]})
            d_loss += tc.test({"x": x[:len_x], "x_shf": x[len_x:]})

    return (loss * loader.batch_size / len(loader.dataset),
            d_loss * loader.batch_size / len(loader.dataset))


def plot_reconstruction(x, q, p, xdim, ydim):
    with torch.no_grad():
        z = q.sample({"x": x}, return_all=False)
        recon_batch = p.sample_mean(z).view(-1, 1, xdim, ydim)
        comparison = torch.cat(
            [x.view(-1, 1, xdim, ydim), recon_batch]).cpu()

        return comparison


def plot_image_from_latent(z_sample, p, xdim, ydim):
    with torch.no_grad():
        sample = p.sample_mean({"z": z_sample}).view(-1, 1, xdim, ydim).cpu()

        return sample


def init_dataloader(root="../data/", cuda=False, batch_size=128):
    kwargs = {"num_workers": 1, "pin_memory": True} if cuda else {}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambd=lambda x: x.view(-1)),
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
    parser = argparse.ArgumentParser(description="FactorVAE MNIST")
    parser.add_argument("--logdir", type=str, default="../logs/tmp/")
    parser.add_argument("--data-root", type=str, default="../data/")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--z-dim", type=int, default=8)
    parser.add_argument("--plot-dim", type=int, default=8)

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

    # Sample data
    _x, _ = iter(test_loader).next()
    _x = _x.to(device)

    # Data dimension
    x_dim = _x.size(1)
    image_dim = int(x_dim ** 0.5)

    # Latent dimension
    z_dim = args.z_dim

    # Dummy latent variable
    plot_dim = args.plot_dim
    z_sample = []
    for i in range(plot_dim):
        z_batch = torch.zeros(plot_dim, z_dim)
        z_batch[:, i] = ((torch.arange(plot_dim, dtype=torch.float32) * 2)
                         / (plot_dim - 1) - 1)
        z_sample.append(z_batch)
    z_sample = torch.cat(z_sample, dim=0).to(device)

    # -------------------------------------------------------------------------
    # 3. Pixyz classses
    # -------------------------------------------------------------------------

    # Distributions
    p = Generator(z_dim, x_dim).to(device)
    q = Inference(x_dim, z_dim).to(device)
    d = Discriminator(z_dim).to(device)
    q_shuffle = InferenceShuffleDim(q).to(device)
    prior = pxd.Normal(
        loc=torch.tensor(0.), scale=torch.tensor(1.),
        var=["z"], features_shape=[z_dim], name="p_prior").to(device)

    # Loss
    reconst = -q.log_prob().expectation(q)
    kl = pxl.KullbackLeibler(q, prior)
    tc = pxl.AdversarialKullbackLeibler(q, q_shuffle, d, optimizer=optim.Adam,
                                        optimizer_params={"lr": 1e-3})
    loss_cls = reconst.mean() + kl.mean() + 10 * tc

    # Model
    model = pxm.Model(loss_cls, distributions=[p, q], optimizer=optim.Adam,
                      optimizer_params={"lr": 1e-3})

    # -------------------------------------------------------------------------
    # 4. Training
    # -------------------------------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        # Training
        train_loss, train_d_loss = data_loop(train_loader, model, tc, device,
                                             train_mode=True)
        test_loss, test_d_loss = data_loop(test_loader, model, tc, device,
                                           train_mode=False)

        # Sample data
        recon = plot_reconstruction(_x[:plot_dim], q, p, image_dim, image_dim)
        sample = plot_image_from_latent(z_sample, p, image_dim, image_dim)

        # Log
        writer.add_scalar("train_loss", train_loss.item(), epoch)
        writer.add_scalar("test_loss", test_loss.item(), epoch)

        writer.add_scalar("train_d_loss", train_d_loss.item(), epoch)
        writer.add_scalar("test_d_loss", test_d_loss.item(), epoch)

        writer.add_images("image_reconstruction", recon, epoch)
        writer.add_images("image_from_latent", sample, epoch)

    writer.close()


if __name__ == "__main__":
    main()

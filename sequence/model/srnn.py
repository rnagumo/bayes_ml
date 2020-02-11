
"""Stochastic RNN

Sequential Neural Models with Stochastic Layers
http://arxiv.org/abs/1605.07571
"""

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

import pixyz.distributions as pxd
import pixyz.losses as pxl
import pixyz.models as pxm

from .iteration_loss import KLAnnealedIterativeLoss


class ForwardRNN(pxd.Deterministic):
    def __init__(self, u_dim, d_dim):
        super().__init__(cond_var=["u"], var=["d"])

        self.rnn = nn.GRU(u_dim, d_dim)
        self.d0 = nn.Parameter(torch.zeros(1, 1, d_dim))

    def forward(self, u):
        d0 = self.d0.expand(1, u.size(1), self.d0.size(2)).contiguous()
        d, _ = self.rnn(u, d0)
        return {"d": d}


class Prior(pxd.Normal):
    def __init__(self, z_dim, d_dim):
        super().__init__(cond_var=["z_prev", "d"], var=["z"])

        self.fc1 = nn.Linear(z_dim + d_dim, 512)
        self.fc21 = nn.Linear(512, z_dim)
        self.fc22 = nn.Linear(512, z_dim)

    def forward(self, z_prev, d):
        h = F.relu(self.fc1(torch.cat([z_prev, d], dim=-1)))
        scale = self.fc21(h)
        loc = F.softplus(self.fc22(h))
        return {"scale": scale, "loc": loc}


class Generator(pxd.Bernoulli):
    def __init__(self, z_dim, d_dim, x_dim):
        super().__init__(cond_var=["z", "d"], var=["x"])

        self.fc1 = nn.Linear(z_dim + d_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, x_dim)

    def forward(self, z, d):
        h = F.relu(self.fc1(torch.cat([z, d], dim=-1)))
        h = F.relu(self.fc2(h))
        probs = torch.sigmoid(self.fc3(h))
        return {"probs": probs}


class BackwardRNN(pxd.Deterministic):
    def __init__(self, x_dim, d_dim, a_dim):
        super().__init__(cond_var=["x", "d"], var=["a"])

        self.rnn = nn.GRU(x_dim + d_dim, a_dim, bidirectional=True)
        self.a0 = nn.Parameter(torch.zeros(2, 1, a_dim))

    def forward(self, x, d):
        a0 = self.a0.expand(2, x.size(1), self.a0.size(2)).contiguous()
        a, _ = self.rnn(torch.cat([x, d], dim=-1), a0)
        return {"a": a[:, :, a.size(2) // 2:]}


class VariationalPrior(pxd.Normal):
    def __init__(self, z_dim, a_dim):
        super().__init__(cond_var=["z_prev", "a"], var=["z"])

        self.fc1 = nn.Linear(z_dim + a_dim, 512)
        self.fc21 = nn.Linear(512, z_dim)
        self.fc22 = nn.Linear(512, z_dim)

    def forward(self, z_prev, a):
        h = F.relu(self.fc1(torch.cat([z_prev, a], dim=-1)))
        scale = self.fc21(h)
        loc = F.softplus(self.fc22(h))
        return {"scale": scale, "loc": loc}


def load_srnn_model(config):

    # Input dimension
    x_dim = config["x_dim"]
    t_dim = config["t_dim"]
    device = config["device"]
    u_dim = x_dim

    # Latent dimension
    d_dim = config["srnn_params"]["d_dim"]
    z_dim = config["srnn_params"]["z_dim"]
    a_dim = config["srnn_params"]["a_dim"]

    # Distributions
    prior = Prior(z_dim, d_dim).to(device)
    frnn = ForwardRNN(u_dim, d_dim).to(device)
    decoder = Generator(z_dim, d_dim, x_dim).to(device)
    brnn = BackwardRNN(x_dim, d_dim, a_dim).to(device)
    encoder = VariationalPrior(z_dim, a_dim).to(device)

    # Loss
    ce = pxl.CrossEntropy(encoder, decoder)
    kl = pxl.KullbackLeibler(encoder, prior)
    _loss = KLAnnealedIterativeLoss(
        ce, kl, max_iter=t_dim, series_var=["x", "d", "a"],
        update_value={"z": "z_prev"}, **config["anneal_params"])

    # Calculate batch loss
    # 1. Forward latent d_{1:T} = frnn(x_{1:T})
    # 2. Backward latent a_{1:T} = brnn(d_{1:T})
    # 3. Latent z_{1:T} from both generative model and variational model
    _loss_batch = _loss.expectation(brnn).expectation(frnn)

    # Mean for batch
    loss = _loss_batch.mean()

    # Model
    dmm = pxm.Model(
        loss, distributions=[prior, frnn, decoder, brnn, encoder],
        optimizer=optim.Adam,
        optimizer_params=config["optimizer_params"])

    # Sampler
    generate_from_prior = prior * frnn * decoder

    return dmm, generate_from_prior, decoder


def init_srnn_variable(minibatch_size, config, x, **kwargs):

    data = {
        "z_prev": torch.zeros(
            minibatch_size, config["srnn_params"]["z_dim"]
        ).to(config["device"]),
        "u": torch.cat(
            [torch.zeros(1, minibatch_size, config["x_dim"]), x[:-1]]
        ).to(config["device"]),
    }

    return data


def get_srnn_update(config):
    data = {
        "z_prev": torch.zeros(
            1, 1, config["srnn_params"]["z_dim"]).to(config["device"]),
        "d_prev": torch.zeros(
            1, 1, config["srnn_params"]["d_dim"]).to(config["device"]),
        "u": torch.zeros(1, 1, config["x_dim"]),
    }
    latent_keys = ["z", "d"]
    update_key_dict = {"z_prev": "z", "d_prev": "d"}

    return data, latent_keys, update_key_dict

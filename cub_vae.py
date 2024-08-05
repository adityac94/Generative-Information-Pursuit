"""
A simple, fully-connected, conditional VAE architecture for CUB attributes.
"""

import torch
from torch import nn
from torch.nn import functional as F

import utils


class CUB_VAE(nn.Module):
    def __init__(self, xdims, category_dims, zdims=100):
        super(CUB_VAE, self).__init__()
        self.category_dims = category_dims
        salientdims = category_dims
        self.zdims = zdims

        self.encoder = nn.Sequential(
            nn.Linear(xdims, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
        )

        # Encode mu and logvar together to share weights
        self.mu_logvar_layer = nn.Linear(128 + salientdims, 2 * zdims)

        self.decoder = nn.Sequential(
            nn.Linear(zdims + salientdims, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, xdims),
        )

    def encode(self, x, category):
        x = self.encoder(x)
        category = F.one_hot(category, self.category_dims)

        # We encode mu and logvar together.
        # First half of output represents mu, the second half is logvar.
        mu_logvar = self.mu_logvar_layer(torch.cat([x, category], dim=1))
        mu, logvar = torch.chunk(mu_logvar, 2, dim=1)
        return mu, logvar

    def decode(self, z, category):
        category = F.one_hot(category, self.category_dims)
        return self.decoder(torch.cat([z, category], dim=1))

    def generate(self, categories):
        # Categories is a tensor of category idxs of shape (num samples to generate,)
        z = torch.randn(len(categories), self.zdims, device=categories.device)
        return self.decode(z, categories), z

    def forward(self, x, category):
        mu, logvar = self.encode(x, category)
        z = utils.vae_sample(mu, logvar)
        return self.decode(z, category), mu, logvar
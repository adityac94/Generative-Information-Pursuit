"""
Simple fully-connected VAE architecture for Bag-of-Words document
classification problem and a dataset loading utility.
"""

import pickle

import torch
from torch import nn
from torch.nn import functional as F

import utils


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc = nn.Linear(in_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.fc(x)))


class VAE(nn.Module):
    def __init__(self, xdims, category_dims, zdims=100):
        super().__init__()
        self.category_dims = category_dims
        salientdims = category_dims
        self.zdims = zdims

        self.encoder = nn.Sequential(
            utils.LinearBlock(xdims, 512),
            utils.LinearBlock(512, 128),
        )

        self.mu_logvar_layer = nn.Linear(128 + salientdims, 2 * zdims)

        self.decoder = nn.Sequential(
            utils.LinearBlock(zdims + salientdims, 128),
            utils.LinearBlock(128, 512),
            nn.Linear(512, xdims),
        )

    def encode(self, x, category):
        x = self.encoder(x)
        category = F.one_hot(category, self.category_dims)
        mu_logvar = self.mu_logvar_layer(torch.cat([x, category], dim=1))
        mu, logvar = torch.chunk(mu_logvar, 2, dim=1)
        return mu, logvar

    def decode(self, z, category):
        category = F.one_hot(category, self.category_dims)
        return self.decoder(torch.cat([z, category], dim=1))

    def generate(self, category):
        # Categories is a tensor of category idxs of shape (num samples to generate,)
        z = torch.randn(len(category), self.zdims, device=category.device)
        return self.decode(z, category)

    def forward(self, x, category):
        mu, logvar = self.encode(x, category)
        z = utils.vae_sample(mu, logvar)
        return self.decode(z, category), mu, logvar


def load_bow_dataset(dataset_name):
    """
    Utility function to load Bag-of-Words document classification dataset
    after preprocessing.
    """

    with open(f"data/doc_classification/{dataset_name}.pkl", "rb") as f:
        data = pickle.load(f)
        x, y = data["x"].toarray(), data["y"]
        label_ids, vocab = data["label_ids"], data["vocab"]

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    train, val, test = torch.utils.data.random_split(
        dataset,
        [
            round(0.8 * len(dataset)),
            round(0.1 * len(dataset)),
            len(dataset) - round(0.8 * len(dataset)) - round(0.1 * len(dataset)),
        ],
        torch.Generator().manual_seed(42),  # Use same seed to split data
    )

    return (train, val, test), vocab, list(label_ids)

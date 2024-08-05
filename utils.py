"""A variety of helpful utility functions."""

import copy
import pdb
import numpy as np
import torch
import lpips
import train
import glob
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
import CUB_dataset
from data.Caltech101Silhouttes import caltech_dataset
import pytorch_lightning as pl
import tqdm
import itertools


def vae_sample(mu, logvar):
    """Utility function to sample from VAE during generation."""

    std_dev = (0.5 * logvar).exp()
    eps = torch.randn_like(std_dev)
    return mu + eps * std_dev


class Binarize:
    """Pytorch transform to binarize grayscale images."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, x):
        return (x > self.threshold).long()


def get_data(dataset, prune=False):
    """Utility function to get the correct dataset with transforms applied."""

    data_dir = "data"
    if dataset == "mnist":
        tsfm = transforms.Compose([transforms.ToTensor(), Binarize()])
        train_ds = datasets.MNIST(data_dir, train=True, transform=tsfm, download=True)
        val_ds = datasets.MNIST(data_dir, train=False, transform=tsfm, download=True)

    elif dataset == "kmnist":
        tsfm = transforms.Compose([transforms.ToTensor(), Binarize()])
        train_ds = datasets.KMNIST(data_dir, train=True, transform=tsfm, download=True)
        val_ds = datasets.KMNIST(data_dir, train=False, transform=tsfm, download=True)

    elif dataset == "fashion_mnist":
        tsfm = transforms.Compose([transforms.ToTensor(), Binarize(threshold=0.1)])
        train_ds = datasets.FashionMNIST(
            data_dir, train=True, transform=tsfm, download=True
        )
        val_ds = datasets.FashionMNIST(
            data_dir, train=False, transform=tsfm, download=True
        )

    elif dataset == "caltech_silhouttes":
        tsfm = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x.long())]
        )
        train_ds = caltech_dataset.CaltechSilhouttes(data_dir, "train", transform=tsfm)
        val_ds = caltech_dataset.CaltechSilhouttes(data_dir, "val", transform=tsfm)

    # "cub_cy" contains (c, y) pairs, "cub_xc" (x, c), "cub_xyc" (x, y, c)
    elif dataset in ["cub_cy", "cub_xc", "cub_xyc"]:
        use_attr = True
        no_img = dataset == "cub_cy"
        uncertain_label = False
        n_class_atr = 1
        image_dir = f"{data_dir}/CUB/CUB_200_2011"
        no_label = dataset == "cub_xc"
        resol = 299

        # resized_resol = int(resol * 256 / 224)
        train_transform = transforms.Compose(
            [
                # transforms.Resize((resized_resol, resized_resol)),
                # transforms.RandomSizedCrop(resol),
                transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                transforms.RandomResizedCrop(resol),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # implicitly divides by 255
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
                # transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
            ]
        )
        train_ds = CUB_dataset.CUBDataset(
            [f"{data_dir}/CUB/train_vote.pkl"],
            use_attr,
            no_img,
            uncertain_label,
            image_dir,
            n_class_atr,
            prune=prune,
            transform=train_transform,
            no_label=no_label,
        )

        val_transform = transforms.Compose(
            [
                # transforms.Resize((resized_resol, resized_resol)),
                transforms.CenterCrop(resol),
                transforms.ToTensor(),  # implicitly divides by 255
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[2, 2, 2])
                # transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ], std = [ 0.229, 0.224, 0.225 ]),
            ]
        )
        val_ds = CUB_dataset.CUBDataset(
            [f"{data_dir}/CUB/val_vote.pkl"],
            use_attr,
            no_img,
            uncertain_label,
            image_dir,
            n_class_atr,
            prune=prune,
            transform=val_transform,
            no_label=no_label,
        )
        test_ds = CUB_dataset.CUBDataset(
            [f"{data_dir}/CUB/test_vote.pkl"],
            use_attr,
            no_img,
            uncertain_label,
            image_dir,
            n_class_atr,
            prune=prune,
            transform=val_transform,
            no_label=no_label,
        )

        return train_ds, val_ds, test_ds
    return train_ds, val_ds

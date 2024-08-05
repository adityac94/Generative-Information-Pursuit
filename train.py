"""
Script to train various models.
"""

import argparse
import math
import os
import sys
import tqdm
import pdbr
from gensim.models import fasttext
import GPUtil
import numpy as np
import pytorch_lightning as pl
import torch
from torch.nn import functional as F
import torchmetrics
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import glob
from torchvision import transforms

import text_vae
import map_estimate
import mnist_vae
import cub_vae, cub_concept_net
import utils

from rich.traceback import install

install()


local_data_dir = "data"
TINY = 1e-16


def loss_function(recon_x, x, mu, logvar, beta, weight=None):
    """VAE loss function."""

    # KLD is KullbackLeibler divergence: how much does one learned
    # distribution deviate from another, in this specific case the
    # learned distribution from the unit Gaussian
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    recon_loss = F.binary_cross_entropy_with_logits(recon_x, x, reduction="none")

    # Weight each output according to its class and output position (word).
    num_loss_outputs = recon_loss.numel()
    if weight is not None:
        recon_loss *= torch.where(x.bool(), weight[1], weight[0])
    recon_loss = recon_loss.mean()

    # KLD tries to push the distributions as close as possible to unit Gaussian.
    kld = -0.5 / num_loss_outputs * (1 + logvar - mu.square() - logvar.exp()).sum()
    # Add two losses together and normalize by the size of the output
    loss = recon_loss + beta * kld
    return loss, {"loss": loss, "recon_loss": recon_loss, "kld": kld}


class TopkMatched(torchmetrics.Metric):
    """Useful pytorch lightning metric for top-k accuracy."""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("matched", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, targets):
        for i in range(len(preds)):
            target_words = targets[i].nonzero().view(-1).tolist()
            topk_recon_words = torch.topk(preds[i], len(target_words)).indices.tolist()
            self.matched += len(set(target_words).intersection(topk_recon_words))
            self.total += len(target_words)

    def compute(self):
        return self.matched / self.total


class MapAccuracyCallback(pl.callbacks.Callback):
    """
    Pytorch lightning callback to calculate MAP VAE posterior accuracy conditioned
    on the entire query set p(y | Q(x)). We use this to tell if our model is learning
    well and properly paying attention to conditional information.

    Note: if `on_num_datapoints` is None, then we do this callback over the entire
    validation dataset
    """

    def __init__(
        self,
        datamodule,
        on_num_datapoints=None,
        num_mc_samples=1000,
        batch_size=None,
        every_n_epochs=1,
    ):
        ys = torch.cat([y for _, y in datamodule.train_dataloader()])
        self.p_y = torch.nn.functional.one_hot(ys).float().mean(dim=0)

        self.val_ds = datamodule.val_dataloader().dataset
        self.on_num_datapoints = on_num_datapoints or len(self.val_ds)
        self.num_mc_samples = num_mc_samples
        self.batch_size = batch_size or datamodule.val_dataloader().batch_size
        self.every_n_epochs = every_n_epochs

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        # Only run every_n_epochs
        if pl_module.current_epoch % self.every_n_epochs != 0:
            return

        correct = 0
        rng = np.random.default_rng()
        for i in tqdm.tqdm(
            rng.choice(len(self.val_ds), size=self.on_num_datapoints, replace=False),
            desc="Calculating Validation MAP Accuracy",
        ):
            x, y = self.val_ds[i]
            posterior = map_estimate.calculate_posterior_full_history(
                pl_module,
                x,
                self.p_y,
                self.num_mc_samples,
                self.batch_size,
            )
            correct += posterior.argmax() == y

        pl_module.log("val_map_accuracy", correct.item() / self.on_num_datapoints)


class VAE(pl.LightningModule):
    batch_size = 128

    def __init__(self, net, epochs, beta=1, metrics={}):
        super().__init__()
        self.net = net

        self.train_metrics = {
            f"train_{k}": v().to(torch.device("cuda")) for k, v in metrics.items()
        }
        self.val_metrics = {
            f"val_{k}": v().to(torch.device("cuda")) for k, v in metrics.items()
        }

        self.beta = beta
        self.epochs = epochs
        self.save_hyperparameters("beta", "epochs")
        self.save_hyperparameters(
            {"architecture": repr(self.net), "batch_size": self.batch_size}
        )

    def _metrics_step(self, x, x_hat, metrics_dict):
        probs = torch.sigmoid(x_hat)
        for m in metrics_dict.values():
            m(probs, x)

    def _compute_beta(self):
        # Beta scheduling. If beta is a list of two numbers, then compute its
        # value on a linear schedule. Otherwise, if beta is a scalar, return it.
        if isinstance(self.beta, list):
            epoch_frac = self.current_epoch / self.epochs
            return (1 - epoch_frac) * self.beta[0] + epoch_frac * self.beta[1]

        return self.beta

    def _step(self, batch, metrics_dict):
        x, c = batch
        x_float = x.float()

        x_hat, mu, logvar = self.net(x_float, c)

        self._metrics_step(x, x_hat, metrics_dict)

        return loss_function(
            x_hat,
            x_float,  # .argmax(dim=2),
            mu,
            logvar,
            self._compute_beta(),
        )

    def training_step(self, batch, batch_idx):
        loss, logs = self._step(batch, self.train_metrics)
        self.log_dict({f"train_{k}": v for k, v in logs.items()})
        self.log_dict(self.train_metrics)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self._step(batch, self.val_metrics)
        self.log_dict({f"val_{k}": v for k, v in logs.items()})
        self.log_dict(self.val_metrics)
        return loss


class BOW_VAE(VAE):
    def __init__(
        self,
        net,
        epochs,
        beta=1,
        dataset_x=None,
        loss_weighting=None,
        ens_beta=None,
    ):
        metrics = {
            "topk_matched": TopkMatched,
            "accuracy": torchmetrics.Accuracy,
            "precision": torchmetrics.Precision,
            "recall": torchmetrics.Recall,
        }

        super().__init__(net, epochs, beta=beta, metrics=metrics)

        if dataset_x is None or loss_weighting is None:
            loss_weights = None
        else:
            class_frequencies = torch.stack(
                [(1 - dataset_x).sum(dim=0), dataset_x.sum(dim=0)]
            ).float()
            if loss_weighting == "inverse_num_samples":
                loss_weights = len(dataset_x) / (class_frequencies + 1)
            elif loss_weighting == "sqrt_inverse_num_samples":
                loss_weights = np.sqrt(len(dataset_x)) / (class_frequencies + 1).sqrt()
            elif loss_weighting == "effective_num_samples":
                loss_weights = (
                    len(dataset_x)
                    * (1 - ens_beta)
                    / (1 - ens_beta ** (class_frequencies + 1))
                )
        self.register_buffer("loss_weights", loss_weights)

        self.save_hyperparameters("loss_weighting", "ens_beta")

    def _step(self, batch, metrics_dict):
        x, c = batch
        x_float = x.float()

        x_hat, mu, logvar = self.net(x_float, c)

        self._metrics_step(x, x_hat, metrics_dict)

        return loss_function(
            x_hat, x_float, mu, logvar, self._compute_beta(), self.loss_weights
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=2e-4)

    @classmethod
    def get_datamodule(cls, dataset_name):
        (train_ds, val_ds, test_ds), vocab, label_ids = text_vae.load_bow_dataset(
            dataset_name
        )
        dm = pl.LightningDataModule.from_datasets(
            train_ds, val_ds, test_ds, batch_size=512, num_workers=0
        )
        return dm, vocab, label_ids

    @classmethod
    def get_backbone(cls, xdims, category_dims):
        return text_vae.VAE(xdims, category_dims)

    @classmethod
    def run(cls, dataset_name):
        parser = argparse.ArgumentParser()
        parser.add_argument("--beta", type=float, default=1)
        parser.add_argument("--loss_weighting", default=None)
        parser.add_argument("--ens_beta", type=float)
        args, unknown = parser.parse_known_args()

        dm, vocab, label_ids = cls.get_datamodule(dataset_name)
        net = cls.get_backbone(xdims=len(vocab), category_dims=len(label_ids))
        epochs = 100
        model = cls(
            net=net,
            epochs=epochs,
            beta=args.beta,
            dataset_x=dm.train_dataloader().dataset.dataset.tensors[0],
            loss_weighting=args.loss_weighting,
            ens_beta=args.ens_beta,
        )
        trainer = pl.Trainer(
            max_epochs=epochs,
            default_root_dir=f"experiments/lightning_logs/{dataset_name}",
            callbacks=[
                MapAccuracyCallback(dm, on_num_datapoints=400),
                ModelCheckpoint(
                    monitor="val_map_accuracy",
                    dirpath=None,
                    mode="max",
                    filename="{epoch:02d}-{val_map_accuracy:.2f}",
                ),
            ],
        )
        trainer.fit(model, dm)


class SupervisedModel(pl.LightningModule):
    def __init__(self, net, arch, loss_fn="ce"):
        super().__init__()
        self.net = net
        self.loss_fn = loss_fn

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.save_hyperparameters({"arch": arch, "net": repr(self.net)})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def _step(self, x, y):
        logits = self.net(x.float())

        if self.loss_fn == "ce":
            loss = torch.nn.functional.cross_entropy(logits, y.long())
            preds = logits.softmax(dim=1)
        elif self.loss_fn == "bce":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            preds = logits.sigmoid()

        return loss, preds

    def training_step(self, batch, batch_idx):
        x, y = batch

        loss, preds = self._step(x, y)

        self.log("train_loss", loss)
        self.train_acc(preds, y.long())
        self.log("train_acc", self.train_acc)
        return {"loss": loss, "y": y, "preds": preds}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, preds = self._step(x, y)

        self.log("val_loss", loss)
        self.val_acc(preds, y.long())
        self.log("val_acc", self.val_acc)
        return {"loss": loss, "y": y, "preds": preds}


class SupervisedBOWModel(SupervisedModel):
    def __init__(self, arch, vocab, category_dims):
        self.vocab = vocab

        if arch == "linear":
            net = torch.nn.Linear(len(vocab), category_dims)
        elif "mlp" in arch:
            net = torch.nn.Sequential(
                torch.nn.Linear(len(vocab), 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 25),
                torch.nn.ReLU(),
                torch.nn.Linear(25, category_dims),
            )
        elif arch == "mlp_embed":
            net = torch.nn.Sequential(
                torch.nn.Linear(len(vocab), 300),
                torch.nn.ReLU(),
                torch.nn.Linear(300, 25),
                torch.nn.ReLU(),
                torch.nn.Linear(25, category_dims),
            )
            wvs = fasttext.load_facebook_vectors("data/raw/crawl-300d-2M-subword.bin")
            self.wv = torch.from_numpy(np.array([wvs[word] for word in vocab]))
            net[0].weight = self.wv.T
            net[0].bias = torch.zeros(300)

        super().__init__(net, arch)

    @classmethod
    def run(cls, dataset_name):
        dm, vocab, label_ids = BOW_VAE.get_datamodule(dataset_name)

        arch = "linear"
        model = cls(arch, vocab, category_dims=len(label_ids))

        trainer = pl.Trainer(
            max_epochs=200,
            default_root_dir=f"experiments/lightning_logs/{dataset_name}",
        )
        trainer.fit(model, dm)


class SupervisedCUBModel(SupervisedModel):
    def __init__(self, arch):

        if arch == "linear":
            net = torch.nn.Linear(312, 200)
        elif arch == "mlp":
            net = torch.nn.Sequential(
                torch.nn.Linear(312, 100),
                torch.nn.ReLU(),
                torch.nn.Linear(100, 25),
                torch.nn.ReLU(),
                torch.nn.Linear(25, 200),
            )
        super().__init__(net, arch)

    @classmethod
    def run(cls):
        trainer = pl.Trainer(
            max_epochs=200,
            default_root_dir="experiments/lightning_logs/cub_supervised",
        )
        model = cls(arch="linear")
        trainer.fit(model, datamodule=CUB_VAE.get_datamodule())


class CUBConceptModel(SupervisedModel):
    def __init__(self):
        net = cub_concept_net.inception_v3(
            pretrained=True,
            freeze=False,
            num_classes=200,
            aux_logits=False,  # should be True?
            n_attributes=312,
            bottleneck=True,
            expand_dim=0,
            three_class=False,
        )
        super().__init__(net, arch="inception_v3", loss_fn="bce")

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=3e-4)
        return torch.optim.SGD(self.net.parameters(), lr=0.01, momentum=0.9)

    def _step(self, x, y):
        logits = self.net(x.float())

        if self.loss_fn == "bce":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                logits, y, reduction="none"
            )

            loss *= torch.where(y.bool(), 9.0, 1.0)
            loss = loss.mean()
            preds = logits.sigmoid()

        return loss, preds

    @classmethod
    def get_datamodule(cls, num_workers=4):
        train_ds, val_ds, test_ds = utils.get_data("cub_xc")
        return pl.LightningDataModule.from_datasets(
            train_ds, val_ds, test_ds, batch_size=64, num_workers=num_workers
        )

    @classmethod
    def run(cls):
        trainer = pl.Trainer(
            max_epochs=2000,
            default_root_dir="experiments/lightning_logs/cub_concept_net",
            gpus=1,
        )
        trainer.fit(
            cls(),
            cls.get_datamodule(),
        )


class CUBConceptModel_Joint(pl.LightningModule):
    def __init__(self):
        super(CUBConceptModel_Joint, self).__init__()
        self.concept_net = CUBConceptModel()
        self.label_net = SupervisedCUBModel(arch="linear")

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        # self.save_hyperparameters({"concept_arch": repr(self.concept_net), "label_net": repr(self.label_net)})

    def configure_optimizers(self):
        # return torch.optim.Adam(self.parameters(), lr=3e-4)
        params = list(self.concept_net.parameters()) + list(self.label_net.parameters())
        return torch.optim.SGD(params, lr=0.01, momentum=0.9)

    def training_step(self, batch, batch_idx):
        x, y, c = batch

        concept_loss, concept_preds = self.concept_net._step(x, c)
        label_loss, label_preds = self.label_net._step(concept_preds, y)

        loss = 0.1 * concept_loss + label_loss

        self.log("train_loss", loss)
        self.train_acc(label_preds, y.long())
        self.log("train_acc", self.train_acc)
        return {"loss": loss, "y": y, "preds": label_preds}

    def validation_step(self, batch, batch_idx):
        x, y, c = batch

        concept_loss, concept_preds = self.concept_net._step(x, c)
        label_loss, label_preds = self.label_net._step(concept_preds, y)

        loss = 0.01 * concept_loss + label_loss

        self.log("val_loss", loss)
        self.val_acc(label_preds, y.long())
        self.log("val_acc", self.val_acc)
        return {"loss": loss, "y": y, "preds": label_preds}

    @classmethod
    def get_datamodule(cls, num_workers=4):
        train_ds, val_ds, test_ds = utils.get_data("cub_xyc")
        return pl.LightningDataModule.from_datasets(
            train_ds, val_ds, test_ds, batch_size=64, num_workers=num_workers
        )

    @classmethod
    def run(cls):
        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc",
            dirpath=None,
            mode="max",
            filename="{epoch:02d}-{val_acc:.2f}",
        )

        trainer = pl.Trainer(
            max_epochs=2000,
            default_root_dir="experiments/lightning_logs/cub_concept_net_joint",
            gpus=1,
            callbacks=[checkpoint_callback],
        )
        trainer.fit(cls(), cls.get_datamodule())


class BinaryImageVAE(VAE):
    batch_size = 128

    @classmethod
    def get_backbone(cls, dataset_name):
        num_classes = 101 if dataset_name == "caltech_silhouttes" else 10
        return mnist_vae.VAE(num_classes=num_classes)

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=1e-3)

    @classmethod
    def get_datamodule(cls, dataset_name):
        train_ds, val_ds = utils.get_data(dataset_name)
        return pl.LightningDataModule.from_datasets(
            train_ds, val_ds, batch_size=cls.batch_size, num_workers=1
        )

    @classmethod
    def run(cls, dataset_name, epochs=200):
        parser = argparse.ArgumentParser()
        parser.add_argument("--beta", type=float, default=1)
        args, unknown = parser.parse_known_args()

        model = cls(net=cls.get_backbone(dataset_name), beta=args.beta, epochs=epochs)
        trainer = pl.Trainer(
            max_epochs=epochs,
            default_root_dir=f"experiments/lightning_logs/{dataset_name}",
            gpus=1,
        )
        trainer.fit(model, cls.get_datamodule(dataset_name))


class CUB_VAE(VAE):
    def __init__(self, net, beta=30, ternary=False, use_noisy_answer=False):
        super().__init__(net, None, beta)
        self.use_noisy = use_noisy_answer
        self.concept_net = None
        if self.use_noisy:
            concept_checkpoint = glob.glob(
                "experiments/lightning_logs/cub_concept_net_joint/lightning_logs/version_5/checkpoints/*.ckpt"
            )[0]
            print("Initializing InceptionV3 Concept Network...")
            self.concept_net = (
                CUBConceptModel_Joint.load_from_checkpoint(concept_checkpoint)
            ).concept_net

            self.concept_net.requires_grad_(False)
            self.concept_net.eval()
            self.concept_net.cuda()

    @classmethod
    def get_backbone(cls, ternary=False):
        if ternary:
            return cub_vae_ternary.CUB_VAE(xdims=312, category_dims=200, zdims=100)
        else:
            return cub_vae.CUB_VAE(xdims=312, category_dims=200, zdims=100)

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=1e-3)

    @classmethod
    def get_datamodule(cls, num_workers=4, prune=False, use_noisy_answer=False):
        if use_noisy_answer:
            train_ds, val_ds, test_ds = utils.get_data("cub_xyc")
            return pl.LightningDataModule.from_datasets(
                train_ds, val_ds, test_ds, batch_size=32, num_workers=4
            )
        else:
            train_ds, val_ds, test_ds = utils.get_data("cub_cy", prune)
            return pl.LightningDataModule.from_datasets(
                train_ds, val_ds, test_ds, batch_size=32, num_workers=num_workers
            )

    def _step(self, batch, metrics_dict):
        if self.use_noisy:
            x, c, _ = batch
            query_preds = self.concept_net.net(x.cuda())
            x_float = (query_preds > 0).float()
        else:
            x, c = batch
            x_float = x.float()

        x_hat, mu, logvar = self.net(x_float, c)

        self._metrics_step(x, x_hat, metrics_dict)

        return loss_function(
            x_hat,
            x_float,  # .argmax(dim=2),
            mu,
            logvar,
            self._compute_beta(),
        )

    @classmethod
    def run(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--beta", type=float, default=1)
        args, unknown = parser.parse_known_args()

        epochs = 200
        model = cls(
            cls.get_backbone(ternary=False),
            args.beta,
            ternary=False,
            use_noisy_answer=True,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=None,
            mode="min",
            filename="{epoch:02d}-{val_loss:.2f}",
        )

        dm = cls.get_datamodule(use_noisy_answer=True)
        trainer = pl.Trainer(
            max_epochs=epochs,
            default_root_dir="experiments/lightning_logs/cub",
            gpus=1,
            callbacks=[checkpoint_callback],
        )

        trainer.fit(model, dm)


class MNIST_MAE(SupervisedModel):
    def __init__(self, net, arch, loss_fn="ce"):
        super().__init__()
        self.net = net
        self.loss_fn = loss_fn

        self.train_acc = torchmetrics.Accuracy()
        self.val_acc = torchmetrics.Accuracy()
        self.save_hyperparameters({"arch": arch, "net": repr(self.net)})

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def _step(self, x, y):
        logits = self.net(x.float())

        if self.loss_fn == "ce":
            loss = torch.nn.functional.cross_entropy(logits, y.long())
            preds = logits.softmax(dim=1)
        elif self.loss_fn == "bce":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
            preds = logits.sigmoid()

        return loss, preds

    def training_step(self, batch, batch_idx):
        x, y = batch

        loss, preds = self._step(x, y)

        self.log("train_loss", loss)
        self.train_acc(preds, y.long())
        self.log("train_acc", self.train_acc)
        return {"loss": loss, "y": y, "preds": preds}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, preds = self._step(x, y)

        self.log("val_loss", loss)
        self.val_acc(preds, y.long())
        self.log("val_acc", self.val_acc)
        return {"loss": loss, "y": y, "preds": preds}

    @classmethod
    def get_datamodule(cls, patch_size=3, num_workers=4, seed=42):
        def patchify(x):
            # Chops us (C x H x W) image into (# Patches x C x Patch W x Patch H)
            patches = x.unfold(1, patch_size, 1).unfold(2, patch_size, 1)
            patches = patches.permute(1, 2, 0, 3, 4)
            return patches.reshape(-1, *patches.shape[2:])

        rng = np.random.default_rng(seed=seed)

        def mask(x):
            num_patches = (28 - patch_size + 1) ** 2
            mask = np.zeros(num_patches)
            mask[rng.integers(num_patches, size=rng.poisson(7))] = 1
            x = x * mask[:, None, None, None]
            return x, mask

        train_ds, val_ds = utils.get_data("mnist")
        train_ds.transform = transforms.Compose([train_ds.transform, patchify, mask])
        val_ds.transform = transforms.Compose([val_ds.transform, patchify, mask])
        return pl.LightningDataModule.from_datasets(
            train_ds, val_ds, batch_size=128, num_workers=num_workers
        )


from train import *


def main():
    dataset = sys.argv[1]

    # NLP architectures
    if dataset in [
        "cleaned_categories10",
        "cleaned_categories10_common_vocab",
        "bbc_news_common_vocab",
        "newsgroups_common_vocab",
    ]:
        BOW_VAE.run(dataset)

    # Supervised baselines
    elif dataset in [
        "cleaned_categories10_supervised",
        "newsgroups_common_vocab_supervised",
        "bbc_news_common_vocab_supervised",
    ]:
        SupervisedBOWModel.run(dataset.split("_supervised")[0])
    elif dataset == "cleaned_categories10_raw":
        RNN.run()
    elif dataset == "cub_supervised":
        SupervisedCUBModel.run()
    elif dataset == "cub_concepts":
        CUBConceptModel.run()
    elif dataset == "cub_concepts_joint":
        CUBConceptModel_Joint.run()

    # Binary image architectures
    elif dataset in ["mnist", "kmnist", "fashion_mnist", "caltech_silhouttes"]:
        BinaryImageVAE.run(dataset)

    elif dataset == "cub":
        CUB_VAE.run()

    else:
        raise ValueError("Invalid VAE experiment name.")


if __name__ == "__main__":
    devices = GPUtil.getAvailable(limit=float("inf"), maxLoad=0.1, maxMemory=0.4)
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(d) for d in devices])
    main()

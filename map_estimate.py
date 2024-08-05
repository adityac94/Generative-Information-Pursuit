"""
This script is used to calculate the posterior accuracy of the VAE
conditioned on the entire query set p(y | Q(x)). This is used as a
baseline and upper bounds the performance of IP which tries to predict
the label from a partial query set S(x) ⊆ Q(x).
"""

import argparse
import glob
import itertools
import math
import os
import pdbr

import numpy as np
import torch
import tqdm
import torch.utils.data
import GPUtil
import yaml
import train
import utils
import torch.nn as nn


def log_importance_ratio(z, mean, sd):
    """likelihood of z ratio on N(0,1) / N(mu, sigma)"""
    normalization_factor = sd.log().sum() / 2 + z.shape[-1] / 2 * math.log(2 * math.pi)
    num = -z.square().sum(dim=-1) / 2
    denom = -((z - mean) / sd).square().sum(dim=-1) / 2
    return normalization_factor + num - denom


def calculate_posterior_full_history(
    model,
    x,
    prior,
    num_samples_per_y,
    batch_size,
    with_importance_sampling=True,
):
    """
    Function to calculate the posterior p(y | Q(x)) using our conditional VAE.

    This is used as a baseline to ensure that when conditioned on the full query
    set, our model is accurate. This quantity is simpler to calculate since the
    query history is not partial, meaning we can use the VAE encoder to find an
    efficient importance sampling distribution to marginalize the latent out of
    the likelihood p(Q(x) | z, y).

    Parameters:
        model: Conditional VAE pytorch lightning model. Contains encoder q(z | Q(x), y)
            and decoder p(Q(x) | z, y).
        prior: Tensor of shape (num_classes,) representing p(y) the prior probability
            of each class in the dataset.
        num_samples_per_y: We will need to calculate the likelihood p(Q(x) | z, y) for
            each class label y. And since this is a Monte Carlo procedure, we use
            multiple samples of z for each likelihood. So total number of samples passed
            through the decoder = num_classes * num_samples_per_y.
        batch_size: For large number of samples of z, we split up the computation into
            several batches. batch_size * num_classes = the number of samples passed
            through per batch.
        with_importance_sampling: Whether to marginalize over z using the encoder as
            an importance sampling distribution. This greatly improves performance.
    """

    x = x.float().to(model.device)
    num_classes = len(prior)
    batch_size = min(num_samples_per_y, batch_size)

    log_likelihoods = []
    for _ in range(0, num_samples_per_y, batch_size):
        # Sample z from the approximate posterior p(z | Q(x), y) learned
        # by the VAE for better results
        z = torch.randn(num_classes, batch_size, model.net.zdims, device=model.device)
        y = torch.arange(num_classes).to(model.device)
        if with_importance_sampling:
            mu, logvar = model.net.encode(torch.stack([x] * num_classes), y)
            mu, logvar = mu.unsqueeze(1), logvar.unsqueeze(1)
            std = torch.exp(logvar / 2.0)
            z = z * std + mu
            importance_weights = log_importance_ratio(z, mu, std)

        logits_x_given_y_z = model.net.decode(
            z.reshape(num_classes * batch_size, -1),
            y.repeat_interleave(batch_size),
        )

        # Calculate the log likelihood of the image given y, z
        px_given_y_z = logits_x_given_y_z.sigmoid().reshape(
            num_classes, batch_size, *x.shape
        )
        event_prob = x * px_given_y_z + (1 - x) * (1 - px_given_y_z)
        log_likelihood = event_prob.flatten(start_dim=2).log().sum(axis=2)

        if with_importance_sampling:
            log_likelihood += importance_weights

        log_likelihoods.append(log_likelihood.logsumexp(dim=1).cpu())

    class_log_likelihoods = torch.stack(log_likelihoods, dim=1).logsumexp(dim=1)
    log_posterior = class_log_likelihoods - math.log(num_samples_per_y) + prior.log()
    posterior = (log_posterior - log_posterior.max()).exp()  # For numerical stability
    posterior /= posterior.sum()  # Normalize

    return posterior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_number")
    args = parser.parse_args()

    dataset_name = "cub_concepts"

    if dataset_name in ["mnist", "kmnist", "fashion_mnist", "caltech_silhouttes"]:
        dm = train.BinaryImageVAE.get_datamodule(dataset_name)
        ds = dm.val_dataloader().dataset

        net = train.BinaryImageVAE.get_backbone(dataset_name)
        model_class = train.BinaryImageVAE
        checkpoint_path = glob.glob(
            f"experiments/lightning_logs/{dataset_name}/lightning_logs/final/checkpoints/*.ckpt"
        )[0]

        num_samples = 12000
        batch_size = 512

    elif dataset_name in [
        "cleaned_categories10",
        "cleaned_categories10_common_vocab",
        "bbc_news_common_vocab",
        "newsgroups_common_vocab",
    ]:
        dm, vocab, label_ids = train.BOW_VAE.get_datamodule(dataset_name)
        ds = dm.test_dataloader().dataset

        net = train.BOW_VAE.get_backbone(xdims=len(vocab), category_dims=len(label_ids))
        model_class = train.BOW_VAE
        num_samples = 10000
        batch_size = 1024

        checkpoint_path = glob.glob(
            f"experiments/lightning_logs/{dataset_name}/lightning_logs/final/checkpoints/*.ckpt"
        )[0]

    elif dataset_name == "cub_concepts":  # "cub_categorical"
        dm = train.CUB_VAE.get_datamodule(use_noisy_answer=True)
        net = train.CUB_VAE.get_backbone()
        model_class = train.CUB_VAE
        num_samples = 12000
        batch_size = 512

        # Load VAE to model p(q(x) | y)
        checkpoint_path = glob.glob(
            f"experiments/lightning_logs/{dataset_name}/lightning_logs/cub_concepts/checkpoints/*.ckpt"
        )[0]

        concept_checkpoint = glob.glob(
            "experiments/lightning_logs/cub_concept_net/lightning_logs/version_12/checkpoints/*.ckpt"
        )[0]
        print("Initializing InceptionV3 Concept Network...")
        concept_net = train.CUBConceptModel.load_from_checkpoint(concept_checkpoint)
        _ = concept_net.requires_grad_(False)
        concept_net.eval()
        concept_net.cuda()

    model = model_class.load_from_checkpoint(
        checkpoint_path,
        net=net,
        epochs=None,
        strict=False,
    )
    model.eval()
    model.cuda()
    _ = model.requires_grad_(False)

    # Load dataset and calculate prior distribution p(y)
    ys = torch.cat(
        [batch[1] for batch in itertools.islice(dm.train_dataloader(), 1000)]
    )
    p_y = torch.histc(ys.float(), bins=len(ys.unique())) / len(ys)

    # Create directory where we will save outputs
    run_number = args.run_number
    if run_number is None:
        try:
            dir_nums = [
                int(d.split("_")[-1])
                for d in glob.glob(f"experiments/map_posteriors/{dataset_name}/run_*")
            ]
            run_number = max(dir_nums) + 1
        except:
            run_number = 0
    save_dir = f"experiments/map_posteriors/{dataset_name}/run_{run_number}"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/hyperparams.yaml", "w") as f:
        yaml.dump({"num_samples": num_samples, "batch_size": batch_size}, f)

    # Calculate posteriors on val dataset
    posterior_results = {}
    preds = []
    labels = []
    rng = np.random.default_rng(1000000 - 1)
    with torch.no_grad():
        for data_index in tqdm.tqdm(rng.choice(len(ds), size=len(ds), replace=False)):

            if not os.path.exists(f"{save_dir}/{data_index}.pth"):
                torch.save(None, f"{save_dir}/{data_index}.pth")

                x, y = ds[data_index][:2]
                if dataset_name == "cub":
                    query_preds = concept_net.net(x.unsqueeze(0).cuda())
                    x = (query_preds > 0).squeeze(0).long()
                posterior = calculate_posterior_full_history(
                    model, x, p_y, num_samples, batch_size
                )
                labels.append(y)
                preds.append(posterior.argmax())

                posterior_results[data_index] = {"label": y, "posterior": posterior}
                torch.save(posterior_results, f"{save_dir}/posteriors.pth")
    labels, preds = torch.tensor(labels), torch.tensor(preds)
    print("Accuracy:", (preds == labels).float().mean())


if __name__ == "__main__":
    devices = GPUtil.getAvailable(limit=float("inf"), maxLoad=0.1, maxMemory=0.2)
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(d) for d in devices])
    main()

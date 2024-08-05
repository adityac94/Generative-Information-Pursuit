"""
This script implements our algorithm. It seeks both to predict
p(y | S(x)), the label distribution conditioned on a partial
query history, and I(q(x), y | S(x)), the mutual information
between the label and remaining queries conditioned on the partial
query history. This both performs inference and tells us which
query to select next.
"""

import pdbr
import argparse
import glob
import itertools
import math
import os

import tqdm
import torch
import numpy as np
import torch.utils.data
import GPUtil
import yaml
import pandas
import pytorch_lightning as pl

import train
import utils

from rich import print
from rich.traceback import install
from torchvision import transforms
from PIL import Image

install()


TINY = 1e-20


dataset_name = None
ula_samples = None
batch_size = None
patch_size = None
query_type = None
num_chains = None
stability_count = None
stability_threshold = None
strategy = None
task_type = None
verbose = None
interative = None
progressbar_on = True


def progress(iterable, *args, **kwargs):
    """A convenient progressbar utility."""
    return tqdm.tqdm(iterable, *args, **kwargs) if progressbar_on else iterable


def make_history_input_for_vae(x, history):
    """
    Utility function that given the query history in dictionary format,
    converts it to a tensor to be used in likelihood calculations with the VAE.
    """
    if task_type in ["topic_classification", "bird_classification_binary"]:
        # history_tensor will have zeros for unrevealed positions, 1s for revealed positive queries
        # and -1 for revealed negative queries
        history_tensor = torch.zeros(x.shape[-1], device=x.device)
        for k in history.keys():
            history_tensor[k] = 2 * x[k] - 1

    else:  # Assume is an image
        mask = torch.zeros(x.shape[1], x.shape[2], device=x.device)
        for (i, j) in history.keys():
            mask[i : i + patch_size, j : j + patch_size] = 1
        history_tensor = torch.cat([mask * x, mask.unsqueeze(0)]).cuda()

    return history_tensor


def vae_px_given_y_z(model, z):
    """
    Utility function that calculates p(Q | z, y) given z's.
    """
    num_categories, b_size = z.shape[0], z.shape[1]
    y = torch.arange(num_categories).repeat_interleave(b_size).cuda()

    px_given_y_z = model.net.decode(z.reshape(num_categories * b_size, -1), y).sigmoid()
    return px_given_y_z.reshape(num_categories, b_size, *px_given_y_z.shape[1:])


def sample_z_given_y_history(
    model, history, old_z, num_samples, num_categories, with_obj_values=False
):
    """
    This method samples z ~ p(z | S(x), y) using ULA to sample from the unnormalized
    VAE decoder p(Q | z, y).
    """

    # To reduce burn-in times, we restart ULA initialized at the samples produced
    # during the last iteration of IP. But on our very first iteration, we just
    # sample from the Gaussian prior and are done.
    if old_z is None:
        return torch.randn(
            num_categories, num_samples, model.net.zdims, device=model.device
        )

    # The ULA algorithm has a variety of hyperparameters. As it is performing
    # gradient descent, it has a stepsize. We also use gradient clipping.
    # Finally, the ULA algorithm is often used with a number of burn-in steps,
    # as the first few samples are often not representative of the distribution
    # we're after (since chains haven't mixed yet). The burn-in samples are discarded.
    step_size = 1e-1
    grad_cutoff = 1e2
    burn_in = 20

    # Some constants.
    normal = torch.distributions.normal.Normal(
        torch.tensor([0.0]).cuda(),
        torch.tensor([1.0]).cuda(),
        validate_args=False,
    )
    b_size = min(num_chains, num_samples)
    one = torch.tensor(1.0).cuda()

    # Initialize ULA from randomly selected samples from the last iteration of IP.
    sample = old_z[:, torch.randperm(num_samples)[:b_size], :].clone().requires_grad_()
    all_samples = []
    obj_values = []

    # We run ULA in batch, so the number of steps = burn_in + num_samples / b_size
    pbar = progress(range(burn_in + math.ceil(num_samples / b_size)))
    for i in pbar:
        px_given_y_z = vae_px_given_y_z(model, sample)

        # Special cases
        if task_type in ["topic_classification", "bird_classification_binary"]:
            # Throw away unrevealed queries (history == 0)
            phist_given_y_z = px_given_y_z[:, :, history != 0]
            # history = 1 means word was present, history = 0 means word wasn't
            # rescale from -1 to 1 now to 0 to 1
            masked_x = (history[history != 0] + 1) / 2

        elif query_type == "binary":  # Assume is an image
            # Throw away unrevealed queries (history == 0)
            phist_given_y_z = px_given_y_z[:, :, history[-1:] != 0]
            # history = 1 means word was present, history = 0 means word wasn't
            # rescale from -1 to 1 now to 0 to 1
            masked_x = history[:-1, history[-1] != 0]

        # compute likelihood of history under vae given y, z
        phist_given_y_z = masked_x * phist_given_y_z + (1 - masked_x) * (
            1 - phist_given_y_z
        )
        phist_given_y_z = phist_given_y_z + TINY

        log_phist_given_y_z = (phist_given_y_z + TINY).log().sum(dim=2)

        # log posterior log p(z | y, B_xk) is proportional to
        #       log(likelihood x prior) = log p(B_xk, | z, y) p (z)
        prior_log_prob = normal.log_prob(sample).sum(dim=2)
        log_pz_given_y_hist = log_phist_given_y_z + prior_log_prob

        # Calculate objective value and gradients (sum of log posterior across classes)
        objective_value = log_pz_given_y_hist.sum()
        objective_value.backward()
        if progressbar_on:
            pbar.set_description(f"ULA (log prob {objective_value / b_size})")
        obj_values.append(objective_value.item() / b_size)

        if not torch.isfinite(objective_value):
            pdbr.set_trace()

        # We have already calculated objective function and its gradients.
        # No we do the update step of the ULA optimization/sampling algorithm,
        # so we don't keep track of gradients.
        with torch.no_grad():
            # Enforce a maximum stepsize per category. This is important since
            # we drop the normalizing factor in the posterior.
            grad_norm = torch.norm(sample.grad.data, dim=(1, 2), keepdim=True)
            updated_step_size = step_size * (
                torch.where(
                    grad_norm < grad_cutoff,
                    one,
                    grad_cutoff / grad_norm,
                )
            )

            # ULA update step of gradient descent on objective plus Gaussian noise.
            # Objective is unnormalized version of log posterior log p(z | y, B_xk):
            # log p(B_xk, z | y) = log p(B_xk | z, y) p(z)
            wiener_samples = torch.randn_like(sample)
            sample.add_(
                updated_step_size * sample.grad.data
                + (2 * updated_step_size).sqrt() * wiener_samples
            )
            if i + 1 >= burn_in:
                all_samples.append(sample.detach().clone())

        # Zero grads
        sample.grad.detach_()
        sample.grad.zero_()

    all_samples = torch.cat(all_samples, dim=1).cpu()
    all_samples = all_samples[:, torch.randperm(all_samples.shape[1])[:num_samples], :]

    if with_obj_values:
        return all_samples, torch.tensor(obj_values)

    return all_samples.cuda()


def generate_all_binary_patches():
    """
    For MNIST and other binary image datasets.
    If patch_size > 3, it would be too memory intensive to generate all possible
    binary patches. So we only use the k most frequent patches and otherwise
    throw out the image.
    """
    if patch_size < 4:
        all_patterns = list(itertools.product([0, 1], repeat=patch_size * patch_size))
    else:
        all_patterns = np.load(f"{patch_size}x{patch_size}_patterns.npy")
    return torch.tensor(all_patterns, device=torch.device("cuda"))


def is_image_covered_by_patch_list(img):
    """
    For binary images on patch sizes larger than 3x3, the total number of possible
    binary patches is 2^(patch_size^2), which can become very large. To deal with
    this, we only use the k most frequent patches in the dataset and otherwise
    throw out the image.
    This function checks whether we should keep the image or not.
    """
    query_answers = img.unfold(1, patch_size, 1).unfold(2, patch_size, 1)
    query_answers = query_answers.squeeze(0).flatten(start_dim=-2).unsqueeze(-1)

    pattern_batch_size = 8
    all_patterns = generate_all_binary_patches().cpu()
    patterns_matched = []
    for i in range(0, len(all_patterns), pattern_batch_size):
        patterns = all_patterns.permute(1, 0)[None, None, :, i : i + pattern_batch_size]
        patterns_matched.append((patterns == query_answers).all(dim=2))

    return torch.cat(patterns_matched, dim=2).any(dim=2).all()


@torch.no_grad()
def px_given_y_z_to_log_pqx_given_y_hist(px):
    """
    Compute log p(Q | S(x), y) given p(Q | y, z) where z ~ p(z | S(x), y) with ULA.
    """

    if task_type in ["topic_classification", "bird_classification_binary"]:
        num_samples = px.shape[1]
        # Marginalize out z
        px = px.sum(dim=1)
        # q(x) can take on two values, so we compute prob of both
        return torch.stack([num_samples - px, px]).log()

    # Tricker in the image case, since patches overlap. We chose to model the
    # image with the VAE p(x | y, z) and then take products of probabilities
    # of pixels in the same patch to get p(Q(x) | y, z).
    if task_type == "binary_image_classification":
        px = px.squeeze(dim=2)  # squeeze out channel dimension

        # Compute all possible patch_size x patch_size binary patches
        all_patterns = generate_all_binary_patches()

        # Generate all overlapping 3x3 patches in image
        pqx = px.unfold(2, patch_size, 1).unfold(3, patch_size, 1)
        # Flatten 2-D grid of patches into a vector of patches and
        # flatten 3x3 patches into 9-entry vector
        pqx = pqx.flatten(start_dim=2, end_dim=3).flatten(start_dim=3)

        # Get probability of each of 512 possible 3x3 patches.
        pattern_batch_size = 16
        prob_pattern_given_y = torch.empty(
            all_patterns.size(0), pqx.size(0), pqx.size(2)
        ).to(px.device)

        for i in range(0, len(all_patterns), pattern_batch_size):
            patterns = all_patterns.permute(1, 0)[
                None, None, None, :, i : i + pattern_batch_size
            ]

            prob_pattern_given_y_z = patterns * pqx.unsqueeze(-1) + (1 - patterns) * (
                1 - pqx.unsqueeze(-1)
            )

            prob_pattern_given_y[i : i + pattern_batch_size] = (
                prob_pattern_given_y_z.log()
                .sum(dim=3)
                .permute(3, 0, 2, 1)
                .logsumexp(dim=-1)
            )

        return prob_pattern_given_y


def calc_log_pqx_given_y_history(
    x,
    model,
    num_categories,
    history,
    old_z,
):
    """
    Calculate the likelihood p(Q(x) | S(x), y) using all the helper methods above.
    """

    history_input = make_history_input_for_vae(x, history)

    # Otherwise, we sample z from p(z | y, B_xk) using ULA.
    z = sample_z_given_y_history(
        model,
        history_input,
        old_z,
        ula_samples,
        num_categories,
    )

    del history_input
    log_pqx_given_y_history_samples = []

    for i in progress(
        range(math.ceil(ula_samples / batch_size)), desc="calculating query probs"
    ):
        batch_z = z[:, i * batch_size : (i + 1) * batch_size, :]
        px_given_y_z = vae_px_given_y_z(model, batch_z)

        # We want to turn this to E_z[p(Q(x) | y, z, B_xk)] = p(Q(x) | y, B_xk)
        log_pqx_given_y_hist = px_given_y_z_to_log_pqx_given_y_hist(px_given_y_z)
        log_pqx_given_y_hist -= math.log(ula_samples)

        log_pqx_given_y_history_samples.append(log_pqx_given_y_hist)

    return torch.stack(log_pqx_given_y_history_samples).logsumexp(dim=0).cpu(), z


def mi_given_history(x, history, model, log_p_y_given_history, z_samples):
    """
    Method calculates the mutual information used to guide the next choice of query for IP
        MI(q(x), y | B_xk) = sum_{q(x), y} [p(q(x), y | B_xk) log(p(q(x) | y, B_xk) / p(q(x) | B_xk))]
    """

    # Calculate p(q(x) | y, B_xk) for all y's
    log_pqx_given_y_history, z_samples = calc_log_pqx_given_y_history(
        x, model, len(log_p_y_given_history), history, z_samples
    )

    # Calculate p(q(x), y | B_xk) = p(q(x) | y, B_xk) * p(y | B_xk) for all y's
    log_pqxy_given_history = (
        log_p_y_given_history.view(1, -1, 1) + log_pqx_given_y_history
    )

    # Calculate p(q(x) | B_xk) = \sum_y p(q(x), y | B_xk)
    log_pqx_given_history = log_pqxy_given_history.logsumexp(dim=1, keepdim=True)

    # Calculate p(q(x), y | B_xk) log [p(q(x) | y, B_xk) / p(q(x) | B_xk)]
    if strategy == "mi":
        mi_inner_term = log_pqxy_given_history.exp() * (
            log_pqx_given_y_history - log_pqx_given_history
        )
        mi_inner_term = torch.where(
            log_pqxy_given_history.exp() < TINY,
            torch.zeros_like(mi_inner_term),
            mi_inner_term,
        )
    elif strategy == "eps_unpredictable":
        mi_inner_term = -log_pqx_given_history.exp() * log_pqx_given_history
    else:
        raise ValueError("Please choose a strategy!")

        eps_val_per_q = torch.absolute(log_pqx_given_history.exp() - 0.5)[0].squeeze()
    # MI(Q(x), Y) = sum_{q(x), y}
    #                 p(q(x) | y, B_xk) p(y | B_xk) log [p(q(x) | y, B_xk) / p(q(x) | B_xk)]
    # So we sum these inner terms over all possible q(x) and y
    mi_per_q = mi_inner_term.sum(dim=(0, 1))
    if mi_per_q.isnan().any():
        pdbr.set_trace()
    return mi_per_q, log_pqx_given_y_history, z_samples, eps_val_per_q


def ip_step(x, history, model, log_p_y_given_history, z_samples, fixed_query=None):
    """
    Computes one step of IP, meaning choosing the next query based on a mutual information
    calculating and updated the posterior p(y | S(x)).
    """

    # Calculate MI
    mi_per_q, log_pqx_given_y_history, z_samples, eps_val_per_q = mi_given_history(
        x, history, model, log_p_y_given_history, z_samples
    )

    if fixed_query is None:
        # Assume we are doing IP on an image dataset with patch queries
        if task_type not in [
            "topic_classification",
            "bird_classification",
            "bird_classification_binary",
        ]:
            num_patches_per_axis = (
                x.shape[1] - patch_size + 1,
                x.shape[2] - patch_size + 1,
            )
            mi_per_q = mi_per_q.view(*num_patches_per_axis)
            eps_val_per_q = eps_val_per_q.view(*num_patches_per_axis)

        # Sometimes because of numerical imprecision, previously asked queries
        # will have a mutual information that is not zero.
        # for k in history:
        #    mi_per_q[k] = 0.0

        for k in history:
            mi_per_q[k] = float("-inf")

        raw_q = mi_per_q.argmax().item()
    else:
        raw_q = fixed_query

    # Update history and p_y_given_history with new query
    if task_type in [
        "topic_classification",
        "bird_classification_binary",
    ]:
        q = raw_q

        if interactive:
            if "bird_classification" not in task_type:
                sys.exit("Not yet implemented for this task")
            attributes = pandas.read_csv(
                "data/CUB/CUB_200_2011/attributes/attributes.txt", sep=" "
            )["attribute_query"]
            val = input(attributes[q] + "? ")
            if val == "Yes":
                query_answer = torch.tensor(1).cuda()
                history[q] = torch.tensor(1).cuda()
            elif val == "No":
                query_answer = torch.tensor(0).cuda()
                history[q] = torch.tensor(0).cuda()

        else:
            query_answer = x[q]
            history[q] = x[q]

    else:  # Assume is image
        num_patches_per_axis = (
            x.shape[1] - patch_size + 1,
            x.shape[2] - patch_size + 1,
        )
        q = (raw_q // num_patches_per_axis[1], raw_q % num_patches_per_axis[1])
        i, j = q

        x_patch = x.cpu().squeeze()[i : i + patch_size, j : j + patch_size].flatten()
        history[q] = x_patch

        if task_type == "binary_image_classification":
            all_patterns = generate_all_binary_patches()
            query_answer = (all_patterns.cpu() == x_patch).prod(dim=1).nonzero()

            # If the true value of the query is not covered by the list of all binary
            # patches (say in 4x4 case) then we skip the image
            if len(query_answer) == 0:
                return "pattern_not_present_in_patch_vocab"

        elif task_type == "unsupervised_semantic_segmentation":
            query_answer = x.squeeze(0)[q]

    log_p_chosen_qx_given_y_history = log_pqx_given_y_history[
        query_answer.long(), :, raw_q
    ]

    # log p(y | B_xk) = log p(q_k(x) | y, B_xk-1) + log p(y | B_xk-1)
    log_p_y_given_history = (
        log_p_chosen_qx_given_y_history + log_p_y_given_history
    ).squeeze()
    log_p_y_given_history -= log_p_y_given_history.logsumexp(
        dim=0
    )  # normalize log p(y | B_xk)

    return q, mi_per_q, history, log_p_y_given_history, z_samples, eps_val_per_q


# Method is only used for debugging in NLP problem
def generate_query_order(x, history_size, strategy):
    if strategy == "uniform":
        idxs = np.random.choice(len(x), size=history_size)

    elif "positive" in strategy:
        positive_ratio = float(strategy.split("-")[1])
        x = x.cpu()
        pos_words_to_sample = min(
            x.count_nonzero(), torch.tensor(positive_ratio * history_size)
        ).long()
        random_positive_idx = np.random.choice(
            x.nonzero().squeeze().numpy(), size=(pos_words_to_sample,), replace=False
        )
        random_negative_idx = np.random.choice(
            (x == 0).nonzero().squeeze().numpy(),
            size=(history_size - pos_words_to_sample,),
            replace=False,
        )
        idxs = np.concatenate([random_positive_idx, random_negative_idx])
        np.random.shuffle(idxs)
    return idxs


def run_ip_on_sample(
    model,
    data_index,
    dataset,
    p_y_prior,
    label_ids,
    save_dir,
    stopping_criterion,
    query_strategy="ip",
    **kwargs,
):
    """
    Run IP on one datapoint

    Parameters:
        p_y_prior: Prior p(y) determined by empirical sampling from the dataset.
        label_ids: A dictionary mapping label indices to their human-readable names.
        stopping_criterion: A dictionary defining the stopping criteria for IP,
            which may include entries for
            - "confidence": stop after p(y | S(x)) moves above a certain threshold
            - "mutual_information": stop after I(q(x), y | S(x)) is below a certain
                threshold for all possible next queries q
            - "num_steps": stop after |S(x)| reaches a certain threshold
        query_strategy: For debugging purposes, we also experimented with other
            query selection strategies other than information pursuit.
    """

    global stability_count

    if type(data_index) == np.int64:
        data, label = dataset[data_index][:2]

    elif type(data_index) == str:
        img = Image.open(data_index).convert("RGB")
        resol = 299
        tfsm = transforms.Compose(
            [
                transforms.Resize(resol),
                transforms.CenterCrop(resol),
                transforms.ToTensor(),  # implicitly divides by 255
            ]
        )

        data = tfsm(img)
        label = 0
        data_index = 0

    data = data.cuda()
    log_p_y_given_history = p_y_prior.log()
    # When we perform IP on cub_concepts, we use the trained concept network
    # p(Q | x) to implement the query set and tell us which concepts are present
    # and not present in the image.
    if dataset_name == "cub_concepts":
        query_preds = kwargs["query_answering_network"].net(data.unsqueeze(0))
        data = (query_preds > 0).squeeze(0).long()

    history = {}
    query_order = []
    mutual_informations = []
    eps_vals = []
    log_p_y_posteriors = []
    z_samples = None

    # Used for debugging on NLP task only
    if query_strategy != "ip":
        query_order_to_choose = generate_query_order(
            data, stopping_criterion["num_steps"], query_strategy
        )

    while True:
        if verbose:
            print(
                f"\nRunning IP on {dataset_name}[{data_index}], label={label} ({label_ids[label]})..."
            )
            print(f"Query {len(mutual_informations) + 1}...")

        # IP step
        res = ip_step(
            data,
            history,
            model,
            log_p_y_given_history,
            z_samples,
            None if query_strategy == "ip" else query_order_to_choose[len(query_order)],
        )
        # In case of specific errors, stop and skip this sample
        if res == "pattern_not_present_in_patch_vocab":
            return
        (q, mi_per_q, history, log_p_y_given_history, z_samples, eps_val_per_q) = res

        if q is None:
            return None

        query_order.append(q)
        mutual_informations.append(mi_per_q)
        log_p_y_posteriors.append(log_p_y_given_history.cpu())
        eps_vals.append(eps_val_per_q)

        # Logging
        y_pred = log_p_y_given_history.argmax().item()
        if verbose:
            if task_type == "topic_classification":
                print(
                    f"query_id: {q}, answer: Is \"{kwargs['vocab'][q]}\" in headline? {'Yes' if data[q] else 'No'}  "
                )
            else:
                print(f"    query patch: {q}")

            print(
                f"prediction: {y_pred} ({label_ids[y_pred]}), predicted prob: {log_p_y_given_history.max().exp().item()}"
            )

        if mi_per_q is not None and strategy == "mi":
            if verbose:
                print(f"mi: {mi_per_q[q]}")
            eps_vals = []

        elif strategy == "eps_unpredictable":
            if verbose:
                print("eps_val:", eps_val_per_q[q])

        if verbose:
            print()

        save_data = {
            "data_index": data_index,
            "log_posteriors": torch.stack(log_p_y_posteriors),
            "history": history,
            "query_order": query_order,
            "mutual_information": mutual_informations,
            "eps_val": eps_vals,
        }
        torch.save(save_data, f"{save_dir}/{data_index}.pth")

        if "stability" in stopping_criterion:
            if "confidence" in stopping_criterion:
                if (
                    log_p_y_given_history.max().exp()
                    >= stopping_criterion["confidence"]
                ):
                    stability_count += 1
                elif (
                    log_p_y_given_history.max().exp() < stopping_criterion["confidence"]
                ):
                    stability_count -= 1
                    stability_count = max(0, stability_count)

            if "mutual_information" in stopping_criterion:
                if mi_per_q.max() <= stopping_criterion["mutual_information"]:
                    stability_count += 1
                elif mi_per_q.max() <= stopping_criterion["mutual_information"]:
                    stability_count -= 1
                    stability_count = max(0, stability_count)

            if "num_steps" in stopping_criterion:
                if len(query_order) >= stopping_criterion["num_steps"]:
                    break

            if stability_count == stopping_criterion["stability"]:
                break

            if not torch.isfinite(mi_per_q[q]):
                break

        else:
            if "confidence" in stopping_criterion:
                if (
                    log_p_y_given_history.max().exp()
                    >= stopping_criterion["confidence"]
                ):
                    break
            if "mutual_information" in stopping_criterion:
                if mi_per_q.max() <= stopping_criterion["mutual_information"]:
                    break
            if "num_steps" in stopping_criterion:
                if len(query_order) >= stopping_criterion["num_steps"]:
                    break
            if "eps" in stopping_criterion:
                if eps_val_per_q[q] > stopping_criterion["eps"]:
                    break
            if not torch.isfinite(mi_per_q[q]):
                break


def main():
    # If run_number is None, we automatically create a new directory to hold results
    # of running IP. If it is specified, we place results in
    # "experiments/ip/dataset_name/run_{run_number}"
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_number")
    args = parser.parse_args()

    # Configuration variables
    global dataset_name
    global ula_samples
    global batch_size
    global num_chains
    global patch_size
    global query_type
    global stability_count
    global stability_threshold
    global strategy
    global task_type
    global verbose
    global interactive
    global progressbar_on

    dataset_name = "cleaned_categories10"
    strategy = "mi"  # "eps_unpredictable"

    user_provided_image = False
    interactive = False

    if dataset_name in ["mnist", "kmnist", "fashion_mnist", "caltech_silhouttes"]:
        ula_samples = 12000
        # Separate batch size and stopping criteria for certain datasets
        if dataset_name == "caltech_silhouttes":
            batch_size = 3
            stopping_criterion = {"mutual_information": 0, "num_steps": 26 * 26}
        else:
            batch_size = 256
            stopping_criterion = {
                "confidence": 0.99,
                "num_steps": 26 * 26,
                "stability": 5,
            }

        num_chains = 256
        patch_size = 3
        checkpoint_path = glob.glob(
            f"experiments/lightning_logs/{dataset_name}/lightning_logs/final/checkpoints/*.ckpt"
        )[0]
        query_type = "binary"
        task_type = "binary_image_classification"

    elif dataset_name in [
        "cleaned_categories10",
        "cleaned_categories10_common_vocab",
        "bbc_news_common_vocab",
        "newsgroups_common_vocab",
    ]:
        ula_samples = 10000
        batch_size = 512
        num_chains = batch_size
        checkpoint_path = glob.glob(
            f"experiments/lightning_logs/{dataset_name}/lightning_logs/final_old/checkpoints/*.ckpt"
        )[0]
        stopping_criterion = {"num_steps": 1000, "mutual_information": 1e-3}
        query_type = "binary"
        task_type = "topic_classification"
        progressbar_on = False
        verbose = True

    elif dataset_name in ["cub", "cub_concepts"]:
        ula_samples = 12000
        batch_size = 256
        num_chains = batch_size
        patch_size = 1
        if strategy == "mi":
            stopping_criterion = {"mutual_information": 1e-6, "num_steps": 312}
        if strategy == "eps_unpredictable":
            stopping_criterion = {"eps": 0.4, "num_steps": 312}

        task_type = "bird_classification_binary"

        query_type = "binary"
        checkpoint_path = glob.glob(
            "experiments/lightning_logs/cub/lightning_logs/noisy_all_queries/checkpoints/*.ckpt"
        )[0]
        progressbar_on = False
        user_provided_image = False
        verbose = True
        interactive = False

    query_strategy = "ip"

    # Load dataset and setup model
    kwargs = {}
    if task_type == "topic_classification":
        # Initialize dataset
        dm, vocab, label_ids = train.BOW_VAE.get_datamodule(dataset_name)
        kwargs["vocab"] = vocab

        # Initialize model
        net = train.BOW_VAE.get_backbone(xdims=len(vocab), category_dims=len(label_ids))
        model = train.BOW_VAE.load_from_checkpoint(
            checkpoint_path, net=net, epochs=None, strict=False
        )
        prior_batches = 1000

    elif task_type == "binary_image_classification":
        # Initialize model
        net = train.BinaryImageVAE.get_backbone(dataset_name)
        model = train.BinaryImageVAE.load_from_checkpoint(
            checkpoint_path, net=net, epochs=None, strict=True
        )

        # Initialize dataset
        dm = model.get_datamodule(dataset_name)
        classes = dm.val_dataloader().dataset.classes
        label_ids = dict(enumerate(classes))
        prior_batches = 1000

    if "bird_classification" in task_type:
        net = train.CUB_VAE.get_backbone()
        model = train.CUB_VAE.load_from_checkpoint(
            checkpoint_path, net=net, strict=False
        )

        if dataset_name == "cub_concepts":
            concept_checkpoint = glob.glob(
                "experiments/lightning_logs/cub_concept_net/lightning_logs/version_31/checkpoints/*.ckpt"
            )[0]
            print("Initializing InceptionV3 Concept Network...")
            concept_net = train.CUBConceptModel.load_from_checkpoint(concept_checkpoint)
            _ = concept_net.requires_grad_(False)
            concept_net.eval()
            concept_net.cuda()
            kwargs["query_answering_network"] = concept_net

            train_ds, val_ds, test_ds = utils.get_data("cub_xyc")

        elif dataset_name == "cub":
            train_ds, val_ds, test_ds = utils.get_data("cub_cy")

        # Initialize datamodule
        dm = pl.LightningDataModule.from_datasets(
            train_ds, val_ds, test_ds, batch_size=256, num_workers=4
        )
        classes_file = pandas.read_csv("data/CUB/classes.txt", sep=" ")
        classes = classes_file["species"]
        label_ids = dict(enumerate(classes))
        prior_batches = 25

    dataset = dm.test_dataloader().dataset
    _ = model.requires_grad_(False)
    model.eval()
    model.cuda()

    # Initially, history is empty, so p_y_given_history can be calculated from dataset
    pbar = tqdm.tqdm(
        itertools.islice(dm.train_dataloader(), prior_batches),
        desc="Loading p_y_prior",
        total=prior_batches,
    )
    ys = torch.cat([batch[1] for batch in pbar])
    p_y_prior = torch.histc(ys.float(), bins=len(ys.unique())) / len(ys)
    # Add smoothing for labels that never appear in subsample of dataset.
    smoothing_prob = 0.1 / len(p_y_prior)
    p_y_prior = (smoothing_prob + p_y_prior) / (1 + smoothing_prob * len(p_y_prior))

    # Create directory where we will save outputs, based on "run_number" argument
    run_number = args.run_number
    if run_number is None:
        try:
            dir_nums = [
                int(d.split("_")[-1])
                for d in glob.glob(f"experiments/ip/{dataset_name}/run_*")
            ]
            run_number = max(dir_nums) + 1
        except:
            run_number = 0
    save_dir = f"experiments/ip/{dataset_name}/run_{run_number}"
    os.makedirs(save_dir, exist_ok=True)

    with open(f"{save_dir}/hyperparams.yaml", "w") as f:
        yaml.dump(
            {
                "ula_samples": ula_samples,
                "query_strategy": query_strategy,
                "batch_size": batch_size,
                "stopping_criterion": stopping_criterion,
                "patch_size": patch_size,
                "checkpoint_path": checkpoint_path,
            },
            f,
        )

    if user_provided_image:
        img_path = (
            "/cis/home/achatto1/semantic-nlp/data/CUB/random_images/blue-budgie.jpeg"
        )
        run_ip_on_sample(
            model,
            img_path,
            dataset,
            p_y_prior,
            label_ids,
            save_dir,
            stopping_criterion,
            query_strategy=query_strategy,
            **kwargs,
        )

    else:
        rng = np.random.default_rng(1000000 - 1)
        for data_index in rng.choice(len(dataset), size=len(dataset), replace=False):
            if not verbose:
                print("Processed data index:", data_index)
            # Check that we haven't started running IP on this sample yet
            stability_count = 0
            if not os.path.exists(f"{save_dir}/{data_index}.pth"):
                torch.save({}, f"{save_dir}/{data_index}.pth")
                run_ip_on_sample(
                    model,
                    data_index,
                    dataset,
                    p_y_prior,
                    label_ids,
                    save_dir,
                    stopping_criterion,
                    query_strategy=query_strategy,
                    **kwargs,
                )


if __name__ == "__main__":
    devices = GPUtil.getAvailable(limit=float("inf"), maxLoad=0.1, maxMemory=0.1)
    os.environ["CUDA_VISIBLE_DEVICES"] = ", ".join([str(d) for d in devices])
    main()

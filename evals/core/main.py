# fmt: off
# flake8: noqa: E501
# fmt: on
import argparse
import time
import functools
import random
from typing import Type, Tuple, Callable, Any, Union, Dict, List, Mapping, Optional
from dataclasses import asdict
import logging
import math
import re
import gc
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import einops
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformer_lens.hook_points import HookedRootModule
from sae_lens.sae import SAE
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens.training.activations_store import ActivationsStore


from evals.core.eval_config import CoreEvalConfig
from evals.core.eval_output import (
    CoreEvalOutput,
    CoreMetricCategories,
    ModelBehaviorPreservationMetrics,
    ModelPerformancePreservationMetrics,
    ReconstructionQualityMetrics,
    ShrinkageMetrics,
    SparsityMetrics,
    TokenStatsMetrics,
    CoreFeatureMetric,
)
from sae_bench_utils import (
    get_eval_uuid,
    get_sae_lens_version,
    get_sae_bench_version,
)

import sae_bench_utils.sae_selection_utils as sae_selection_utils
import sae_bench_utils.general_utils as general_utils

logger = logging.getLogger(__name__)

# you can truncate to save space/bandwidth, but be warned that this will
# likely screw up the feature density metrics among others. 10 is a good
# compromise.
DEFAULT_FLOAT_PRECISION = 10


def retry_with_exponential_backoff(
    retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
) -> Callable:
    """
    Decorator for retrying a function with exponential backoff.

    Args:
        retries: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delay
        exceptions: Exception(s) to catch and retry on
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception = None

            for retry_count in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if retry_count == retries:
                        logger.error(f"Failed after {retries} retries: {str(e)}")
                        raise

                    # Calculate delay with optional jitter
                    current_delay = min(delay * (exponential_base**retry_count), max_delay)
                    if jitter:
                        current_delay *= 1 + random.random() * 0.1  # 10% jitter

                    logger.warning(
                        f"Attempt {retry_count + 1}/{retries} failed: {str(e)}. "
                        f"Retrying in {current_delay:.2f} seconds..."
                    )
                    time.sleep(current_delay)

            if last_exception:
                raise last_exception
            return None

        return wrapper

    return decorator


def get_library_version() -> str:
    try:
        return version("sae_lens")
    except PackageNotFoundError:
        return "unknown"


def get_git_hash() -> str:
    """
    Retrieves the current Git commit hash.
    Returns 'unknown' if the hash cannot be determined.
    """
    try:
        # Ensure the command is run in the directory where .git exists
        git_dir = Path(__file__).resolve().parent.parent  # Adjust if necessary
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=git_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


# Everything by default is false so the user can just set the ones they want to true
@dataclass
class MultipleEvalsConfig:
    batch_size_prompts: int | None = None
    n_eval_reconstruction_batches: int = 10
    n_eval_sparsity_variance_batches: int = 1
    compute_kl: bool = False
    compute_ce_loss: bool = False
    compute_l2_norms: bool = False
    compute_sparsity_metrics: bool = False
    compute_variance_metrics: bool = False
    library_version: str = field(default_factory=get_library_version)
    git_hash: str = field(default_factory=get_git_hash)


def get_multiple_evals_everything_config(
    batch_size_prompts: int | None = None,
    n_eval_reconstruction_batches: int = 10,
    n_eval_sparsity_variance_batches: int = 1,
) -> MultipleEvalsConfig:
    """
    Returns a MultipleEvalsConfig object with all metrics set to True
    """
    return MultipleEvalsConfig(
        batch_size_prompts=batch_size_prompts,
        n_eval_reconstruction_batches=n_eval_reconstruction_batches,
        compute_kl=True,
        compute_ce_loss=True,
        compute_l2_norms=True,
        n_eval_sparsity_variance_batches=n_eval_sparsity_variance_batches,
        compute_sparsity_metrics=True,
        compute_variance_metrics=True,
    )


@torch.no_grad()
def run_evals(
    sae: SAE,
    activation_store: ActivationsStore,
    model: HookedRootModule,
    eval_config: CoreEvalConfig = CoreEvalConfig(),
    model_kwargs: Mapping[str, Any] = {},
    ignore_tokens: set[int | None] = set(),
    verbose: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hook_name = sae.cfg.hook_name
    actual_batch_size = eval_config.batch_size_prompts or activation_store.store_batch_size_prompts

    # TODO: Come up with a cleaner long term strategy here for SAEs that do reshaping.
    # turn off hook_z reshaping mode if it's on, and restore it after evals
    if "hook_z" in hook_name:
        previous_hook_z_reshaping_mode = sae.hook_z_reshaping_mode
        sae.turn_off_forward_pass_hook_z_reshaping()
    else:
        previous_hook_z_reshaping_mode = None

    all_metrics = {
        "model_behavior_preservation": {},
        "model_performance_preservation": {},
        "reconstruction_quality": {},
        "shrinkage": {},
        "sparsity": {},
        "token_stats": {},
    }

    if eval_config.compute_kl or eval_config.compute_ce_loss:
        assert eval_config.n_eval_reconstruction_batches > 0
        reconstruction_metrics = get_downstream_reconstruction_metrics(
            sae,
            model,
            activation_store,
            compute_kl=eval_config.compute_kl,
            compute_ce_loss=eval_config.compute_ce_loss,
            n_batches=eval_config.n_eval_reconstruction_batches,
            eval_batch_size_prompts=actual_batch_size,
            ignore_tokens=ignore_tokens,
            exclude_special_tokens_from_reconstruction=eval_config.exclude_special_tokens_from_reconstruction,
            verbose=verbose,
        )

        if eval_config.compute_kl:
            all_metrics["model_behavior_preservation"].update(
                {
                    "kl_div_score": reconstruction_metrics["kl_div_score"],
                    "kl_div_with_ablation": reconstruction_metrics["kl_div_with_ablation"],
                    "kl_div_with_sae": reconstruction_metrics["kl_div_with_sae"],
                }
            )

        if eval_config.compute_ce_loss:
            all_metrics["model_performance_preservation"].update(
                {
                    "ce_loss_score": reconstruction_metrics["ce_loss_score"],
                    "ce_loss_with_ablation": reconstruction_metrics["ce_loss_with_ablation"],
                    "ce_loss_with_sae": reconstruction_metrics["ce_loss_with_sae"],
                    "ce_loss_without_sae": reconstruction_metrics["ce_loss_without_sae"],
                }
            )

        activation_store.reset_input_dataset()

    if (
        eval_config.compute_l2_norms
        or eval_config.compute_sparsity_metrics
        or eval_config.compute_variance_metrics
    ):
        assert eval_config.n_eval_sparsity_variance_batches > 0
        sparsity_variance_metrics, feature_metrics = get_sparsity_and_variance_metrics(
            sae,
            model,
            activation_store,
            compute_l2_norms=eval_config.compute_l2_norms,
            compute_sparsity_metrics=eval_config.compute_sparsity_metrics,
            compute_variance_metrics=eval_config.compute_variance_metrics,
            compute_featurewise_density_statistics=eval_config.compute_featurewise_density_statistics,
            n_batches=eval_config.n_eval_sparsity_variance_batches,
            eval_batch_size_prompts=actual_batch_size,
            model_kwargs=model_kwargs,
            ignore_tokens=ignore_tokens,
            verbose=verbose,
        )

        if eval_config.compute_l2_norms:
            all_metrics["shrinkage"].update(
                {
                    "l2_norm_in": sparsity_variance_metrics["l2_norm_in"],
                    "l2_norm_out": sparsity_variance_metrics["l2_norm_out"],
                    "l2_ratio": sparsity_variance_metrics["l2_ratio"],
                    "relative_reconstruction_bias": sparsity_variance_metrics[
                        "relative_reconstruction_bias"
                    ],
                }
            )

        if eval_config.compute_sparsity_metrics:
            all_metrics["sparsity"].update(
                {
                    "l0": sparsity_variance_metrics["l0"],
                    "l1": sparsity_variance_metrics["l1"],
                }
            )

        if eval_config.compute_variance_metrics:
            all_metrics["reconstruction_quality"].update(
                {
                    "explained_variance": sparsity_variance_metrics["explained_variance"],
                    "mse": sparsity_variance_metrics["mse"],
                    "cossim": sparsity_variance_metrics["cossim"],
                }
            )
    else:
        feature_metrics = {}

    if eval_config.compute_featurewise_weight_based_metrics:
        feature_metrics |= get_featurewise_weight_based_metrics(sae)

    if len(all_metrics) == 0:
        raise ValueError("No metrics were computed, please set at least one metric to True.")

    # restore previous hook z reshaping mode if necessary
    if "hook_z" in hook_name:
        if previous_hook_z_reshaping_mode and not sae.hook_z_reshaping_mode:
            sae.turn_on_forward_pass_hook_z_reshaping()
        elif not previous_hook_z_reshaping_mode and sae.hook_z_reshaping_mode:
            sae.turn_off_forward_pass_hook_z_reshaping()

    total_tokens_evaluated_eval_reconstruction = (
        activation_store.context_size
        * eval_config.n_eval_reconstruction_batches
        * actual_batch_size
    )

    total_tokens_evaluated_eval_sparsity_variance = (
        activation_store.context_size
        * eval_config.n_eval_sparsity_variance_batches
        * actual_batch_size
    )

    all_metrics["token_stats"] = {
        "total_tokens_eval_reconstruction": total_tokens_evaluated_eval_reconstruction,
        "total_tokens_eval_sparsity_variance": total_tokens_evaluated_eval_sparsity_variance,
    }

    # Remove empty metric groups
    all_metrics = {k: v for k, v in all_metrics.items() if v}

    return all_metrics, feature_metrics


def get_featurewise_weight_based_metrics(sae: SAE) -> dict[str, Any]:
    unit_norm_encoders = (sae.W_enc / sae.W_enc.norm(dim=0, keepdim=True)).cpu()
    unit_norm_decoder = (sae.W_dec.T / sae.W_dec.T.norm(dim=0, keepdim=True)).cpu()

    encoder_norms = sae.W_enc.norm(dim=-2).cpu().tolist()

    # gated models have a different bias (no b_enc)
    if not hasattr(sae, "b_enc") and not hasattr(sae, "b_mag"):
        encoder_bias = torch.zeros(sae.cfg.d_sae).cpu().tolist()
    elif sae.cfg.architecture != "gated":
        encoder_bias = sae.b_enc.cpu().tolist()
    else:
        encoder_bias = sae.b_mag.cpu().tolist()

    encoder_decoder_cosine_sim = (
        torch.nn.functional.cosine_similarity(
            unit_norm_decoder.T,
            unit_norm_encoders.T,
        )
        .cpu()
        .tolist()
    )

    return {
        "encoder_bias": encoder_bias,
        "encoder_norm": encoder_norms,
        "encoder_decoder_cosine_sim": encoder_decoder_cosine_sim,
    }


def get_downstream_reconstruction_metrics(
    sae: SAE,
    model: HookedRootModule,
    activation_store: ActivationsStore,
    compute_kl: bool,
    compute_ce_loss: bool,
    n_batches: int,
    eval_batch_size_prompts: int,
    ignore_tokens: set[int | None] = set(),
    exclude_special_tokens_from_reconstruction: bool = False,
    verbose: bool = False,
):
    metrics_dict = {}
    if compute_kl:
        metrics_dict["kl_div_with_sae"] = []
        metrics_dict["kl_div_with_ablation"] = []
    if compute_ce_loss:
        metrics_dict["ce_loss_with_sae"] = []
        metrics_dict["ce_loss_without_sae"] = []
        metrics_dict["ce_loss_with_ablation"] = []

    batch_iter = range(n_batches)
    if verbose:
        batch_iter = tqdm(batch_iter, desc="Reconstruction Batches")

    for _ in batch_iter:
        batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)
        for metric_name, metric_value in get_recons_loss(
            sae,
            model,
            batch_tokens,
            activation_store,
            compute_kl=compute_kl,
            compute_ce_loss=compute_ce_loss,
            ignore_tokens=ignore_tokens,
            exclude_special_tokens_from_reconstruction=exclude_special_tokens_from_reconstruction,
        ).items():
            if len(ignore_tokens) > 0:
                mask = torch.logical_not(
                    torch.any(
                        torch.stack([batch_tokens == token for token in ignore_tokens], dim=0),
                        dim=0,
                    )
                )
                if metric_value.shape[1] != mask.shape[1]:
                    # ce loss will be missing the last value
                    mask = mask[:, :-1]
                metric_value = metric_value[mask]

            metrics_dict[metric_name].append(metric_value)

    metrics: dict[str, float] = {}
    for metric_name, metric_values in metrics_dict.items():
        metrics[f"{metric_name}"] = torch.cat(metric_values).mean().item()

    if compute_kl:
        metrics["kl_div_score"] = (
            metrics["kl_div_with_ablation"] - metrics["kl_div_with_sae"]
        ) / metrics["kl_div_with_ablation"]

    if compute_ce_loss:
        metrics["ce_loss_score"] = (
            metrics["ce_loss_with_ablation"] - metrics["ce_loss_with_sae"]
        ) / (metrics["ce_loss_with_ablation"] - metrics["ce_loss_without_sae"])

    return metrics


def get_sparsity_and_variance_metrics(
    sae: SAE,
    model: HookedRootModule,
    activation_store: ActivationsStore,
    n_batches: int,
    compute_l2_norms: bool,
    compute_sparsity_metrics: bool,
    compute_variance_metrics: bool,
    compute_featurewise_density_statistics: bool,
    eval_batch_size_prompts: int,
    model_kwargs: Mapping[str, Any],
    ignore_tokens: set[int | None] = set(),
    verbose: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hook_name = sae.cfg.hook_name
    hook_head_index = sae.cfg.hook_head_index

    metric_dict = {}
    feature_metric_dict = {}

    if compute_l2_norms:
        metric_dict["l2_norm_in"] = []
        metric_dict["l2_norm_out"] = []
        metric_dict["l2_ratio"] = []
        metric_dict["relative_reconstruction_bias"] = []
    if compute_sparsity_metrics:
        metric_dict["l0"] = []
        metric_dict["l1"] = []
    if compute_variance_metrics:
        metric_dict["explained_variance"] = []
        metric_dict["mse"] = []
        metric_dict["cossim"] = []
    if compute_featurewise_density_statistics:
        feature_metric_dict["feature_density"] = []
        feature_metric_dict["consistent_activation_heuristic"] = []

    total_feature_acts = torch.zeros(sae.cfg.d_sae, device=sae.device)
    total_feature_prompts = torch.zeros(sae.cfg.d_sae, device=sae.device)
    total_tokens = 0

    batch_iter = range(n_batches)
    if verbose:
        batch_iter = tqdm(batch_iter, desc="Sparsity and Variance Batches")

    for _ in batch_iter:
        batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)

        if len(ignore_tokens) > 0:
            mask = torch.logical_not(
                torch.any(
                    torch.stack([batch_tokens == token for token in ignore_tokens], dim=0),
                    dim=0,
                )
            )
        else:
            mask = torch.ones_like(batch_tokens, dtype=torch.bool)
        flattened_mask = mask.flatten()

        # get cache
        _, cache = model.run_with_cache(
            batch_tokens,
            prepend_bos=False,
            names_filter=[hook_name],
            stop_at_layer=sae.cfg.hook_layer + 1,
            **model_kwargs,
        )

        # we would include hook z, except that we now have base SAE's
        # which will do their own reshaping for hook z.
        has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
        if hook_head_index is not None:
            original_act = cache[hook_name][:, :, hook_head_index]
        elif any(substring in hook_name for substring in has_head_dim_key_substrings):
            original_act = cache[hook_name].flatten(-2, -1)
        else:
            original_act = cache[hook_name]

        # normalise if necessary (necessary in training only, otherwise we should fold the scaling in)
        if activation_store.normalize_activations == "expected_average_only_in":
            original_act = activation_store.apply_norm_scaling_factor(original_act)

        # send the (maybe normalised) activations into the SAE
        sae_feature_activations = sae.encode(original_act.to(sae.device))
        sae_out = sae.decode(sae_feature_activations).to(original_act.device)
        del cache

        if activation_store.normalize_activations == "expected_average_only_in":
            sae_out = activation_store.unscale(sae_out)

        flattened_sae_input = einops.rearrange(original_act, "b ctx d -> (b ctx) d")
        flattened_sae_feature_acts = einops.rearrange(
            sae_feature_activations, "b ctx d -> (b ctx) d"
        )
        flattened_sae_out = einops.rearrange(sae_out, "b ctx d -> (b ctx) d")

        # TODO: Clean this up.
        # apply mask
        masked_sae_feature_activations = sae_feature_activations * mask.unsqueeze(-1)
        flattened_sae_input = flattened_sae_input[flattened_mask]
        flattened_sae_feature_acts = flattened_sae_feature_acts[flattened_mask]
        flattened_sae_out = flattened_sae_out[flattened_mask]

        if compute_l2_norms:
            l2_norm_in = torch.norm(flattened_sae_input, dim=-1)
            l2_norm_out = torch.norm(flattened_sae_out, dim=-1)
            l2_norm_in_for_div = l2_norm_in.clone()
            l2_norm_in_for_div[torch.abs(l2_norm_in_for_div) < 0.0001] = 1
            l2_norm_ratio = l2_norm_out / l2_norm_in_for_div

            # Equation 10 from https://arxiv.org/abs/2404.16014
            # https://github.com/saprmarks/dictionary_learning/blob/main/evaluation.py
            x_hat_norm_squared = torch.norm(flattened_sae_out, dim=-1) ** 2
            x_dot_x_hat = (flattened_sae_input * flattened_sae_out).sum(dim=-1)
            relative_reconstruction_bias = (
                x_hat_norm_squared.mean() / x_dot_x_hat.mean()
            ).unsqueeze(0)

            metric_dict["l2_norm_in"].append(l2_norm_in)
            metric_dict["l2_norm_out"].append(l2_norm_out)
            metric_dict["l2_ratio"].append(l2_norm_ratio)
            metric_dict["relative_reconstruction_bias"].append(relative_reconstruction_bias)

        if compute_sparsity_metrics:
            l0 = (flattened_sae_feature_acts > 0).sum(dim=-1).float()
            l1 = flattened_sae_feature_acts.sum(dim=-1)
            metric_dict["l0"].append(l0)
            metric_dict["l1"].append(l1)

        if compute_variance_metrics:
            resid_sum_of_squares = (flattened_sae_input - flattened_sae_out).pow(2).sum(dim=-1)
            total_sum_of_squares = (
                (flattened_sae_input - flattened_sae_input.mean(dim=0)).pow(2).sum(-1)
            )

            mse = resid_sum_of_squares / flattened_mask.sum()
            explained_variance = 1 - resid_sum_of_squares / total_sum_of_squares

            x_normed = flattened_sae_input / torch.norm(flattened_sae_input, dim=-1, keepdim=True)
            x_hat_normed = flattened_sae_out / torch.norm(flattened_sae_out, dim=-1, keepdim=True)
            cossim = (x_normed * x_hat_normed).sum(dim=-1)

            metric_dict["explained_variance"].append(explained_variance)
            metric_dict["mse"].append(mse)
            metric_dict["cossim"].append(cossim)

        if compute_featurewise_density_statistics:
            sae_feature_activations_bool = (masked_sae_feature_activations > 0).float()
            total_feature_acts += sae_feature_activations_bool.sum(dim=1).sum(dim=0)
            total_feature_prompts += (sae_feature_activations_bool.sum(dim=1) > 0).sum(dim=0)
            total_tokens += mask.sum()

    # Aggregate scalar metrics
    metrics: dict[str, float] = {}
    for metric_name, metric_values in metric_dict.items():
        metrics[f"{metric_name}"] = torch.cat(metric_values).mean().item()

    # Aggregate feature-wise metrics
    feature_metrics: dict[str, list[float]] = {}
    feature_metrics["feature_density"] = (total_feature_acts / total_tokens).tolist()
    feature_metrics["consistent_activation_heuristic"] = (
        total_feature_acts / total_feature_prompts
    ).tolist()

    return metrics, feature_metrics


@torch.no_grad()
def get_recons_loss(
    sae: SAE,
    model: HookedRootModule,
    batch_tokens: torch.Tensor,
    activation_store: ActivationsStore,
    compute_kl: bool,
    compute_ce_loss: bool,
    ignore_tokens: set[int | None] = set(),
    exclude_special_tokens_from_reconstruction: bool = False,
    model_kwargs: Mapping[str, Any] = {},
) -> dict[str, Any]:
    hook_name = sae.cfg.hook_name
    head_index = sae.cfg.hook_head_index

    original_logits, original_ce_loss = model(
        batch_tokens, return_type="both", loss_per_token=True, **model_kwargs
    )

    if len(ignore_tokens) > 0 and exclude_special_tokens_from_reconstruction:
        mask = torch.logical_not(
            torch.any(
                torch.stack([batch_tokens == token for token in ignore_tokens], dim=0),
                dim=0,
            )
        )
    else:
        mask = torch.ones_like(batch_tokens, dtype=torch.bool)

    metrics = {}

    # TODO(tomMcGrath): the rescaling below is a bit of a hack and could probably be tidied up
    def standard_replacement_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        # SAE class agnost forward forward pass.
        reconstructed_activations = sae.decode(sae.encode(activations)).to(activations.dtype)

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            reconstructed_activations = activation_store.unscale(reconstructed_activations)

        reconstructed_activations = torch.where(
            mask[..., None], reconstructed_activations, activations
        )

        return reconstructed_activations.to(original_device)

    def all_head_replacement_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        # SAE class agnost forward forward pass.
        new_activations = sae.decode(sae.encode(activations.flatten(-2, -1))).to(activations.dtype)

        new_activations = new_activations.reshape(
            activations.shape
        )  # reshape to match original shape

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            new_activations = activation_store.unscale(new_activations)

        # Apply mask to keep original activations for ignored tokens
        new_activations = torch.where(mask[..., None, None], new_activations, activations)

        return new_activations.to(original_device)

    def single_head_replacement_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)

        # Handle rescaling if SAE expects it
        if activation_store.normalize_activations == "expected_average_only_in":
            activations = activation_store.apply_norm_scaling_factor(activations)

        # Create a copy of activations to modify
        new_activations = activations.clone()

        # Only reconstruct the specified head
        head_activations = sae.decode(sae.encode(activations[:, :, head_index])).to(
            activations.dtype
        )

        # Apply mask only to the reconstructed head
        masked_head_activations = torch.where(
            mask[..., None], head_activations, activations[:, :, head_index]
        )
        new_activations[:, :, head_index] = masked_head_activations

        # Unscale if activations were scaled prior to going into the SAE
        if activation_store.normalize_activations == "expected_average_only_in":
            new_activations = activation_store.unscale(new_activations)

        return new_activations.to(original_device)

    def standard_zero_ablate_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)
        activations = torch.zeros_like(activations)
        return activations.to(original_device)

    def single_head_zero_ablate_hook(activations: torch.Tensor, hook: Any):
        original_device = activations.device
        activations = activations.to(sae.device)
        activations[:, :, head_index] = torch.zeros_like(activations[:, :, head_index])
        return activations.to(original_device)

    # we would include hook z, except that we now have base SAE's
    # which will do their own reshaping for hook z.
    has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
    if any(substring in hook_name for substring in has_head_dim_key_substrings):
        if head_index is None:
            replacement_hook = all_head_replacement_hook
            zero_ablate_hook = standard_zero_ablate_hook
        else:
            replacement_hook = single_head_replacement_hook
            zero_ablate_hook = single_head_zero_ablate_hook
    else:
        replacement_hook = standard_replacement_hook
        zero_ablate_hook = standard_zero_ablate_hook

    recons_logits, recons_ce_loss = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_name, partial(replacement_hook))],
        loss_per_token=True,
        **model_kwargs,
    )

    zero_abl_logits, zero_abl_ce_loss = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_name, zero_ablate_hook)],
        loss_per_token=True,
        **model_kwargs,
    )

    def kl(original_logits: torch.Tensor, new_logits: torch.Tensor):
        original_probs = torch.nn.functional.softmax(original_logits, dim=-1)
        log_original_probs = torch.log(original_probs)
        new_probs = torch.nn.functional.softmax(new_logits, dim=-1)
        log_new_probs = torch.log(new_probs)
        kl_div = original_probs * (log_original_probs - log_new_probs)
        kl_div = kl_div.sum(dim=-1)
        return kl_div

    if compute_kl:
        recons_kl_div = kl(original_logits, recons_logits)
        zero_abl_kl_div = kl(original_logits, zero_abl_logits)
        metrics["kl_div_with_sae"] = recons_kl_div
        metrics["kl_div_with_ablation"] = zero_abl_kl_div

    if compute_ce_loss:
        metrics["ce_loss_with_sae"] = recons_ce_loss
        metrics["ce_loss_without_sae"] = original_ce_loss
        metrics["ce_loss_with_ablation"] = zero_abl_ce_loss

    return metrics


def all_loadable_saes() -> list[tuple[str, str, float, float]]:
    all_loadable_saes = []
    saes_directory = get_pretrained_saes_directory()
    for release, lookup in tqdm(saes_directory.items()):
        for sae_name in lookup.saes_map.keys():
            expected_var_explained = lookup.expected_var_explained[sae_name]
            expected_l0 = lookup.expected_l0[sae_name]
            all_loadable_saes.append((release, sae_name, expected_var_explained, expected_l0))

    return all_loadable_saes


def nested_dict() -> defaultdict[Any, Any]:
    return defaultdict(nested_dict)


def dict_to_nested(flat_dict: dict[str, Any]) -> defaultdict[Any, Any]:
    nested = nested_dict()
    for key, value in flat_dict.items():
        parts = key.split("/")
        d = nested
        for part in parts[:-1]:
            d = d[part]
        d[parts[-1]] = value
    return nested


def convert_feature_metrics(
    flattened_feature_metrics: Dict[str, List[float]],
) -> List[CoreFeatureMetric]:
    """Convert feature metrics from parallel lists to list of dicts.

    Args:
        flattened_feature_metrics: Dict mapping metric names to lists of values

    Returns:
        List of CoreFeatureMetric objects, one per feature
    """
    feature_metrics_by_feature = []
    if flattened_feature_metrics:
        num_features = len(flattened_feature_metrics["consistent_activation_heuristic"])
        for i in range(num_features):
            feature_metrics_by_feature.append(
                CoreFeatureMetric(
                    index=i,
                    consistent_activation_heuristic=round(
                        flattened_feature_metrics["consistent_activation_heuristic"][i],
                        DEFAULT_FLOAT_PRECISION,
                    ),
                    encoder_bias=round(
                        flattened_feature_metrics["encoder_bias"][i],
                        DEFAULT_FLOAT_PRECISION,
                    ),
                    encoder_decoder_cosine_sim=round(
                        flattened_feature_metrics["encoder_decoder_cosine_sim"][i],
                        DEFAULT_FLOAT_PRECISION,
                    ),
                    encoder_norm=round(
                        flattened_feature_metrics["encoder_norm"][i],
                        DEFAULT_FLOAT_PRECISION,
                    ),
                    feature_density=round(
                        flattened_feature_metrics["feature_density"][i],
                        DEFAULT_FLOAT_PRECISION,
                    ),
                )
            )
    return feature_metrics_by_feature


def save_single_eval_result(
    result: Dict[str, Any],
    eval_instance_id: str,
    sae_lens_version: str,
    sae_bench_commit_hash: str,
    output_path: Path,
    sae: SAE,
) -> Path:
    """Save a single evaluation result to a JSON file."""
    # Get the eval_config directly - it's already a CoreEvalConfig object
    eval_config = result["eval_cfg"]

    # Create metric categories
    metric_categories = CoreMetricCategories(
        model_behavior_preservation=ModelBehaviorPreservationMetrics(
            **result["metrics"].get("model_behavior_preservation", {})
        ),
        model_performance_preservation=ModelPerformancePreservationMetrics(
            **result["metrics"].get("model_performance_preservation", {})
        ),
        reconstruction_quality=ReconstructionQualityMetrics(
            **result["metrics"].get("reconstruction_quality", {})
        ),
        shrinkage=ShrinkageMetrics(**result["metrics"].get("shrinkage", {})),
        sparsity=SparsityMetrics(**result["metrics"].get("sparsity", {})),
        token_stats=TokenStatsMetrics(**result["metrics"].get("token_stats", {})),
    )

    # Create feature metrics
    flattened_feature_metrics = result.get("feature_metrics", {})

    # Convert feature metrics from parallel lists to list of dicts
    feature_metrics_by_feature = convert_feature_metrics(flattened_feature_metrics)

    # Create the full output object
    eval_output = CoreEvalOutput(
        eval_config=eval_config,
        eval_id=eval_instance_id,
        datetime_epoch_millis=int(time.time() * 1000),
        eval_result_metrics=metric_categories,
        eval_result_details=feature_metrics_by_feature,
        eval_result_unstructured={},  # Add empty dict for unstructured results
        sae_bench_commit_hash=sae_bench_commit_hash,
        sae_lens_id=result["sae_id"],
        sae_lens_release_id=result["sae_set"],
        sae_lens_version=sae_lens_version,
        sae_cfg_dict=asdict(sae.cfg),
    )

    # Save individual JSON file
    json_filename = (
        f"{result['unique_id']}_{eval_config.context_size}_{eval_config.dataset}.json".replace(
            "/", "_"
        )
    )
    json_path = output_path / json_filename
    eval_output.to_json_file(json_path)

    return json_path


def multiple_evals(
    filtered_saes: list[tuple[str, str]] | list[tuple[str, SAE]],
    n_eval_reconstruction_batches: int,
    n_eval_sparsity_variance_batches: int,
    eval_batch_size_prompts: int = 8,
    compute_featurewise_density_statistics: bool = False,
    compute_featurewise_weight_based_metrics: bool = False,
    exclude_special_tokens_from_reconstruction: bool = False,
    dataset: str = "Skylion007/openwebtext",
    context_size: int = 128,
    output_folder: str = "eval_results",
    verbose: bool = False,
    dtype: str = "float32",
) -> List[Dict[str, Any]]:
    device = general_utils.setup_environment()
    assert len(filtered_saes) > 0, "No SAEs to evaluate"

    eval_results = []
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get evaluation metadata once at the start
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    multiple_evals_config = get_multiple_evals_everything_config(
        batch_size_prompts=eval_batch_size_prompts,
        n_eval_reconstruction_batches=n_eval_reconstruction_batches,
        n_eval_sparsity_variance_batches=n_eval_sparsity_variance_batches,
    )

    current_model = None
    current_model_str = None

    for sae_release_name, sae_id in tqdm(filtered_saes):
        # Wrap SAE loading with retry
        @retry_with_exponential_backoff(
            retries=5,
            exceptions=(
                Exception,
            ),  # You might want to be more specific about which exceptions to catch
            initial_delay=1.0,
            max_delay=60.0,
        )
        def load_sae():
            return SAE.from_pretrained(
                release=sae_release_name,
                sae_id=sae_id,
                device=device,
            )[0]

        del_sae = False
        # Handle both pretrained SAEs (identified by string) and custom SAEs (passed as objects)
        if isinstance(sae_id, str):
            try:
                sae = load_sae()
                del_sae = True
            except Exception as e:
                logger.error(f"Failed to load SAE {sae_id} from {sae_release_name}: {str(e)}")
                continue  # Skip this SAE and continue with the next one
        else:
            sae = sae_id
            sae_id = "custom_sae"

        sae.to(device)
        sae = sae.to(general_utils.str_to_dtype(dtype))

        # TODO: Check if results already exist and skip if so, add force_rerun flag

        if current_model_str != sae.cfg.model_name:
            # Wrap model loading with retry
            @retry_with_exponential_backoff(
                retries=5,
                exceptions=(
                    Exception,
                ),  # We might want to be more specific about which exceptions to catch
                initial_delay=1.0,
                max_delay=60.0,
            )
            def load_model():
                return HookedTransformer.from_pretrained_no_processing(
                    sae.cfg.model_name,
                    device=device,
                    dtype=sae.W_enc.dtype,
                    **sae.cfg.model_from_pretrained_kwargs,
                )

            try:
                del current_model
                current_model_str = sae.cfg.model_name
                current_model = load_model()
            except Exception as e:
                logger.error(f"Failed to load model {sae.cfg.model_name}: {str(e)}")
                continue  # Skip this SAE and continue with the next one

        assert current_model is not None

        try:
            # Create a CoreEvalConfig for this specific evaluation
            core_eval_config = CoreEvalConfig(
                model_name=sae.cfg.model_name,
                batch_size_prompts=multiple_evals_config.batch_size_prompts or 16,
                n_eval_reconstruction_batches=multiple_evals_config.n_eval_reconstruction_batches,
                n_eval_sparsity_variance_batches=multiple_evals_config.n_eval_sparsity_variance_batches,
                exclude_special_tokens_from_reconstruction=exclude_special_tokens_from_reconstruction,
                dataset=dataset,
                context_size=context_size,
                compute_kl=multiple_evals_config.compute_kl,
                compute_ce_loss=multiple_evals_config.compute_ce_loss,
                compute_l2_norms=multiple_evals_config.compute_l2_norms,
                compute_sparsity_metrics=multiple_evals_config.compute_sparsity_metrics,
                compute_variance_metrics=multiple_evals_config.compute_variance_metrics,
                compute_featurewise_density_statistics=compute_featurewise_density_statistics,
                compute_featurewise_weight_based_metrics=compute_featurewise_weight_based_metrics,
                llm_dtype=dtype,
            )

            # Wrap activation store creation with retry
            @retry_with_exponential_backoff(
                retries=3,
                exceptions=(Exception,),
                initial_delay=1.0,
                max_delay=30.0,
            )
            def create_activation_store():
                return ActivationsStore.from_sae(
                    current_model, sae, context_size=context_size, dataset=dataset
                )

            activation_store = create_activation_store()
            activation_store.shuffle_input_dataset(seed=42)

            eval_metrics = nested_dict()
            eval_metrics["unique_id"] = f"{sae_release_name}_{sae_id}".replace(".", "_")
            eval_metrics["sae_set"] = f"{sae_release_name}"
            eval_metrics["sae_id"] = f"{sae_id}"
            eval_metrics["eval_cfg"] = core_eval_config

            scalar_metrics, feature_metrics = run_evals(
                sae=sae,
                activation_store=activation_store,
                model=current_model,  # type: ignore
                eval_config=core_eval_config,
                ignore_tokens={
                    current_model.tokenizer.pad_token_id,  # type: ignore
                    current_model.tokenizer.eos_token_id,  # type: ignore
                    current_model.tokenizer.bos_token_id,  # type: ignore
                },
                verbose=verbose,
            )
            eval_metrics["metrics"] = scalar_metrics

            if compute_featurewise_density_statistics or compute_featurewise_weight_based_metrics:
                eval_metrics["feature_metrics"] = feature_metrics

            # Clean NaN values before saving
            cleaned_metrics = replace_nans_with_negative_one(eval_metrics)

            # Save results immediately after each evaluation
            saved_path = save_single_eval_result(
                cleaned_metrics,
                eval_instance_id,
                sae_lens_version,
                sae_bench_commit_hash,
                output_path,
                sae,
            )

            if verbose:
                print(f"Saved evaluation results to: {saved_path}")

            eval_results.append(eval_metrics)
        except Exception as e:
            logger.error(
                f"Failed to evaluate SAE {sae_id} from {sae_release_name} "
                f"with context length {context_size} on dataset {dataset}: {str(e)}"
            )
            continue  # Skip this combination and continue with the next one

        if del_sae:
            del sae
        gc.collect()
        torch.cuda.empty_cache()

    return eval_results


def run_evaluations(args: argparse.Namespace) -> List[Dict[str, Any]]:
    # Filter SAEs based on regex patterns
    filtered_saes = sae_selection_utils.get_saes_from_regex(
        args.sae_regex_pattern, args.sae_block_pattern
    )

    # print the filtered SAEs
    print("Filtered SAEs based on provided patterns:")
    for sae in filtered_saes:
        print(sae)

    num_sae_sets = len(set(sae_set for sae_set, _ in filtered_saes))
    num_all_sae_ids = len(filtered_saes)

    print("Filtered SAEs based on provided patterns:")
    print(f"Number of SAE sets: {num_sae_sets}")
    print(f"Total number of SAE IDs: {num_all_sae_ids}")

    eval_results = multiple_evals(
        filtered_saes=filtered_saes,
        n_eval_reconstruction_batches=args.n_eval_reconstruction_batches,
        n_eval_sparsity_variance_batches=args.n_eval_sparsity_variance_batches,
        eval_batch_size_prompts=args.batch_size_prompts,
        compute_featurewise_density_statistics=args.compute_featurewise_density_statistics,
        compute_featurewise_weight_based_metrics=args.compute_featurewise_weight_based_metrics,
        exclude_special_tokens_from_reconstruction=args.exclude_special_tokens_from_reconstruction,
        dataset=args.dataset,
        context_size=args.context_size,
        output_folder=args.output_folder,
        verbose=args.verbose,
        dtype=args.llm_dtype,
    )

    return eval_results


def replace_nans_with_negative_one(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: replace_nans_with_negative_one(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nans_with_negative_one(item) for item in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return -1
    else:
        return obj


def arg_parser():
    parser = argparse.ArgumentParser(description="Run core evaluation")
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        help="Model name. Currently this flag is ignored and the model name is inferred from sae.cfg.model_name.",
    )
    parser.add_argument(
        "sae_regex_pattern",
        type=str,
        help="Regex pattern to match SAE names. Can be an entire SAE name to match a specific SAE.",
    )
    parser.add_argument(
        "sae_block_pattern",
        type=str,
        help="Regex pattern to match SAE block names. Can be an entire block name to match a specific block.",
    )
    parser.add_argument(
        "--batch_size_prompts",
        type=int,
        default=16,
        help="Batch size for evaluation prompts.",
    )
    parser.add_argument(
        "--n_eval_reconstruction_batches",
        type=int,
        default=10,
        help="Number of evaluation batches for reconstruction metrics.",
    )
    parser.add_argument(
        "--compute_kl",
        action="store_true",
        help="Compute KL divergence.",
    )
    parser.add_argument(
        "--compute_ce_loss",
        action="store_true",
        help="Compute cross-entropy loss.",
    )
    parser.add_argument(
        "--n_eval_sparsity_variance_batches",
        type=int,
        default=1,
        help="Number of evaluation batches for sparsity and variance metrics.",
    )
    parser.add_argument(
        "--compute_l2_norms",
        action="store_true",
        help="Compute L2 norms.",
    )
    parser.add_argument(
        "--compute_sparsity_metrics",
        action="store_true",
        help="Compute sparsity metrics.",
    )
    parser.add_argument(
        "--compute_variance_metrics",
        action="store_true",
        help="Compute variance metrics.",
    )
    parser.add_argument(
        "--compute_featurewise_density_statistics",
        action="store_true",
        help="Compute featurewise density statistics.",
    )
    parser.add_argument(
        "--compute_featurewise_weight_based_metrics",
        action="store_true",
        help="Compute featurewise weight-based metrics.",
    )
    parser.add_argument(
        "--exclude_special_tokens_from_reconstruction",
        action="store_true",
        help="Exclude special tokens like BOS, EOS, PAD from reconstruction.",
    )
    parser.add_argument(
        "--dataset",
        default="Skylion007/openwebtext",
        help="Dataset to evaluate on, such as 'Skylion007/openwebtext' or 'lighteval/MATH'.",
    )
    parser.add_argument(
        "--context_size",
        type=int,
        default=128,
        help="Context size to evaluate on.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with tqdm loaders.",
    )
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun of experiments")
    parser.add_argument(
        "--llm_dtype",
        type=str,
        default="float32",
        choices=["float32", "float64", "float16", "bfloat16"],
        help="Data type for computation",
    )

    return parser


if __name__ == "__main__":
    """
    python evals/core/main.py "sae_bench_pythia70m_sweep_standard_ctx128_0712" "blocks.4.hook_resid_post__trainer_10" \
    --batch_size_prompts 16 \
    --n_eval_sparsity_variance_batches 2000 \
    --n_eval_reconstruction_batches 200 \
    --output_folder "eval_results/core" \
    --exclude_special_tokens_from_reconstruction --verbose

    python evals/core/main.py "sae_bench_gemma-2-2b_topk_width-2pow14_date-1109" "blocks.19.hook_resid_post__trainer_2" \
    --batch_size_prompts 16 \
    --n_eval_sparsity_variance_batches 2000 \
    --n_eval_reconstruction_batches 200 \
    --output_folder "eval_results/core" \
    --exclude_special_tokens_from_reconstruction --verbose --llm_dtype bfloat16
    """
    args = arg_parser().parse_args()
    eval_results = run_evaluations(args)

    print("Evaluation complete. All results have been saved incrementally.")  # type: ignore
    # print(f"Combined JSON: {output_files['combined_json']}")
    # print(f"CSV: {output_files['csv']}")


# Use this code snippet to use custom SAE objects
# if __name__ == "__main__":
#     import custom_saes.identity_sae as identity_sae
#     import custom_saes.jumprelu_sae as jumprelu_sae

#     start_time = time.time()

#     random_seed = 42
#     output_folder = "eval_results/core"

#     batch_size_prompts = 16
#     n_eval_reconstruction_batches = 20
#     n_eval_sparsity_variance_batches = 20
#     context_size = 128
#     dataset_name = "Skylion007/openwebtext"
#     exclude_special_tokens_from_reconstruction = True

#     model_name = "gemma-2-2b"
#     hook_layer = 20
#     llm_dtype = torch.bfloat16

#     repo_id = "google/gemma-scope-2b-pt-res"
#     filename = f"layer_{hook_layer}/width_16k/average_l0_71/params.npz"
#     sae = jumprelu_sae.load_jumprelu_sae(repo_id, filename, hook_layer)
#     selected_saes = [(f"{repo_id}_{filename}_gemmascope_sae", sae)]

#     # it's recommended to specify the dtype of the SAE
#     for sae_name, sae in selected_saes:
#         sae.cfg.dtype = "bfloat16"

#     multiple_evals(
#         filtered_saes=selected_saes,
#         n_eval_reconstruction_batches=n_eval_reconstruction_batches,
#         n_eval_sparsity_variance_batches=n_eval_sparsity_variance_batches,
#         eval_batch_size_prompts=batch_size_prompts,
#         exclude_special_tokens_from_reconstruction=exclude_special_tokens_from_reconstruction,
#         dataset=dataset_name,
#         context_size=context_size,
#         output_folder=output_folder,
#         verbose=True,
#         dtype=llm_dtype,
#     )

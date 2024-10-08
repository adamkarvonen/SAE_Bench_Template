import gc
import json
import os
import random
import select
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional, Protocol

import torch as t
from collectibles import ListCollection
from einops import rearrange
from eval_config import EvalConfig
from loguru import logger
from sae_lens import SAE, ActivationsStore
from sae_lens.sae import TopK
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from torch import nn
from tqdm import tqdm
from transformer_lens import HookedTransformer

sys.path.append("/Users/Kola/Documents/VSCode/open_source/SAE_Bench_Template/sae_bench_utils")

from sae_bench_utils import activation_collection, formatting_utils


class Decodable(Protocol):
    def decode(self, x: t.Tensor) -> t.Tensor: ...


def build_bins(
    feature_activations_BsF: t.Tensor,
    bin_precision: Optional[float] = None,  # 0.2,
    num_bins: Optional[int] = None,  # 16)
) -> list[t.Tensor]:
    if bin_precision is not None and num_bins is not None:
        raise ValueError("Only one of bin_precision or num_bins should be provided")
    if bin_precision is None and num_bins is None:
        raise ValueError("Either bin_precision or num_bins should be provided")

    _, num_features = feature_activations_BsF.shape

    max_activations_F = t.max(feature_activations_BsF, dim=-1).values

    # positive_mask_BsF = feature_activations_BsF > 0
    # masked_activations_BsF = t.where(positive_mask_BsF, feature_activations_BsF, t.inf)
    # min_pos_activations_F = t.min(masked_activations_BsF, dim=-1).values
    # min_pos_activations_F = t.where(
    #     t.isfinite(min_pos_activations_F), min_pos_activations_F, 0
    # )
    min_pos_activations_F = t.zeros_like(max_activations_F)

    logger.debug(max_activations_F)
    logger.debug(min_pos_activations_F)

    bins_F_list_Bi: list[t.Tensor] = []

    if bin_precision is not None:
        for feature_idx in range(num_features):
            bins = t.arange(
                min_pos_activations_F[feature_idx].item(),
                max_activations_F[feature_idx].item() + 2 * bin_precision,
                bin_precision,
            )
            bins_F_list_Bi.append(bins)

        return bins_F_list_Bi

    else:
        assert num_bins is not None
        for feature_idx in range(num_features):
            bins = t.linspace(
                min_pos_activations_F[feature_idx].item(),
                max_activations_F[feature_idx].item(),
                num_bins + 1,
            )
            bins_F_list_Bi.append(bins)

        return bins_F_list_Bi


def _calculate_dl(
    feature_activations_BsF: t.Tensor,
    bins_F_list_Bi: list[t.Tensor],
) -> float:
    _bs, num_features = feature_activations_BsF.shape

    bool_prob_F = (feature_activations_BsF > 0).float().mean(dim=0)

    bool_entropy_F = t.zeros(num_features)
    for feature_idx in range(num_features):
        bool_prob = bool_prob_F[feature_idx]
        if bool_prob == 0 or bool_prob == 1:
            bool_entropy = 0
        else:
            bool_entropy = -bool_prob * t.log2(bool_prob) - (1 - bool_prob) * t.log2(
                1 - bool_prob
            )
        bool_entropy_F[feature_idx] = bool_entropy

    float_entropy_F = t.zeros(num_features)

    for feature_idx in range(num_features):
        feature_activations_Bs = feature_activations_BsF[:, feature_idx]
        bins = bins_F_list_Bi[feature_idx]

        counts, _bin_edges = t.histogram(feature_activations_Bs, bins=bins)

        probs_Bi = counts / counts.sum()
        probs_Bi = probs_Bi[(probs_Bi > 0) & (probs_Bi < 1)]

        if len(probs_Bi) == 0:
            float_entropy = 0
        else:
            # H[p] = -sum(p * log2(p))
            float_entropy = -t.sum(probs_Bi * t.log2(probs_Bi))

        float_entropy_F[feature_idx] = float_entropy

    total_entropy_F = bool_entropy_F + bool_prob_F * float_entropy

    description_length = total_entropy_F.sum().item()

    return description_length


def quantize_features_to_bin_midpoints(
    features_BF: t.Tensor, bins_F_list_Bi: list[t.Tensor]
) -> t.Tensor:
    """
    Quantize features to the bin midpoints of their corresponding histograms.
    """
    _, num_features = features_BF.shape

    quantized_features_BF = t.empty_like(features_BF)

    for feature_idx in range(num_features):
        # Extract the feature values and bin edges for the current histogram
        features_B = features_BF[:, feature_idx]
        bin_edges_Bi = bins_F_list_Bi[feature_idx]

        num_bins = len(bin_edges_Bi) - 1

        bin_indices_B = t.bucketize(features_B, bin_edges_Bi)
        bin_indices_clipped_B = t.clamp(bin_indices_B, min=1, max=num_bins) - 1

        # Calculate the midpoints of the bins
        bin_mids_Bi = 0.5 * (bin_edges_Bi[:-1] + bin_edges_Bi[1:])

        quantized_features_BF[:, feature_idx] = bin_mids_Bi[bin_indices_clipped_B]

    return quantized_features_BF


def calculate_dl(
    feature_activations: t.Tensor,
    k: Optional[int] = None,
    *,
    bin_precision: Optional[float] = None,
    num_bins: Optional[int] = None,
    bins: Optional[list[t.Tensor]] = None,
) -> float:
    if feature_activations.ndim == 2:
        feature_activations_BsF = feature_activations
    elif feature_activations.ndim == 3:
        feature_activations_BsF = rearrange(
            feature_activations,
            "batch seq_len num_features -> (batch seq_len) num_features",
        )
    else:
        raise ValueError("feature_activations should be 2D or 3D tensor")

    if k is not None:
        feature_activations_BsF = t.topk(feature_activations_BsF, k=k, dim=-1).values

    bins = (
        build_bins(feature_activations_BsF, bin_precision, num_bins) if bins is None else bins
    )
    entropy = _calculate_dl(feature_activations_BsF, bins)
    return entropy


def check_quantised_features_reach_mse_threshold(
    feature_activations_BsF: t.Tensor,
    bins_F_list_Bi: list[t.Tensor],
    mse_threshold: float,
    x_BsF: t.Tensor,
    autoencoder: SAE,
    k: Optional[int] = None,
) -> tuple[bool, float]:

    if k is not None:
        feature_activations_BsF = t.topk(feature_activations_BsF, k=k, dim=-1).values

    quantised_feature_activations_BsF = quantize_features_to_bin_midpoints(
        feature_activations_BsF, bins_F_list_Bi
    )
    reconstructed_x_BsF: t.Tensor = autoencoder.decode(quantised_feature_activations_BsF)

    mse_criterion = nn.MSELoss()
    mse_loss: t.Tensor = mse_criterion(reconstructed_x_BsF, x_BsF)

    within_threshold = bool((mse_loss < mse_threshold).item())

    return within_threshold, mse_loss.item()


class IdentityAE(nn.Module):
    def forward(self, x):
        return x

    def decode(self, x):
        return x


@dataclass
class MDLEvalResult:
    num_bins: int
    bins: list[t.Tensor]
    k: Optional[int]

    description_length: float
    within_threshold: bool
    mse_loss: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MDLEvalResultsCollection(ListCollection[MDLEvalResult]):
    num_bins: list[int]
    bins: list[list[t.Tensor]]
    k: list[Optional[int]]

    description_length: list[float]
    within_threshold: list[bool]
    mse_loss: list[float]

    def pick_minimum_viable(self) -> MDLEvalResult:
        all_description_lengths = t.tensor(self.description_length)
        threshold_mask = t.tensor(self.within_threshold)

        min_dl_idx = int(t.argmin(all_description_lengths[threshold_mask]).item())

        return self[min_dl_idx]


def _run_single_eval(
    config: EvalConfig,
    sae: SAE,
    dataset_name: str,
    model: HookedTransformer,
    device: str,
) -> MDLEvalResult:
    mdl_eval_results_list: list[MDLEvalResult] = []

    # llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
    # llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

    # train_df, test_df = dataset_utils.load_huggingface_dataset(dataset_name)
    # train_data, test_data = dataset_utils.get_multi_label_train_test_data(
    #     train_df,
    #     test_df,
    #     dataset_name,
    #     config.sae_batch_size,
    #     config.sae_batch_size,
    #     config.random_seed,
    # )

    # chosen_classes = dataset_info.chosen_classes_per_dataset[dataset_name]

    # train_data = dataset_utils.filter_dataset(train_data, chosen_classes)
    # test_data = dataset_utils.filter_dataset(test_data, chosen_classes)

    # train_data = dataset_utils.tokenize_data(
    #     train_data, model.tokenizer, config.context_length, device
    # )
    # test_data = dataset_utils.tokenize_data(
    #     test_data, model.tokenizer, config.context_length, device
    # )

    activations_store = ActivationsStore.from_sae(
        model, sae, config.sae_batch_size, dataset="EleutherAI/pile", device=device
    )

    # print(f"Running evaluation for layer {config.layer}")
    # hook_name = f"blocks.{config.layer}.hook_resid_post"

    neuron_activations_BSN = activations_store.get_buffer(config.sae_batch_size)

    feature_activations_BSF = sae.encode(neuron_activations_BSN)

    for num_bins in config.num_bins_values:
        for k in config.k_values:
            bins = build_bins(feature_activations_BSF, num_bins=num_bins)

            within_threshold, mse_loss = check_quantised_features_reach_mse_threshold(
                feature_activations_BsF=feature_activations_BSF,
                bins_F_list_Bi=bins,
                mse_threshold=config.mse_epsilon_threshold,
                x_BsF=feature_activations_BSF,
                autoencoder=sae,
                k=k,
            )

        description_length = calculate_dl(feature_activations_BSF, num_bins=2)

        mdl_eval_results_list.append(
            MDLEvalResult(
                num_bins=num_bins,
                bins=bins,
                k=k,
                description_length=description_length,
                within_threshold=within_threshold,
                mse_loss=mse_loss,
            )
        )

    mdl_eval_results = MDLEvalResultsCollection(mdl_eval_results_list)

    minimum_viable_eval_result = mdl_eval_results.pick_minimum_viable()

    minimum_viable_description_length = minimum_viable_eval_result.description_length
    logger.debug(minimum_viable_description_length)

    return minimum_viable_eval_result


def run_eval(
    config: EvalConfig,
    selected_saes_dict: dict[str, list[str]],
    device: str,
) -> dict[str, Any]:
    results_dict = {}

    random.seed(config.random_seed)
    t.manual_seed(config.random_seed)

    llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]
    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

    for dataset_name in config.dataset_names:
        for sae_release_name, sae_specific_names in selected_saes_dict.items():
            for sae_specific_name in sae_specific_names:
                sae, _, _ = SAE.from_pretrained(
                    sae_release_name, sae_specific_name, device=device
                )

                eval_result = _run_single_eval(config, sae, dataset_name, model, device)
                results_dict[f"{dataset_name}_{sae_specific_name}_results"] = (
                    eval_result.to_dict()
                )

    results_dict["custom_eval_config"] = asdict(config)
    results_dict["custom_eval_results"] = formatting_utils.average_results_dictionaries(
        results_dict, config.dataset_names
    )

    return results_dict


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    # feature_activations_BSF = t.relu(t.randn(10, 5))

    # minimum_viable_description_length = _run_single_eval(
    #     feature_activations_BSF,
    # )
    # print(minimum_viable_description_length)
    # pass

    start_time = time.time()

    if t.backends.mps.is_available():
        # device = "mps"
        device = "cpu"
    else:
        device = "cuda" if t.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    config = EvalConfig()

    # populate selected_saes_dict using config values
    for release in config.sae_releases:
        if "gemma-scope" in release:
            config.selected_saes_dict[release] = (
                formatting_utils.find_gemmascope_average_l0_sae_names(config.layer)
            )
        else:
            config.selected_saes_dict[release] = formatting_utils.filter_sae_names(
                sae_names=release,
                layers=[config.layer],
                include_checkpoints=config.include_checkpoints,
                trainer_ids=config.trainer_ids,
            )

        print(f"SAE release: {release}, SAEs: {config.selected_saes_dict[release]}")

    results_dict = run_eval(config, config.selected_saes_dict, device)

    # create output filename and save results
    checkpoints_str = ""
    if config.include_checkpoints:
        checkpoints_str = "_with_checkpoints"

    output_filename = (
        config.model_name + f"_layer_{config.layer}{checkpoints_str}_eval_results.json"
    )
    output_folder = "results"  # at evals/<eval_name>

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    output_location = os.path.join(output_folder, output_filename)

    with open(output_location, "w") as f:
        json.dump(results_dict, f)

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time} seconds")

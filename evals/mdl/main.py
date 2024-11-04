import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional, Protocol

import torch as t
import torch.nn.functional as F
from collectibles import ListCollection
from einops import rearrange
from loguru import logger
from sae_lens import SAE, ActivationsStore
from sae_lens.sae import TopK
from torch import nn
from transformer_lens import HookedTransformer
import argparse
from datetime import datetime

from evals.mdl.eval_config import MDLEvalConfig
from sae_bench_utils import activation_collection, formatting_utils
from sae_bench_utils import (
    get_eval_uuid,
    get_sae_lens_version,
    get_sae_bench_version,
)
from sae_bench_utils.sae_selection_utils import (
    get_saes_from_regex,
    select_saes_multiple_patterns,
)


class Decodable(Protocol):
    def decode(self, x: t.Tensor) -> t.Tensor: ...


def build_bins(
    min_pos_activations_F: t.Tensor,
    max_activations_F: t.Tensor,
    bin_precision: Optional[float] = None,  # 0.2,
    num_bins: Optional[int] = None,  # 16)
) -> list[t.Tensor]:
    if bin_precision is not None and num_bins is not None:
        raise ValueError("Only one of bin_precision or num_bins should be provided")
    if bin_precision is None and num_bins is None:
        raise ValueError("Either bin_precision or num_bins should be provided")

    num_features = len(max_activations_F)

    assert len(max_activations_F) == num_features

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
                device=max_activations_F.device,
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
                device=max_activations_F.device,
            )
            bins_F_list_Bi.append(bins)

        return bins_F_list_Bi


def calculate_dl(
    num_features: int,
    bins_F_list_Bi: list[t.Tensor],
    device: str,
    activations_store: ActivationsStore,
    sae: SAE,
    k: int,
) -> float:
    float_entropy_F = t.zeros(num_features, device=device)
    bool_entropy_F = t.zeros(num_features, device=device)

    x_BSN = activations_store.get_buffer(config.sae_batch_size)
    feature_activations_BsF = sae.encode(x_BSN).squeeze()

    if feature_activations_BsF.ndim == 2:
        feature_activations_BsF = feature_activations_BsF
    elif feature_activations_BsF.ndim == 3:
        feature_activations_BsF = rearrange(
            feature_activations_BsF,
            "batch seq_len num_features -> (batch seq_len) num_features",
        )
    else:
        raise ValueError("feature_activations should be 2D or 3D tensor")

    for feature_idx in range(num_features):
        # BOOL entropy
        bool_prob = t.zeros(1, device=device)

        bool_prob_F = (feature_activations_BsF > 0).float().mean(dim=0)
        bool_prob = bool_prob + bool_prob_F[feature_idx]

        if bool_prob == 0 or bool_prob == 1:
            bool_entropy = 0
        else:
            bool_entropy = -bool_prob * t.log2(bool_prob) - (1 - bool_prob) * t.log2(1 - bool_prob)
        bool_entropy_F[feature_idx] = bool_entropy

        # FLOAT entropy
        num_bins = len(bins_F_list_Bi[feature_idx]) - 1
        counts_Bi = t.zeros(num_bins, device="cpu")

        feature_activations_Bs = feature_activations_BsF[:, feature_idx]
        bins = bins_F_list_Bi[feature_idx]

        temp_counts_Bi, _bin_edges = t.histogram(feature_activations_Bs.cpu(), bins=bins.cpu())
        counts_Bi = counts_Bi + temp_counts_Bi

        counts_Bi = counts_Bi.to(device)

        probs_Bi = counts_Bi / counts_Bi.sum()
        probs_Bi = probs_Bi[(probs_Bi > 0) & (probs_Bi < 1)]

        if len(probs_Bi) == 0:
            float_entropy = 0
        else:
            # H[p] = -sum(p * log2(p))
            float_entropy = -t.sum(probs_Bi * t.log2(probs_Bi)).item()

        float_entropy_F[feature_idx] = float_entropy

    total_entropy_F = bool_entropy_F.cuda() + bool_prob_F.cuda() * float_entropy_F.cuda()

    description_length = total_entropy_F.sum().item()

    return description_length


def quantize_features_to_bin_midpoints(
    features_BF: t.Tensor, bins_F_list_Bi: list[t.Tensor]
) -> t.Tensor:
    """
    Quantize features to the bin midpoints of their corresponding histograms.
    """
    _, num_features = features_BF.shape

    quantized_features_BF = t.empty_like(features_BF, device=features_BF.device)

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


# def calculate_dl(
#     activations_store: ActivationsStore,
#     sae: SAE,
#     bins: list[t.Tensor],
#     k: Optional[int] = None,
# ) -> float:
#     for i in range(10):
#         x_BSN = activations_store.get_buffer(config.sae_batch_size)
#         feature_activations_BsF = sae.encode(x_BSN).squeeze()

#         if feature_activations_BsF.ndim == 2:
#             feature_activations_BsF = feature_activations_BsF
#         elif feature_activations_BsF.ndim == 3:
#             feature_activations_BsF = rearrange(
#                 feature_activations_BsF,
#                 "batch seq_len num_features -> (batch seq_len) num_features",
#             )
#         else:
#             raise ValueError("feature_activations should be 2D or 3D tensor")

#         if k is not None:
#             topk_fn = TopK(k)
#             feature_activations_BsF = topk_fn(feature_activations_BsF)

#         entropy = _calculate_dl_single(feature_activations_BsF, bins)
#     return entropy


def check_quantised_features_reach_mse_threshold(
    bins_F_list_Bi: list[t.Tensor],
    activations_store: ActivationsStore,
    sae: SAE,
    mse_threshold: float,
    autoencoder: SAE,
    k: Optional[int] = None,
) -> tuple[bool, float]:
    mse_losses: list[t.Tensor] = []

    for i in range(1):
        x_BSN = activations_store.get_buffer(config.sae_batch_size)
        feature_activations_BSF = sae.encode(x_BSN).squeeze()

        if k is not None:
            topk_fn = TopK(k)
            feature_activations_BSF = topk_fn(feature_activations_BSF)

        quantised_feature_activations_BsF = quantize_features_to_bin_midpoints(
            feature_activations_BSF, bins_F_list_Bi
        )

        reconstructed_x_BSN: t.Tensor = autoencoder.decode(quantised_feature_activations_BsF)

        mse_loss: t.Tensor = F.mse_loss(reconstructed_x_BSN, x_BSN.squeeze(), reduction="mean")
        mse_loss = t.sqrt(mse_loss) / sae.cfg.d_in
        mse_losses.append(mse_loss)

    avg_mse_loss = t.mean(t.stack(mse_losses))
    within_threshold = bool((avg_mse_loss < mse_threshold).item())

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
        out = asdict(self)
        out["bins"] = []
        return out


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

        viable_description_lengths = all_description_lengths[threshold_mask]
        if len(viable_description_lengths) > 0:
            min_dl_idx = int(t.argmin(viable_description_lengths).item())
            return self[min_dl_idx]

        else:
            min_dl_idx = int(t.argmin(all_description_lengths).item())
            return self[min_dl_idx]


def _run_single_eval(
    config: MDLEvalConfig,
    sae: SAE,
    model: HookedTransformer,
    device: str,
    dataset_name: str = "HuggingFaceFW/fineweb",
) -> MDLEvalResult:
    mdl_eval_results_list: list[MDLEvalResult] = []

    t.set_grad_enabled(False)

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
    sae.cfg.dataset_trust_remote_code = True
    sae = sae.to(device)
    model = model.to(device)  # type: ignore

    activations_store = ActivationsStore.from_sae(
        model, sae, config.sae_batch_size, dataset=dataset_name, device=device
    )

    # print(f"Running evaluation for layer {config.layer}")
    # hook_name = f"blocks.{config.layer}.hook_resid_post"
    num_features = sae.cfg.d_sae

    def get_min_max_activations() -> tuple[t.Tensor, t.Tensor]:
        min_pos_activations_1F = t.zeros(1, num_features, device=device)
        max_activations_1F = t.zeros(1, num_features, device=device) + 100

        for _ in range(10):
            neuron_activations_BSN = activations_store.get_buffer(config.sae_batch_size)

            feature_activations_BsF = sae.encode(neuron_activations_BSN).squeeze()

            cat_feature_activations_BsF = t.cat(
                [
                    feature_activations_BsF,
                    min_pos_activations_1F,
                    max_activations_1F,
                ],
                dim=0,
            )
            min_pos_activations_1F = t.min(cat_feature_activations_BsF, dim=0).values.unsqueeze(0)
            max_activations_1F = t.max(cat_feature_activations_BsF, dim=0).values.unsqueeze(0)

        min_pos_activations_F = min_pos_activations_1F.squeeze()
        max_activations_F = max_activations_1F.squeeze()

        return min_pos_activations_F, max_activations_F

    min_pos_activations_F, max_activations_F = get_min_max_activations()

    print("num_bins_values", config.num_bins_values)
    print("k_values", config.k_values)

    for num_bins in config.num_bins_values:
        for k in config.k_values:
            assert k is not None

            bins = build_bins(min_pos_activations_F, max_activations_F, num_bins=num_bins)

            # print("Built bins")

            within_threshold, mse_loss = check_quantised_features_reach_mse_threshold(
                bins_F_list_Bi=bins,
                activations_store=activations_store,
                sae=sae,
                mse_threshold=config.mse_epsilon_threshold,
                autoencoder=sae,
                k=k,
            )
            if not within_threshold:
                logger.warning(
                    f"mse_loss for num_bins = {num_bins} and k = {k} is {mse_loss}, which is not within threshold"
                )
                continue

            # print("Checked threshold")

            description_length = calculate_dl(
                num_features=num_features,
                bins_F_list_Bi=bins,
                device=device,
                activations_store=activations_store,
                sae=sae,
                k=k,
            )

            logger.info(
                f"Description length: {description_length} for num_bins = {num_bins} and k = {k}"
            )

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
    logger.info(minimum_viable_description_length)

    return minimum_viable_eval_result


def run_eval(
    config: MDLEvalConfig,
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
                try:
                    sae, _, _ = SAE.from_pretrained(
                        sae_release_name, sae_specific_name, device=device
                    )
                except ValueError as e:
                    logger.error(
                        f"Error loading SAE {sae_specific_name} from {sae_release_name}: {e}"
                    )
                    sae_specific_name = "blocks.3.hook_resid_post__trainer_10"
                    sae, _, _ = SAE.from_pretrained(
                        sae_release_name, sae_specific_name, device=device
                    )

                eval_result = _run_single_eval(
                    config=config,
                    sae=sae,
                    model=model,
                    dataset_name=dataset_name,
                    device=device,
                )
                results_dict[f"{dataset_name}_{sae_specific_name}_results"] = eval_result.to_dict()

    results_dict["custom_eval_config"] = asdict(config)
    # results_dict["custom_eval_results"] = formatting_utils.average_results_dictionaries(
    #     results_dict, config.dataset_names
    # )

    return results_dict


def setup_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if t.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if t.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    return device


def create_config_and_selected_saes(
    args,
) -> tuple[MDLEvalConfig, dict[str, list[str]]]:
    config = MDLEvalConfig(
        random_seed=args.random_seed,
        model_name=args.model_name,
    )

    selected_saes_dict = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)

    assert len(selected_saes_dict) > 0, "No SAEs selected"

    for release, saes in selected_saes_dict.items():
        print(f"SAE release: {release}, Number of SAEs: {len(saes)}")
        print(f"Sample SAEs: {saes[:5]}...")

    return config, selected_saes_dict


def arg_parser():
    parser = argparse.ArgumentParser(description="Run sparse probing evaluation")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model_name", type=str, default="pythia-70m-deduped", help="Model name")
    parser.add_argument(
        "--sae_regex_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE selection",
    )
    parser.add_argument(
        "--sae_block_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE block selection",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="evals/sparse_probing/results",
        help="Output folder",
    )
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun of experiments")
    parser.add_argument(
        "--clean_up_activations",
        action="store_false",
        help="Clean up activations after evaluation",
    )

    return parser


if __name__ == "__main__":
    """python main.py \
    --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
    --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
    --model_name pythia-70m-deduped """
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    args = arg_parser().parse_args()
    device = setup_environment()

    start_time = time.time()

    sae_regex_patterns = [
        r"(sae_bench_pythia70m_sweep_topk_ctx128_0730).*",
        r"(sae_bench_pythia70m_sweep_standard_ctx128_0712).*",
    ]
    sae_block_pattern = [
        r".*blocks\.([4])\.hook_resid_post__trainer_(2|6|10|14)$",
        r".*blocks\.([4])\.hook_resid_post__trainer_(2|6|10|14)$",
    ]

    sae_regex_patterns = None
    sae_block_pattern = None

    config, selected_saes_dict = create_config_and_selected_saes(args)

    if sae_regex_patterns is not None:
        selected_saes_dict = select_saes_multiple_patterns(sae_regex_patterns, sae_block_pattern)

    print(selected_saes_dict)

    # config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
    # config.llm_dtype = str(activation_collection.LLM_NAME_TO_DTYPE[config.model_name]).split(".")[
    #     -1
    # ]

    # create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    config = MDLEvalConfig(
        k_values=[12, 16, 24, 32],
        num_bins_values=[8, 12, 16, 32, 64, 128],
        mse_epsilon_threshold=0.2,
    )
    logger.info(config)

    results_dict = run_eval(config, selected_saes_dict, device)

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

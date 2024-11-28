import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Optional, Protocol

import torch
import torch.nn.functional as F
from collectibles import ListCollection
from einops import rearrange
from loguru import logger
from sae_lens import SAE, ActivationsStore
from sae_lens.sae import TopK
from torch import nn
import gc
from transformer_lens import HookedTransformer
import argparse
from datetime import datetime
from tqdm import tqdm

from evals.mdl.eval_config import MDLEvalConfig
from sae_bench_utils import activation_collection, general_utils
from sae_bench_utils import (
    get_eval_uuid,
    get_sae_lens_version,
    get_sae_bench_version,
)
from sae_bench_utils.sae_selection_utils import (
    get_saes_from_regex,
    select_saes_multiple_patterns,
)

EVAL_TYPE = "mdl"


class Decodable(Protocol):
    def decode(self, x: torch.Tensor) -> torch.Tensor: ...


def build_bins(
    min_pos_activations_F: torch.Tensor,
    max_activations_F: torch.Tensor,
    bin_precision: Optional[float] = None,  # 0.2,
    num_bins: Optional[int] = None,  # 16)
) -> list[torch.Tensor]:
    if bin_precision is not None and num_bins is not None:
        raise ValueError("Only one of bin_precision or num_bins should be provided")
    if bin_precision is None and num_bins is None:
        raise ValueError("Either bin_precision or num_bins should be provided")

    num_features = len(max_activations_F)

    assert len(max_activations_F) == num_features

    # positive_mask_BsF = feature_activations_BsF > 0
    # masked_activations_BsF = torch.where(positive_mask_BsF, feature_activations_BsF, torch.inf)
    # min_pos_activations_F = torch.min(masked_activations_BsF, dim=-1).values
    # min_pos_activations_F = torch.where(
    #     torch.isfinite(min_pos_activations_F), min_pos_activations_F, 0
    # )
    min_pos_activations_F = torch.zeros_like(max_activations_F)

    logger.debug(max_activations_F)
    logger.debug(min_pos_activations_F)

    bins_F_list_Bi: list[torch.Tensor] = []

    if bin_precision is not None:
        for feature_idx in range(num_features):
            bins = torch.arange(
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
            bins = torch.linspace(
                min_pos_activations_F[feature_idx].item(),
                max_activations_F[feature_idx].item(),
                num_bins + 1,
                device=max_activations_F.device,
            )
            bins_F_list_Bi.append(bins)

        return bins_F_list_Bi


def calculate_dl(
    num_features: int,
    bins_F_list_Bi: list[torch.Tensor],
    device: str,
    activations_store: ActivationsStore,
    sae: SAE,
    k: int,
) -> float:
    float_entropy_F = torch.zeros(num_features, device=device, dtype=torch.float32)
    bool_entropy_F = torch.zeros(num_features, device=device, dtype=torch.float32)

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

    for feature_idx in tqdm(range(num_features), desc="Calculating DL"):
        # BOOL entropy
        bool_prob = torch.zeros(1, device=device)

        bool_prob_F = (feature_activations_BsF > 0).float().mean(dim=0)
        bool_prob = bool_prob + bool_prob_F[feature_idx]

        if bool_prob == 0 or bool_prob == 1:
            bool_entropy = 0
        else:
            bool_entropy = -bool_prob * torch.log2(bool_prob) - (1 - bool_prob) * torch.log2(
                1 - bool_prob
            )
        bool_entropy_F[feature_idx] = bool_entropy

        # FLOAT entropy
        num_bins = len(bins_F_list_Bi[feature_idx]) - 1
        counts_Bi = torch.zeros(num_bins, device="cpu")

        feature_activations_Bs = feature_activations_BsF[:, feature_idx].to(dtype=torch.float32)
        bins = bins_F_list_Bi[feature_idx]

        temp_counts_Bi, _bin_edges = torch.histogram(feature_activations_Bs.cpu(), bins=bins.cpu())
        counts_Bi = counts_Bi + temp_counts_Bi

        counts_Bi = counts_Bi.to(device)

        probs_Bi = counts_Bi / counts_Bi.sum()
        probs_Bi = probs_Bi[(probs_Bi > 0) & (probs_Bi < 1)]

        if len(probs_Bi) == 0:
            float_entropy = 0
        else:
            # H[p] = -sum(p * log2(p))
            float_entropy = -torch.sum(probs_Bi * torch.log2(probs_Bi)).item()

        float_entropy_F[feature_idx] = float_entropy

    total_entropy_F = bool_entropy_F.cuda() + bool_prob_F.cuda() * float_entropy_F.cuda()

    description_length = total_entropy_F.sum().item()

    return description_length


def quantize_features_to_bin_midpoints(
    features_BF: torch.Tensor, bins_F_list_Bi: list[torch.Tensor]
) -> torch.Tensor:
    """
    Quantize features to the bin midpoints of their corresponding histograms.
    """
    _, num_features = features_BF.shape

    quantized_features_BF = torch.empty_like(features_BF, device=features_BF.device)

    for feature_idx in range(num_features):
        # Extract the feature values and bin edges for the current histogram
        features_B = features_BF[:, feature_idx]
        bin_edges_Bi = bins_F_list_Bi[feature_idx]

        num_bins = len(bin_edges_Bi) - 1

        bin_indices_B = torch.bucketize(features_B, bin_edges_Bi)
        bin_indices_clipped_B = torch.clamp(bin_indices_B, min=1, max=num_bins) - 1

        # Calculate the midpoints of the bins
        bin_mids_Bi = 0.5 * (bin_edges_Bi[:-1] + bin_edges_Bi[1:])

        quantized_features_BF[:, feature_idx] = bin_mids_Bi[bin_indices_clipped_B]

    return quantized_features_BF


# def calculate_dl(
#     activations_store: ActivationsStore,
#     sae: SAE,
#     bins: list[torch.Tensor],
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
    bins_F_list_Bi: list[torch.Tensor],
    activations_store: ActivationsStore,
    sae: SAE,
    mse_threshold: float,
    autoencoder: SAE,
    k: Optional[int] = None,
) -> tuple[bool, float]:
    mse_losses: list[torch.Tensor] = []

    for i in range(1):
        x_BSN = activations_store.get_buffer(config.sae_batch_size)
        feature_activations_BSF = sae.encode(x_BSN).squeeze()

        if k is not None:
            topk_fn = TopK(k)
            feature_activations_BSF = topk_fn(feature_activations_BSF)

        quantised_feature_activations_BsF = quantize_features_to_bin_midpoints(
            feature_activations_BSF, bins_F_list_Bi
        )

        reconstructed_x_BSN: torch.Tensor = autoencoder.decode(quantised_feature_activations_BsF)

        mse_loss: torch.Tensor = F.mse_loss(reconstructed_x_BSN, x_BSN.squeeze(), reduction="mean")
        mse_loss = torch.sqrt(mse_loss) / sae.cfg.d_in
        mse_losses.append(mse_loss)

    avg_mse_loss = torch.mean(torch.stack(mse_losses))
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
    bins: list[torch.Tensor]
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
    bins: list[list[torch.Tensor]]
    k: list[Optional[int]]

    description_length: list[float]
    within_threshold: list[bool]
    mse_loss: list[float]

    def pick_minimum_viable(self) -> MDLEvalResult:
        all_description_lengths = torch.tensor(self.description_length)
        threshold_mask = torch.tensor(self.within_threshold)

        viable_description_lengths = all_description_lengths[threshold_mask]
        if len(viable_description_lengths) > 0:
            min_dl_idx = int(torch.argmin(viable_description_lengths).item())
            return self[min_dl_idx]

        else:
            min_dl_idx = int(torch.argmin(all_description_lengths).item())
            return self[min_dl_idx]


def run_eval_single_sae(
    config: MDLEvalConfig,
    sae: SAE,
    model: HookedTransformer,
    device: str,
    dataset_name: str = "HuggingFaceFW/fineweb",
) -> MDLEvalResultsCollection:
    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    torch.set_grad_enabled(False)
    mdl_eval_results_list: list[MDLEvalResult] = []

    sae.cfg.dataset_trust_remote_code = True
    sae = sae.to(device)
    model = model.to(device)  # type: ignore

    activations_store = ActivationsStore.from_sae(
        model, sae, config.sae_batch_size, dataset=dataset_name, device=device
    )

    num_features = sae.cfg.d_sae

    def get_min_max_activations() -> tuple[torch.Tensor, torch.Tensor]:
        min_pos_activations_1F = torch.zeros(1, num_features, device=device)
        max_activations_1F = torch.zeros(1, num_features, device=device) + 100

        for _ in range(10):
            neuron_activations_BSN = activations_store.get_buffer(config.sae_batch_size)

            feature_activations_BsF = sae.encode(neuron_activations_BSN).squeeze()

            cat_feature_activations_BsF = torch.cat(
                [
                    feature_activations_BsF,
                    min_pos_activations_1F,
                    max_activations_1F,
                ],
                dim=0,
            )
            min_pos_activations_1F = torch.min(cat_feature_activations_BsF, dim=0).values.unsqueeze(
                0
            )
            max_activations_1F = torch.max(cat_feature_activations_BsF, dim=0).values.unsqueeze(0)

        min_pos_activations_F = min_pos_activations_1F.squeeze()
        max_activations_F = max_activations_1F.squeeze()

        return min_pos_activations_F, max_activations_F

    min_pos_activations_F, max_activations_F = get_min_max_activations()

    print("num_bins_values", config.num_bins_values)
    print("k_values", config.k_values)

    for num_bins in config.num_bins_values:
        for k in config.k_values:
            bins = build_bins(min_pos_activations_F, max_activations_F, num_bins=num_bins)

            print("Built bins")

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

            print("Checked threshold")

            description_length = calculate_dl(
                num_features=num_features,
                bins_F_list_Bi=bins,
                device=device,
                activations_store=activations_store,
                sae=sae,
                k=k,
            )

            logger.info(
                f"Description length: {description_length} for num_bins = {num_bins} and k = {k} and mse = {mse_loss}"
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

    result = []

    for mdl_eval_result in mdl_eval_results:
        result.append(mdl_eval_result.to_dict())

    return result

    # minimum_viable_eval_result = mdl_eval_results.pick_minimum_viable()

    # minimum_viable_description_length = minimum_viable_eval_result.description_length
    # logger.info(minimum_viable_description_length)

    # return minimum_viable_eval_result


def run_eval(
    config: MDLEvalConfig,
    selected_saes: list[tuple[str, SAE]] | list[tuple[str, str]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
) -> dict[str, Any]:
    """
    selected_saes is a list of either tuples of (sae_lens release, sae_lens id) or (sae_name, SAE object)
    """
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    results_dict = {}

    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)

    print(f"Using dtype: {llm_dtype}")

    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

    for sae_release, sae_id in tqdm(
        selected_saes, desc="Running SAE evaluation on all selected SAEs"
    ):
        del_sae = False
        # Handle both pretrained SAEs (identified by string) and custom SAEs (passed as objects)
        if isinstance(sae_id, str):
            sae = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=device,
            )[0]
            # If loading from pretrained, we delete the SAE object after use
            del_sae = True
        else:
            sae = sae_id
            sae_id = "custom_sae"

        sae_result_file = f"{sae_release}_{sae_id}_eval_results.json"
        sae_result_file = sae_result_file.replace("/", "_")
        sae_result_path = os.path.join(output_path, sae_result_file)

        eval_output = run_eval_single_sae(
            config=config,
            sae=sae,
            model=model,
            dataset_name=config.dataset_name,
            device=device,
        )

        sae_eval_result = {
            "eval_instance_id": eval_instance_id,
            "sae_lens_release": sae_release,
            "sae_lens_id": sae_id,
            "eval_type_id": EVAL_TYPE,
            "sae_lens_version": sae_lens_version,
            "sae_bench_version": sae_bench_commit_hash,
            "date_time": datetime.now().isoformat(),
            "eval_config": asdict(config),
            "eval_results": eval_output,
            "eval_artifacts": {"artifacts": "None"},
            "sae_cfg_dict": asdict(sae.cfg),
        }

        with open(sae_result_path, "w") as f:
            json.dump(sae_eval_result, f, indent=4)

        results_dict[sae_result_file] = eval_output

    results_dict["custom_eval_config"] = asdict(config)

    if del_sae:
        del sae
    gc.collect()
    torch.cuda.empty_cache()

    return results_dict


def create_config_and_selected_saes(
    args,
) -> tuple[MDLEvalConfig, list[tuple[str, str]]]:
    config = MDLEvalConfig(
        model_name=args.model_name,
    )

    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    else:
        config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]

    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    else:
        config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

    if args.random_seed is not None:
        config.random_seed = args.random_seed

    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    assert len(selected_saes) > 0, "No SAEs selected"

    releases = set([release for release, _ in selected_saes])

    print(f"Selected SAEs from releases: {releases}")

    for release, sae in selected_saes:
        print(f"Sample SAEs: {release}, {sae}")

    return config, selected_saes


def arg_parser():
    parser = argparse.ArgumentParser(description="Run MDL evaluation")
    parser.add_argument("--random_seed", type=int, default=None, help="Random seed")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
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
        default="eval_results/mdl",
        help="Output folder",
    )
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun of experiments")
    parser.add_argument(
        "--clean_up_activations",
        action="store_false",
        help="Clean up activations after evaluation",
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=None,
        help="Batch size for LLM. If None, will be populated using LLM_NAME_TO_BATCH_SIZE",
    )
    parser.add_argument(
        "--llm_dtype",
        type=str,
        default=None,
        choices=[None, "float32", "float64", "float16", "bfloat16"],
        help="Data type for LLM. If None, will be populated using LLM_NAME_TO_DTYPE",
    )

    return parser


if __name__ == "__main__":
    """python evals/mdl/main.py \
    --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
    --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
    --model_name pythia-70m-deduped """
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    args = arg_parser().parse_args()
    device = general_utils.setup_environment()

    start_time = time.time()

    config, selected_saes = create_config_and_selected_saes(args)

    print(selected_saes)

    # create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    config = MDLEvalConfig(
        k_values=[None],
        # num_bins_values=[8, 12, 16, 32, 64, 128],
        num_bins_values=[8, 16, 32, 64],
        # num_bins_values=[8],
        mse_epsilon_threshold=0.2,
        model_name=args.model_name,
    )
    logger.info(config)

    results_dict = run_eval(
        config,
        selected_saes,
        device,
        args.output_folder,
        args.force_rerun,
    )

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time} seconds")

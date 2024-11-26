import os
import gc
import time
import random
import argparse
from tqdm import tqdm
from typing import Literal

import torch
from sae_lens import SAE
from nnsight import LanguageModel
import matplotlib.pyplot as plt

import sae_bench_utils.activation_collection as activation_collection
from sae_bench_utils.sae_selection_utils import get_saes_from_regex
from sae_bench_utils import general_utils as general_utils
from sae_bench_utils import (
    get_eval_uuid,
    get_sae_lens_version,
    get_sae_bench_version,
)

from evals.ravel.instance import create_filtered_dataset
from evals.ravel.eval_config import RAVELEvalConfig
from evals.ravel.uniprobe import run_feature_selection_probe
from evals.ravel.intervention import compute_disentanglement
from evals.ravel.eval_output import (
    RAVELEvalOutput,
    RAVELMetricResults,
    RAVELMetricCategories,
)



eval_config = RAVELEvalConfig()
rng = random.Random(eval_config.random_seed)


def run_eval_single_dataset(
    config: RAVELEvalConfig,
    model: LanguageModel,
    model_id: str,
    sae: SAE,
    entity_class: str,
):


    # TODO restrict to chosen attributes
    # Create filtered dataset IFF it does not exist
    dataset = create_filtered_dataset(
        model_id=model_id,
        chosen_entity=entity_class,
        model=model,
        force_recompute=config.force_dataset_recompute,
        n_samples_per_attribute_class=config.n_samples_per_attribute_class,
        top_n_entities=config.top_n_entities,
        top_n_templates=config.top_n_templates,
    )

    # TODO save results
    all_attributes = dataset.get_attributes()
    print(f"All attributes: {all_attributes}")

    attribute_feature_dict = run_feature_selection_probe(
        model,
        sae,
        dataset,
        all_attributes=all_attributes,
        coeffs=config.probe_coefficients,
        max_samples_per_attribute=config.max_samples_per_attribute,
        layer=sae.cfg.hook_layer,
        llm_batch_size=config.llm_batch_size,
    )


    results_detail = compute_disentanglement(
        model,
        sae,
        sae.cfg.hook_layer,
        dataset,
        entity_class,
        all_attributes,
        attribute_feature_dict,
        n_interventions=config.n_interventions,
        n_generated_tokens=config.n_generated_tokens,
        inv_batch_size=config.inv_batch_size,
    )

    return results_detail


def run_eval_single_sae(
    config: RAVELEvalConfig,
    model: LanguageModel,
    model_id: str,
    sae: SAE,
):
    results_detail_per_entity = []
    for entity_class in config.entity_attribute_selection:
        results_detail = run_eval_single_dataset(
            config=config,
            model=model,
            model_id=model_id,
            sae=sae,
            entity_class=entity_class,
        )
        results_detail_per_entity.append(results_detail)
    return results_detail_per_entity



# TODO test for MODEL_ID = 'pythia-70m'
def run_eval(
    eval_config: RAVELEvalConfig,
    selected_saes: list[tuple[str, str]],
    output_path: str,
    device: Literal["cpu", "cuda", 'mps'],
):
    # Instanciate evaluation run
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    # Load model
    if eval_config.model_name == "pythia-70m":
        model_id = "eleutherAI/pythia-70m-deduped"
        model_kwargs = {}
    elif eval_config.model_name == "gemma-2-2b":
        model_id = "google/gemma-2-2b"
        model_kwargs = {'low_cpu_mem_usage': True,'attn_implementation': 'eager'}
    else:
        raise ValueError(f"Invalid model name: {eval_config.model_name}")

    if eval_config.llm_dtype == "bfloat16":
        llm_dtype = torch.bfloat16
    elif eval_config.llm_dtype == "float32":
        llm_dtype = torch.float32
    else:
        raise ValueError(f"Invalid dtype: {eval_config.llm_dtype}")

    model = LanguageModel(
        model_id,
        cache_dir=eval_config.model_dir,
        device_map=device,
        torch_dtype=llm_dtype,
        dispatch=True,
        **model_kwargs,
    )
    model.requires_grad_(False)
    model.eval()

    # Initialize directories and output file
    artifacts_base_folder = 'artifacts/ravel'
    os.makedirs(artifacts_base_folder, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)
    results_dict = {}

    # Iteratively evaluate each SAE
    for sae_release, sae_id in tqdm(
        selected_saes, desc="Running SAE evaluation on all selected SAEs"
    ):
        gc.collect()
        torch.cuda.empty_cache()

        # Handle both pretrained SAEs (identified by string) and custom SAEs (passed as objects)
        if isinstance(sae_id, str):
            sae = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=device,
            )[0]
        else:
            sae = sae_id
            sae_id = "custom_sae"

        sae = sae.to(device=device, dtype=llm_dtype)

        # Initialize SAE specific directories
        artifacts_folder = os.path.join(
            artifacts_base_folder, eval_config.model_name, sae.cfg.hook_name
        )
        sae_result_file = f"{sae_release}_{sae_id}_eval_results.json"
        sae_result_file = sae_result_file.replace("/", "_")
        sae_result_path = os.path.join(output_path, sae_result_file)

        # Run evaluation
        result_details = run_eval_single_sae(
            config=eval_config,
            model=model,
            model_id=model_id,
            sae=sae,
        )

        # Aggregate results
        # Find cause, isolation, and disentanglement scores for max disentanglement
        max_disentanglement_scores, max_cause_scores, max_isolation_scores = [], [], []
        for result_detail in result_details:
            max_disentanglement = 0
            best_threshold = 0
            for t, score in result_detail["mean_disentanglement"].items():
                if score > max_disentanglement:
                    max_disentanglement = score
                    best_threshold = t
            max_disentanglement_scores.append(max_disentanglement)
            max_cause_scores.append(result_detail["cause_scores"][best_threshold])
            max_isolation_scores.append(result_detail["isolation_scores"][best_threshold])

        mean_max_disentanglement = sum(max_disentanglement_scores) / len(max_disentanglement_scores)
        mean_max_cause = sum(max_cause_scores) / len(max_cause_scores)
        mean_max_isolation = sum(max_isolation_scores) / len(max_isolation_scores)
            
        aggregated_results = RAVELMetricResults(
            disentanglement_score=mean_max_disentanglement,
            cause_score=mean_max_cause,
            isolation_score=mean_max_isolation,
        )


        # Save results
        results = RAVELEvalOutput(
            eval_config=eval_config,
            eval_id=eval_instance_id,
            datetime_epoch_millis=int(time.time() * 1000),
            eval_result_metrics=RAVELMetricCategories(sae=aggregated_results),
            eval_result_details=result_details,
            sae_lens_id=sae_id,
            sae_lens_release_id=sae_release,
            sae_lens_version=sae_lens_version,
            sae_bench_commit_hash=sae_bench_commit_hash,
        )




def create_config_and_selected_saes(
    args,
) -> tuple[RAVELEvalConfig, list[tuple[str, str]]]:
    config = RAVELEvalConfig(
        random_seed=args.random_seed,
        model_name=args.model_name,
    )

    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    assert len(selected_saes) > 0, "No SAEs selected"

    releases = set([release for release, _ in selected_saes])

    print(f"Selected SAEs from releases: {releases}")

    for release, sae in selected_saes:
        print(f"Sample SAEs: {release}, {sae}")

    return config, selected_saes


def arg_parser():
    parser = argparse.ArgumentParser(description="Run SCR or TPP evaluation")
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
        default="eval_results/ravel",
    )
    return parser


if __name__ == "__main__":
    """
    Example Gemma-2-2B SAE Bench usage:
    python evals/ravel/main.py \
    --sae_regex_pattern "sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824" \
    --sae_block_pattern "blocks.19.hook_resid_post__trainer_2" \
    --model_name "gemma-2-2b" \
    """
    args = arg_parser().parse_args()
    device = general_utils.setup_environment()

    start_time = time.time()

    config, selected_saes = create_config_and_selected_saes(args)

    print(selected_saes)

    config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
    config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

    # create output folder
    # os.makedirs(args.output_folder, exist_ok=True)

    # run the evaluation on all selected SAEs
    results_dict = run_eval(
        config,
        selected_saes,
        args.output_folder,
        device,
    )

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time} seconds")
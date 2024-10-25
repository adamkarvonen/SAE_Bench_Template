import gc
import json
import os
import shutil
import random
import time
from dataclasses import asdict
from typing import Optional

import pandas as pd
import torch
from sae_lens import SAE
from sae_lens.sae import TopK
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from tqdm import tqdm
from transformer_lens import HookedTransformer
import argparse
from datetime import datetime


import evals.sparse_probing.eval_config as eval_config
import evals.sparse_probing.probe_training as probe_training
import sae_bench_utils.activation_collection as activation_collection
import sae_bench_utils.dataset_info as dataset_info
import sae_bench_utils.dataset_utils as dataset_utils
import sae_bench_utils.formatting_utils as formatting_utils

from sae_bench_utils import (
    formatting_utils,
    activation_collection,
    get_eval_uuid,
    get_sae_lens_version,
    get_sae_bench_version,
)
from sae_bench_utils.sae_selection_utils import get_saes_from_regex, select_saes_multiple_patterns


def average_test_accuracy(test_accuracies: dict[str, float]) -> float:
    return sum(test_accuracies.values()) / len(test_accuracies)


def get_dataset_activations(
    dataset_name: str,
    config: eval_config.EvalConfig,
    model: HookedTransformer,
    llm_batch_size: int,
    layer: int,
    hook_point: str,
    device: str,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    train_data, test_data = dataset_utils.get_multi_label_train_test_data(
        dataset_name,
        config.probe_train_set_size,
        config.probe_test_set_size,
        config.random_seed,
    )

    chosen_classes = dataset_info.chosen_classes_per_dataset[dataset_name]

    train_data = dataset_utils.filter_dataset(train_data, chosen_classes)
    test_data = dataset_utils.filter_dataset(test_data, chosen_classes)

    train_data = dataset_utils.tokenize_data(
        train_data, model.tokenizer, config.context_length, device
    )
    test_data = dataset_utils.tokenize_data(
        test_data, model.tokenizer, config.context_length, device
    )

    all_train_acts_BLD = activation_collection.get_all_llm_activations(
        train_data, model, llm_batch_size, layer, hook_point
    )
    all_test_acts_BLD = activation_collection.get_all_llm_activations(
        test_data, model, llm_batch_size, layer, hook_point
    )

    return all_train_acts_BLD, all_test_acts_BLD


def run_eval_single_dataset(
    dataset_name: str,
    config: eval_config.EvalConfig,
    sae: SAE,
    model: HookedTransformer,
    layer: int,
    hook_point: str,
    device: str,
    artifacts_folder: str,
    force_rerun: bool,
    save_activations: bool,
) -> dict[str, float]:
    """config: eval_config.EvalConfig contains all hyperparameters to reproduce the evaluation.
    It is saved in the results_dict for reproducibility.
    selected_saes_dict: dict[str, list[str]] is a dict of SAE release name: list of SAE names to evaluate.
    Example: sae_bench_pythia70m_sweep_topk_ctx128_0730 :
    ['pythia70m_sweep_topk_ctx128_0730/resid_post_layer_4/trainer_10',
    'pythia70m_sweep_topk_ctx128_0730/resid_post_layer_4/trainer_12']"""

    results_dict = {}

    activations_filename = f"{dataset_name}_activations.pt".replace("/", "_")

    activations_path = os.path.join(artifacts_folder, activations_filename)

    if not os.path.exists(activations_path) or force_rerun:
        all_train_acts_BLD, all_test_acts_BLD = get_dataset_activations(
            dataset_name, config, model, config.llm_batch_size, layer, hook_point, device
        )

        all_train_acts_BD = activation_collection.create_meaned_model_activations(
            all_train_acts_BLD
        )

        all_test_acts_BD = activation_collection.create_meaned_model_activations(all_test_acts_BLD)

        llm_probes, llm_test_accuracies = probe_training.train_probe_on_activations(
            all_train_acts_BD,
            all_test_acts_BD,
            select_top_k=None,
        )

        llm_results = {"llm_test_accuracy": average_test_accuracy(llm_test_accuracies)}

        llm_test_accuracy = average_test_accuracy(llm_test_accuracies)

        for k in config.k_values:
            llm_top_k_probes, llm_top_k_test_accuracies = probe_training.train_probe_on_activations(
                all_train_acts_BD,
                all_test_acts_BD,
                select_top_k=k,
            )
            llm_results[f"llm_top_{k}_test_accuracy"] = average_test_accuracy(
                llm_top_k_test_accuracies
            )

        acts = {
            "train": all_train_acts_BLD,
            "test": all_test_acts_BLD,
            "llm_results": llm_results,
        }

        if save_activations:
            torch.save(acts, activations_path)
    else:
        print(f"Loading activations from {activations_path}")
        acts = torch.load(activations_path)
        all_train_acts_BLD = acts["train"]
        all_test_acts_BLD = acts["test"]
        llm_results = acts["llm_results"]

    all_train_acts_BD = activation_collection.create_meaned_model_activations(all_train_acts_BLD)
    all_test_acts_BD = activation_collection.create_meaned_model_activations(all_test_acts_BLD)

    all_sae_train_acts_BF = activation_collection.get_sae_meaned_activations(
        all_train_acts_BLD, sae, config.sae_batch_size
    )
    all_sae_test_acts_BF = activation_collection.get_sae_meaned_activations(
        all_test_acts_BLD, sae, config.sae_batch_size
    )

    # This is optional, checking the accuracy of a probe trained on the entire SAE activations
    # We use GPU here as sklearn.fit is slow on large input dimensions, all other probe training is done with sklearn.fit
    _, sae_test_accuracies = probe_training.train_probe_on_activations(
        all_sae_train_acts_BF,
        all_sae_test_acts_BF,
        select_top_k=None,
        use_sklearn=False,
        batch_size=250,
        epochs=100,
        lr=1e-2,
    )

    for llm_result_key, llm_result_value in llm_results.items():
        results_dict[llm_result_key] = llm_result_value

    results_dict["sae_test_accuracy"] = average_test_accuracy(sae_test_accuracies)

    for k in config.k_values:
        sae_top_k_probes, sae_top_k_test_accuracies = probe_training.train_probe_on_activations(
            all_sae_train_acts_BF,
            all_sae_test_acts_BF,
            select_top_k=k,
        )
        results_dict[f"sae_top_{k}_test_accuracy"] = average_test_accuracy(
            sae_top_k_test_accuracies
        )

    return results_dict


def run_eval_single_sae(
    config: eval_config.EvalConfig,
    sae: SAE,
    model: HookedTransformer,
    layer: int,
    hook_point: str,
    device: str,
    artifacts_folder: str,
    force_rerun: bool,
    save_activations: bool = True,
) -> dict[str, float | dict[str, float]]:
    """hook_point: str is transformer lens format. example: f'blocks.{layer}.hook_resid_post'
    By default, we save activations for all datasets, and then reuse them for each sae.
    This is important to avoid recomputing activations for each SAE, and to ensure that the same activations are used for all SAEs.
    However, it can use 10s of GBs of disk space."""

    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    results_dict = {}

    dataset_results = {}
    for dataset_name in config.dataset_names:
        dataset_results[f"{dataset_name}_results"] = run_eval_single_dataset(
            dataset_name,
            config,
            sae,
            model,
            layer,
            hook_point,
            device,
            artifacts_folder,
            force_rerun,
            save_activations,
        )

    results_dict = formatting_utils.average_results_dictionaries(
        dataset_results, config.dataset_names
    )

    for dataset_name, dataset_result in dataset_results.items():
        results_dict[f"{dataset_name}_results"] = dataset_result

    return results_dict


def run_eval(
    config: eval_config.EvalConfig,
    selected_saes_dict: dict[str, list[str]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
    clean_up_activations: bool = True,
) -> dict[str, dict[str, float | dict[str, float]]]:
    """By default, clean_up_activations is True, which means that the activations are deleted after the evaluation is done.
    This is because activations for all datasets can easily be 10s of GBs.
    Return dict is a dict of SAE name: evaluation results for that SAE."""
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_version = get_sae_bench_version()

    artifacts_base_folder = "artifacts"

    results_dict = {}

    if config.llm_dtype == "bfloat16":
        llm_dtype = torch.bfloat16
    elif config.llm_dtype == "float32":
        llm_dtype = torch.float32
    else:
        raise ValueError(f"Invalid dtype: {config.llm_dtype}")

    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

    for sae_release in selected_saes_dict:
        print(
            f"Running evaluation for SAE release: {sae_release}, SAEs: {selected_saes_dict[sae_release]}"
        )

        for sae_id in tqdm(
            selected_saes_dict[sae_release],
            desc="Running SAE evaluation on all selected SAEs",
        ):
            gc.collect()
            torch.cuda.empty_cache()

            sae = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=device,
            )[0]
            sae = sae.to(device=device, dtype=llm_dtype)

            artifacts_folder = os.path.join(
                artifacts_base_folder, "sparse_probing", config.model_name, sae.cfg.hook_name
            )
            os.makedirs(artifacts_folder, exist_ok=True)

            sparse_probing_results = run_eval_single_sae(
                config,
                sae,
                model,
                sae.cfg.hook_layer,
                sae.cfg.hook_name,
                device,
                artifacts_folder,
                force_rerun,
            )

            sae_eval_result = {
                "eval_instance_id": eval_instance_id,
                "sae_lens_release": sae_release,
                "sae_lens_id": sae_id,
                "eval_type_id": "sparse_probing",
                "sae_lens_version": sae_lens_version,
                "sae_bench_version": sae_bench_version,
                "date_time": datetime.now().isoformat(),
                "eval_config": asdict(config),
                "eval_results": sparse_probing_results,
                "eval_artifacts": {"artifacts": os.path.relpath(artifacts_folder)},
            }

            results_dict[sae_id] = sae_eval_result

            # Save individual SAE result
            sae_result_file = f"{sae_release}_{sae_id}_eval_results.json"
            sae_result_file = sae_result_file.replace("/", "_")
            sae_result_path = os.path.join(output_path, sae_result_file)

            with open(sae_result_path, "w") as f:
                json.dump(sae_eval_result, f, indent=4)

    if clean_up_activations:
        shutil.rmtree(artifacts_folder)

    return results_dict


def setup_environment():
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    return device


def create_config_and_selected_saes(
    args,
) -> tuple[eval_config.EvalConfig, dict[str, list[str]]]:
    config = eval_config.EvalConfig(
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
        "--sae_regex_pattern", type=str, required=True, help="Regex pattern for SAE selection"
    )
    parser.add_argument(
        "--sae_block_pattern", type=str, required=True, help="Regex pattern for SAE block selection"
    )
    parser.add_argument(
        "--output_folder", type=str, default="evals/sparse_probing/results", help="Output folder"
    )
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun of experiments")
    parser.add_argument(
        "--clean_up_activations", action="store_false", help="Clean up activations after evaluation"
    )

    return parser


if __name__ == "__main__":
    """
    python evals/sparse_probing/main.py \
    --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
    --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
    --model_name pythia-70m-deduped 
    
    
    """
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

    # sae_regex_patterns = None
    # sae_block_pattern = None

    config, selected_saes_dict = create_config_and_selected_saes(args)

    if sae_regex_patterns is not None:
        selected_saes_dict = select_saes_multiple_patterns(sae_regex_patterns, sae_block_pattern)

    print(selected_saes_dict)

    config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
    config.llm_dtype = str(activation_collection.LLM_NAME_TO_DTYPE[config.model_name]).split(".")[
        -1
    ]

    # create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # run the evaluation on all selected SAEs
    results_dict = run_eval(
        config,
        selected_saes_dict,
        device,
        args.output_folder,
        args.force_rerun,
        args.clean_up_activations,
    )

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time} seconds")

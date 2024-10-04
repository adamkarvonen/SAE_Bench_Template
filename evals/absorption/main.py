from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import random
import time
from sae_lens import SAE
import torch
from tqdm import tqdm
import pandas as pd
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens.sae import TopK

from evals.absorption import common, eval_config
from evals.absorption.k_sparse_probing import run_k_sparse_probing_experiment
from utils import formatting_utils, activation_collection
from transformer_lens import HookedTransformer


def average_test_accuracy(test_accuracies: dict[str, float]) -> float:
    return sum(test_accuracies.values()) / len(test_accuracies)


def run_eval(
    config: eval_config.EvalConfig,
    selected_saes_dict: dict[str, list[str]],
    device: str,
    force_rerun: bool = False,
):
    """config: eval_config.EvalConfig contains all hyperparameters to reproduce the evaluation.
    It is saved in the results_dict for reproducibility.
    selected_saes_dict: dict[str, list[str]] is a dict of SAE release name: list of SAE names to evaluate.
    Example: sae_bench_pythia70m_sweep_topk_ctx128_0730 :
    ['pythia70m_sweep_topk_ctx128_0730/resid_post_layer_4/trainer_10',
    'pythia70m_sweep_topk_ctx128_0730/resid_post_layer_4/trainer_12']"""
    # TODO: Make this nicer.
    sae_map_df = pd.DataFrame.from_records(
        {k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}
    ).T

    results_dict = {}
    results_dict["custom_eval_results"] = {}

    llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
    llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

    print(f"Running evaluation for layer {config.layer}")

    for sae_release in selected_saes_dict:
        print(
            f"Running evaluation for SAE release: {sae_release}, SAEs: {selected_saes_dict[sae_release]}"
        )
        sae_id_to_name_map = sae_map_df.saes_map[sae_release]
        sae_name_to_id_map = {v: k for k, v in sae_id_to_name_map.items()}

        for sae_name in tqdm(
            selected_saes_dict[sae_release],
            desc="Running SAE evaluation on all selected SAEs",
        ):
            gc.collect()
            torch.cuda.empty_cache()

            sae_id = sae_name_to_id_map[sae_name]

            sae = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=device,
            )[0]
            sae = sae.to(device=device)
            sae = _fix_topk(sae, sae_name, sae_release)

            run_k_sparse_probing_experiment(
                model=model,
                sae=sae,
                layer=config.layer,
                sae_name=sae_name,
                force=force_rerun,
                max_k_value=config.max_k_value,
                f1_jump_threshold=config.f1_jump_threshold,
            )

            results_dict["custom_eval_results"][sae_name] = {}

    results_dict["custom_eval_config"] = asdict(config)
    return results_dict


def _fix_topk(
    sae: SAE,
    sae_name: str,
    sae_release: str,
):
    if "topk" in sae_name:
        if isinstance(sae.activation_fn, TopK):
            return sae

        sae = formatting_utils.fix_topk_saes(sae, sae_release, sae_name, data_dir="../")

        assert isinstance(sae.activation_fn, TopK)
    return sae


if __name__ == "__main__":
    start_time = time.time()

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    config = eval_config.EvalConfig()

    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

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

    # run the evaluation on all selected SAEs
    results_dict = run_eval(config, config.selected_saes_dict, device)

    # create output filename and save results
    checkpoints_str = ""
    if config.include_checkpoints:
        checkpoints_str = "_with_checkpoints"

    output_filename = (
        config.model_name + f"_layer_{config.layer}{checkpoints_str}_eval_results.json"
    )
    output_folder = "sparse_probing_results"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    output_location = os.path.join(output_folder, output_filename)

    with open(output_location, "w") as f:
        json.dump(results_dict, f)

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time} seconds")

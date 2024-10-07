import os
import time
import torch
import pandas as pd
import random
import gc
import json
from tqdm import tqdm
from dataclasses import asdict
from transformer_lens import HookedTransformer
from sae_lens import SAE
from sae_lens.sae import TopK
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from evals.unlearning.utils.feature_activation import (
    check_existing_results,
    ensure_sae_weights,
    get_shuffled_forget_retain_tokens,
    calculate_sparsity,
    save_results
)

import evals.unlearning.eval_config as eval_config
import utils.eval as run_eval_single_sae
import utils.activation_collection as activation_collection
import utils.formatting_utils as formatting_utils

def save_feature_sparsity(model, sae):

    if check_existing_results(args.sae_folder):
        print(f"Sparsity calculation for {args.sae_folder} is already done")
        return

    ensure_sae_weights(args.sae_folder)

    forget_tokens, retain_tokens = get_shuffled_forget_retain_tokens(model, batch_size=2048, seq_len=1024)
    
    feature_sparsity_forget, feature_sparsity_retain = calculate_sparsity(model, sae, forget_tokens, retain_tokens)
    
    save_results(args.sae_folder, feature_sparsity_forget, feature_sparsity_retain)


def run_eval(
    config: eval_config.EvalConfig,
    selected_saes_dict: dict[str, list[str]],
    device: str,
):
    results_dict = {}

    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]
    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

    sae_map_df = pd.DataFrame.from_records(
        {k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}
    ).T
    
    # TODO: add eval results
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

            sae, cfg_dict, sparsity = SAE.from_pretrained(
                release=sae_release,
                sae_id=sae_id,
                device=device,
            )
            sae = sae.to(device=device)

            if "topk" in sae_name:
                assert isinstance(sae.activation_fn, TopK)
                
            save_feature_sparsity(model, sae)
            single_sae_eval_results = run_eval_single_sae(model, sae)
            results_dict[sae_name] = single_sae_eval_results
                
    
    results_dict["custom_eval_config"] = asdict(config)
    results_dict["custom_eval_results"] = formatting_utils.average_results_dictionaries(
        results_dict, config.dataset_names
    )

    return results_dict


if __name__ == "__main__":
    start_time = time.time()

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    config = eval_config.EvalConfig()

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
    output_folder = "results"  # at evals/<eval_name>

    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    output_location = os.path.join(output_folder, output_filename)

    with open(output_location, "w") as f:
        json.dump(results_dict, f)

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time} seconds")

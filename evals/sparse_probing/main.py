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

import evals.sparse_probing.eval_config as eval_config
import evals.sparse_probing.probe_training as probe_training
import utils.dataset_utils as dataset_utils
import utils.activation_collection as activation_collection
import utils.formatting_utils as formatting_utils
import utils.dataset_info as dataset_info


def average_test_accuracy(test_accuracies: dict[str, float]) -> float:
    return sum(test_accuracies.values()) / len(test_accuracies)


def run_eval_single_dataset(
    config: eval_config.EvalConfig,
    selected_saes_dict: dict[str, list[str]],
    dataset_name: str,
    model: HookedTransformer,
    device: str,
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

    llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
    llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

    train_df, test_df = dataset_utils.load_huggingface_dataset(dataset_name)
    train_data, test_data = dataset_utils.get_multi_label_train_test_data(
        train_df,
        test_df,
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

    print(f"Running evaluation for layer {config.layer}")
    hook_name = f"blocks.{config.layer}.hook_resid_post"

    all_train_acts_BLD = activation_collection.get_all_llm_activations(
        train_data, model, llm_batch_size, hook_name
    )
    all_test_acts_BLD = activation_collection.get_all_llm_activations(
        test_data, model, llm_batch_size, hook_name
    )

    all_train_acts_BD = activation_collection.create_meaned_model_activations(all_train_acts_BLD)
    all_test_acts_BD = activation_collection.create_meaned_model_activations(all_test_acts_BLD)

    llm_probes, llm_test_accuracies = probe_training.train_probe_on_activations(
        all_train_acts_BD,
        all_test_acts_BD,
        select_top_k=None,
    )

    llm_results = {"llm_test_accuracy": average_test_accuracy(llm_test_accuracies)}

    for k in config.k_values:
        llm_top_k_probes, llm_top_k_test_accuracies = probe_training.train_probe_on_activations(
            all_train_acts_BD,
            all_test_acts_BD,
            select_top_k=k,
        )
        llm_results[f"llm_top_{k}_test_accuracy"] = average_test_accuracy(llm_top_k_test_accuracies)

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

            all_sae_train_acts_BF = activation_collection.get_sae_meaned_activations(
                all_train_acts_BLD, sae, config.sae_batch_size, llm_dtype
            )
            all_sae_test_acts_BF = activation_collection.get_sae_meaned_activations(
                all_test_acts_BLD, sae, config.sae_batch_size, llm_dtype
            )

            _, sae_test_accuracies = probe_training.train_probe_on_activations(
                all_sae_train_acts_BF,
                all_sae_test_acts_BF,
                select_top_k=None,
                use_sklearn=False,
            )

            results_dict[sae_name] = {}

            for llm_result_key, llm_result_value in llm_results.items():
                results_dict[sae_name][llm_result_key] = llm_result_value

            results_dict[sae_name]["sae_test_accuracy"] = average_test_accuracy(sae_test_accuracies)

            for k in config.k_values:
                sae_top_k_probes, sae_top_k_test_accuracies = (
                    probe_training.train_probe_on_activations(
                        all_sae_train_acts_BF,
                        all_sae_test_acts_BF,
                        select_top_k=k,
                    )
                )
                results_dict[sae_name][f"sae_top_{k}_test_accuracy"] = average_test_accuracy(
                    sae_top_k_test_accuracies
                )

    return results_dict


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

    for dataset_name in config.dataset_names:
        results_dict[f"{dataset_name}_results"] = run_eval_single_dataset(
            config, selected_saes_dict, dataset_name, model, device
        )

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

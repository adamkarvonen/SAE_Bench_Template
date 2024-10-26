import gc
import json
import os
import shutil
import random
import time
from dataclasses import asdict
from typing import Optional

import einops
import pandas as pd
import torch
from sae_lens import SAE
from sae_lens.sae import TopK
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from tqdm import tqdm
from transformer_lens import HookedTransformer
import argparse
from datetime import datetime
import pickle

import evals.shift_and_tpp.dataset_creation as dataset_creation
import evals.shift_and_tpp.eval_config as eval_config
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

COLUMN2_VALS_LOOKUP = {
    "LabHC/bias_in_bios_class_set1": ("male", "female"),
    "canrager/amazon_reviews_mcauley_1and5": (1.0, 5.0),
}


@torch.no_grad()
def get_effects_per_class_precomputed_acts(
    sae: SAE,
    probe: probe_training.Probe,
    class_idx: str,
    precomputed_acts: dict[str, torch.Tensor],
    spurious_corr: bool,
    sae_batch_size: int,
) -> torch.Tensor:
    device = sae.device

    inputs_train_BLD, labels_train_B = probe_training.prepare_probe_data(
        precomputed_acts, class_idx, spurious_corr
    )

    all_acts_list_F = []

    assert inputs_train_BLD.shape[0] == len(labels_train_B)

    for i in range(0, inputs_train_BLD.shape[0], sae_batch_size):
        activation_batch_BLD = inputs_train_BLD[i : i + sae_batch_size]
        labels_batch_B = labels_train_B[i : i + sae_batch_size]
        dtype = activation_batch_BLD.dtype

        activations_BL = einops.reduce(activation_batch_BLD, "B L D -> B L", "sum")
        nonzero_acts_BL = (activations_BL != 0.0).to(dtype=dtype)
        nonzero_acts_B = einops.reduce(nonzero_acts_BL, "B L -> B", "sum")

        f_BLF = sae.encode(activation_batch_BLD)
        f_BLF = f_BLF * nonzero_acts_BL[:, :, None]  # zero out masked tokens

        # Get the average activation per input. We divide by the number of nonzero activations for the attention mask
        average_sae_acts_BF = einops.reduce(f_BLF, "B L F -> B F", "sum") / nonzero_acts_B[:, None]

        pos_sae_acts_BF = average_sae_acts_BF[labels_batch_B == dataset_info.POSITIVE_CLASS_LABEL]
        neg_sae_acts_BF = average_sae_acts_BF[labels_batch_B == dataset_info.NEGATIVE_CLASS_LABEL]

        average_pos_sae_acts_F = einops.reduce(pos_sae_acts_BF, "B F -> F", "mean")
        average_neg_sae_acts_F = einops.reduce(neg_sae_acts_BF, "B F -> F", "mean")

        sae_acts_diff_F = average_pos_sae_acts_F - average_neg_sae_acts_F

        all_acts_list_F.append(sae_acts_diff_F)

    all_acts_BF = torch.stack(all_acts_list_F, dim=0)
    average_acts_F = einops.reduce(all_acts_BF, "B F -> F", "mean").to(dtype=torch.float32)

    probe_weight_D = probe.net.weight.to(dtype=torch.float32, device=device)

    decoder_weight_DF = sae.W_dec.data.T.to(dtype=torch.float32, device=device)

    dot_prod_F = (probe_weight_D @ decoder_weight_DF).squeeze()

    if not spurious_corr:
        # Only consider activations from the positive class
        average_acts_F.clamp_(min=0.0)

    effects_F = average_acts_F * dot_prod_F

    if spurious_corr:
        effects_F = effects_F.abs()

    return effects_F


def get_all_node_effects_for_one_sae(
    sae: SAE,
    probes: dict[str, probe_training.Probe],
    chosen_class_indices: list[str],
    spurious_corr: bool,
    indirect_effect_acts: dict[str, torch.Tensor],
    sae_batch_size: int,
) -> dict[str, torch.Tensor]:
    node_effects = {}
    for ablated_class_idx in chosen_class_indices:
        node_effects[ablated_class_idx] = get_effects_per_class_precomputed_acts(
            sae,
            probes[ablated_class_idx],
            ablated_class_idx,
            indirect_effect_acts,
            spurious_corr,
            sae_batch_size,
        )

    return node_effects


def select_top_n_features(effects: torch.Tensor, n: int, class_name: str) -> torch.Tensor:
    assert (
        n <= effects.numel()
    ), f"n ({n}) must not be larger than the number of features ({effects.numel()}) for ablation class {class_name}"

    # Find non-zero effects
    non_zero_mask = effects != 0
    non_zero_effects = effects[non_zero_mask]
    num_non_zero = non_zero_effects.numel()

    if num_non_zero < n:
        print(
            f"WARNING: only {num_non_zero} non-zero effects found for ablation class {class_name}, which is less than the requested {n}."
        )

    # Select top n or all non-zero effects, whichever is smaller
    k = min(n, num_non_zero)

    if k == 0:
        print(
            f"WARNING: No non-zero effects found for ablation class {class_name}. Returning an empty mask."
        )
        top_n_features = torch.zeros_like(effects, dtype=torch.bool)
    else:
        # Get the indices of the top N effects
        _, top_indices = torch.topk(effects, k)

        # Create a boolean mask tensor
        mask = torch.zeros_like(effects, dtype=torch.bool)
        mask[top_indices] = True

        top_n_features = mask

    return top_n_features


def ablated_precomputed_activations(
    ablation_acts_BLD: torch.Tensor,
    sae: SAE,
    to_ablate: torch.Tensor,
    sae_batch_size: int,
) -> torch.Tensor:
    """NOTE: We don't pass in the attention mask. Thus, we must have already zeroed out all masked tokens in ablation_acts_BLD."""

    all_acts_list_BD = []

    for i in range(0, ablation_acts_BLD.shape[0], sae_batch_size):
        activation_batch_BLD = ablation_acts_BLD[i : i + sae_batch_size]
        dtype = activation_batch_BLD.dtype

        activations_BL = einops.reduce(activation_batch_BLD, "B L D -> B L", "sum")
        nonzero_acts_BL = (activations_BL != 0.0).to(dtype=dtype)
        nonzero_acts_B = einops.reduce(nonzero_acts_BL, "B L -> B", "sum")

        f_BLF = sae.encode(activation_batch_BLD)
        x_hat_BLD = sae.decode(f_BLF)

        error_BLD = activation_batch_BLD - x_hat_BLD

        f_BLF[..., to_ablate] = 0.0  # zero ablation

        modified_acts_BLD = sae.decode(f_BLF) + error_BLD

        # Get the average activation per input. We divide by the number of nonzero activations for the attention mask
        probe_acts_BD = (
            einops.reduce(modified_acts_BLD, "B L D -> B D", "sum") / nonzero_acts_B[:, None]
        )
        all_acts_list_BD.append(probe_acts_BD)

    all_acts_BD = torch.cat(all_acts_list_BD, dim=0)

    return all_acts_BD


def get_probe_test_accuracy(
    probes: dict[str, probe_training.Probe],
    all_class_list: list[str],
    all_activations: dict[str, torch.Tensor],
    probe_batch_size: int,
    spurious_corr: bool,
) -> dict[str, float]:
    test_accuracies = {}
    for class_name in all_class_list:
        test_acts, test_labels = probe_training.prepare_probe_data(
            all_activations, class_name, spurious_corr=spurious_corr
        )

        test_acc_probe = probe_training.test_probe_gpu(
            test_acts,
            test_labels,
            probe_batch_size,
            probes[class_name],
        )
        test_accuracies[class_name] = test_acc_probe

    if spurious_corr:
        shift_probe_accuracies = get_shift_probe_test_accuracy(
            probes, all_class_list, all_activations, probe_batch_size
        )
        test_accuracies.update(shift_probe_accuracies)

    return test_accuracies


def get_shift_probe_test_accuracy(
    probes: dict[str, probe_training.Probe],
    all_class_list: list[str],
    all_activations: dict[str, torch.Tensor],
    probe_batch_size: int,
) -> dict[str, float]:
    """Tests e.g. male_professor / female_nurse probe on professor / nurse labels"""
    test_accuracies = {}
    for class_name in all_class_list:
        if class_name not in dataset_info.PAIRED_CLASS_KEYS:
            continue
        spurious_class_names = [key for key in dataset_info.PAIRED_CLASS_KEYS if key != class_name]
        test_acts, test_labels = probe_training.prepare_probe_data(
            all_activations, class_name, spurious_corr=True
        )

        for spurious_class_name in spurious_class_names:
            test_acc_probe = probe_training.test_probe_gpu(
                test_acts,
                test_labels,
                probe_batch_size,
                probes[spurious_class_name],
            )
            combined_class_name = f"{spurious_class_name} probe on {class_name} data"
            test_accuracies[combined_class_name] = test_acc_probe

    return test_accuracies


def perform_feature_ablations(
    probes: dict[str, probe_training.Probe],
    sae: SAE,
    sae_batch_size: int,
    all_test_acts_BLD: dict[str, torch.Tensor],
    node_effects: dict[str, torch.Tensor],
    top_n_values: list[int],
    chosen_classes: list[str],
    probe_batch_size: int,
    spurious_corr: bool,
) -> dict[str, dict[int, dict[str, float]]]:
    ablated_class_accuracies = {}
    for ablated_class_name in chosen_classes:
        ablated_class_accuracies[ablated_class_name] = {}
        for top_n in top_n_values:
            selected_features_F = select_top_n_features(
                node_effects[ablated_class_name], top_n, ablated_class_name
            )
            test_acts_ablated = {}
            for evaluated_class_name in all_test_acts_BLD.keys():
                test_acts_ablated[evaluated_class_name] = ablated_precomputed_activations(
                    all_test_acts_BLD[evaluated_class_name],
                    sae,
                    selected_features_F,
                    sae_batch_size,
                )

            ablated_class_accuracies[ablated_class_name][top_n] = get_probe_test_accuracy(
                probes, chosen_classes, test_acts_ablated, probe_batch_size, spurious_corr
            )
    return ablated_class_accuracies


def get_spurious_correlation_plotting_dict(
    class_accuracies: dict[str, dict[int, dict[str, float]]],
    llm_clean_accs: dict[str, float],
) -> dict[str, float]:
    """raw_results: dict[class_name][threshold][class_name] = float
    llm_clean_accs: dict[class_name] = float
    Returns: dict[metric_name] = float"""

    results = {}
    eval_probe_class_id = "male_professor / female_nurse"

    dirs = [1, 2]

    dir1_class_name = f"{eval_probe_class_id} probe on professor / nurse data"
    dir2_class_name = f"{eval_probe_class_id} probe on male / female data"

    dir1_acc = llm_clean_accs[dir1_class_name]
    dir2_acc = llm_clean_accs[dir2_class_name]

    for dir in dirs:
        if dir == 1:
            ablated_probe_class_id = "male / female"
            eval_data_class_id = "professor / nurse"
        elif dir == 2:
            ablated_probe_class_id = "professor / nurse"
            eval_data_class_id = "male / female"
        else:
            raise ValueError("Invalid dir.")

        for threshold in class_accuracies[ablated_probe_class_id]:
            clean_acc = llm_clean_accs[eval_data_class_id]

            combined_class_name = f"{eval_probe_class_id} probe on {eval_data_class_id} data"

            original_acc = llm_clean_accs[combined_class_name]

            changed_acc = class_accuracies[ablated_probe_class_id][threshold][combined_class_name]

            changed_acc = (changed_acc - original_acc) / (clean_acc - original_acc)
            metric_key = f"scr_dir{dir}_threshold_{threshold}"

            results[metric_key] = changed_acc

            scr_metric_key = f"scr_metric_threshold_{threshold}"
            if dir1_acc < dir2_acc and dir == 1:
                results[scr_metric_key] = changed_acc
            elif dir1_acc > dir2_acc and dir == 2:
                results[scr_metric_key] = changed_acc

    return results


def create_tpp_plotting_dict(
    class_accuracies: dict[str, dict[int, dict[str, float]]],
    llm_clean_accs: dict[str, float],
) -> dict[str, float]:
    """raw_results: dict[class_name][threshold][class_name] = float
    llm_clean_accs: dict[class_name] = float
    Returns: dict[metric_name] = float"""

    results = {}
    intended_diffs = {}
    unintended_diffs = {}

    classes = list(llm_clean_accs.keys())

    for class_name in classes:
        if " probe on " in class_name:
            raise ValueError("This is SHIFT spurious correlations, shouldn't be here.")

        intended_clean_acc = llm_clean_accs[class_name]

        for threshold in class_accuracies[class_name]:
            intended_patched_acc = class_accuracies[class_name][threshold][class_name]

            intended_diff = intended_clean_acc - intended_patched_acc

            if threshold not in intended_diffs:
                intended_diffs[threshold] = []

            intended_diffs[threshold].append(intended_diff)

        for intended_class_id in classes:
            for unintended_class_id in classes:
                if intended_class_id == unintended_class_id:
                    continue

                unintended_clean_acc = llm_clean_accs[unintended_class_id]

                for threshold in class_accuracies[intended_class_id]:
                    unintended_patched_acc = class_accuracies[intended_class_id][threshold][
                        unintended_class_id
                    ]
                    unintended_diff = unintended_clean_acc - unintended_patched_acc

                    if threshold not in unintended_diffs:
                        unintended_diffs[threshold] = []

                    unintended_diffs[threshold].append(unintended_diff)

        for threshold in intended_diffs:
            assert threshold in unintended_diffs

            average_intended_diff = sum(intended_diffs[threshold]) / len(intended_diffs[threshold])
            average_unintended_diff = sum(unintended_diffs[threshold]) / len(
                unintended_diffs[threshold]
            )
            average_diff = average_intended_diff - average_unintended_diff

            results[f"tpp_threshold_{threshold}_total_metric"] = average_diff
            results[f"tpp_threshold_{threshold}_intended_diff_only"] = average_intended_diff
            results[f"tpp_threshold_{threshold}_unintended_diff_only"] = average_unintended_diff

    return results


def get_dataset_activations(
    dataset_name: str,
    config: eval_config.EvalConfig,
    model: HookedTransformer,
    llm_batch_size: int,
    layer: int,
    hook_point: str,
    device: str,
    chosen_classes: list[str],
    column1_vals: Optional[tuple[str, str]] = None,
    column2_vals: Optional[tuple[str, str]] = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    train_data, test_data = dataset_creation.get_train_test_data(
        dataset_name,
        config.spurious_corr,
        config.train_set_size,
        config.test_set_size,
        config.random_seed,
        column1_vals,
        column2_vals,
    )

    if not config.spurious_corr:
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
    save_activations: bool = True,
    column1_vals: Optional[tuple[str, str]] = None,
) -> tuple[dict[str, dict[str, dict[int, dict[str, float]]]], dict[str, float]]:
    """Return dict is of the form:
    dict[ablated_class_name][threshold][measured_acc_class_name] = float

    config: eval_config.EvalConfig contains all hyperparameters to reproduce the evaluation.
    It is saved in the results_dict for reproducibility."""

    column2_vals = COLUMN2_VALS_LOOKUP[dataset_name]

    if not config.spurious_corr:
        chosen_classes = dataset_info.chosen_classes_per_dataset[dataset_name]
        activations_filename = f"{dataset_name}_activations.pt".replace("/", "_")
        probes_filename = f"{dataset_name}_probes.pkl".replace("/", "_")
    else:
        chosen_classes = list(dataset_info.PAIRED_CLASS_KEYS.keys())
        activations_filename = (
            f"{dataset_name}_{column1_vals[0]}_{column1_vals[1]}_activations.pt".replace("/", "_")
        )
        probes_filename = f"{dataset_name}_{column1_vals[0]}_{column1_vals[1]}_probes.pkl".replace(
            "/", "_"
        )

    activations_path = os.path.join(artifacts_folder, activations_filename)
    probes_path = os.path.join(artifacts_folder, probes_filename)

    if not os.path.exists(activations_path):
        all_train_acts_BLD, all_test_acts_BLD = get_dataset_activations(
            dataset_name,
            config,
            model,
            config.llm_batch_size,
            layer,
            hook_point,
            device,
            chosen_classes,
            column1_vals,
            column2_vals,
        )

        all_meaned_train_acts_BD = activation_collection.create_meaned_model_activations(
            all_train_acts_BLD
        )
        all_meaned_test_acts_BD = activation_collection.create_meaned_model_activations(
            all_test_acts_BLD
        )

        torch.set_grad_enabled(True)

        llm_probes, llm_test_accuracies = probe_training.train_probe_on_activations(
            all_meaned_train_acts_BD,
            all_meaned_test_acts_BD,
            select_top_k=None,
            use_sklearn=False,
            batch_size=config.probe_train_batch_size,
            epochs=config.probe_epochs,
            lr=config.probe_lr,
            spurious_corr=config.spurious_corr,
            early_stopping_patience=config.early_stopping_patience,
        )

        torch.set_grad_enabled(False)

        llm_test_accuracies = get_probe_test_accuracy(
            llm_probes,
            chosen_classes,
            all_meaned_test_acts_BD,
            config.probe_test_batch_size,
            config.spurious_corr,
        )

        acts = {
            "train": all_train_acts_BLD,
            "test": all_test_acts_BLD,
        }

        llm_probes_dict = {
            "llm_probes": llm_probes,
            "llm_test_accuracies": llm_test_accuracies,
        }

        if save_activations:
            torch.save(acts, activations_path)
            with open(probes_path, "wb") as f:
                pickle.dump(llm_probes_dict, f)
    else:
        print(f"Loading activations from {activations_path}")
        acts = torch.load(activations_path)
        all_train_acts_BLD = acts["train"]
        all_test_acts_BLD = acts["test"]

        print(f"Loading probes from {probes_path}")
        with open(probes_path, "rb") as f:
            llm_probes_dict = pickle.load(f)

        llm_probes = llm_probes_dict["llm_probes"]
        llm_test_accuracies = llm_probes_dict["llm_test_accuracies"]

    torch.set_grad_enabled(False)

    sae_node_effects = get_all_node_effects_for_one_sae(
        sae,
        llm_probes,
        chosen_classes,
        config.spurious_corr,
        all_train_acts_BLD,
        config.sae_batch_size,
    )

    ablated_class_accuracies = perform_feature_ablations(
        llm_probes,
        sae,
        config.sae_batch_size,
        all_test_acts_BLD,
        sae_node_effects,
        config.n_values,
        chosen_classes,
        config.probe_test_batch_size,
        config.spurious_corr,
    )

    return ablated_class_accuracies, llm_test_accuracies


def run_eval_single_sae(
    config: eval_config.EvalConfig,
    sae: SAE,
    model: HookedTransformer,
    layer: int,
    hook_point: str,
    device: str,
    artifacts_folder: str,
    save_activations: bool = True,
) -> dict[str, float | dict[str, float]]:
    """hook_point: str is transformer lens format. example: f'blocks.{layer}.hook_resid_post'
    By default, we save activations for all datasets, and then reuse them for each sae.
    This is important to avoid recomputing activations for each SAE, and to ensure that the same activations are used for all SAEs.
    However, it can use 10s of GBs of disk space."""

    random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)

    dataset_results = {}

    averaging_names = []

    for dataset_name in config.dataset_names:
        if config.spurious_corr:
            column1_vals_list = config.column1_vals_lookup[dataset_name]
            for column1_vals in column1_vals_list:
                run_name = f"{dataset_name}_scr_{column1_vals[0]}_{column1_vals[1]}"
                raw_results, llm_clean_accs = run_eval_single_dataset(
                    dataset_name,
                    config,
                    sae,
                    model,
                    layer,
                    hook_point,
                    device,
                    artifacts_folder,
                    save_activations,
                    column1_vals,
                )

                processed_results = get_spurious_correlation_plotting_dict(
                    raw_results, llm_clean_accs
                )

                dataset_results[f"{run_name}_results"] = processed_results

                averaging_names.append(run_name)

        else:
            run_name = f"{dataset_name}_tpp"
            raw_results, llm_clean_accs = run_eval_single_dataset(
                dataset_name,
                config,
                sae,
                model,
                layer,
                hook_point,
                device,
                artifacts_folder,
                save_activations,
            )

            processed_results = create_tpp_plotting_dict(raw_results, llm_clean_accs)
            dataset_results[f"{run_name}_results"] = processed_results

            averaging_names.append(run_name)

    results_dict = formatting_utils.average_results_dictionaries(dataset_results, averaging_names)
    results_dict.update(dataset_results)

    return results_dict


def run_eval(
    config: eval_config.EvalConfig,
    selected_saes_dict: dict[str, list[str]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
    clean_up_activations: bool = True,
):
    """By default, clean_up_activations is True, which means that the activations are deleted after the evaluation is done.
    This is because activations for all datasets can easily be 10s of GBs.
    Return dict is a dict of SAE name: evaluation results for that SAE."""
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_version = get_sae_bench_version()

    if config.spurious_corr:
        eval_type = "scr"
    else:
        eval_type = "tpp"
    output_path = os.path.join(output_path, eval_type)
    os.makedirs(output_path, exist_ok=True)

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
                artifacts_base_folder, eval_type, config.model_name, sae.cfg.hook_name
            )
            os.makedirs(artifacts_folder, exist_ok=True)

            sae_result_file = f"{sae_release}_{sae_id}_eval_results.json"
            sae_result_file = sae_result_file.replace("/", "_")
            sae_result_path = os.path.join(output_path, sae_result_file)

            if os.path.exists(sae_result_path) and not force_rerun:
                print(f"Loading existing results from {sae_result_path}")
                with open(sae_result_path, "r") as f:
                    shift_or_tpp_results = json.load(f)
            else:
                shift_or_tpp_results = run_eval_single_sae(
                    config,
                    sae,
                    model,
                    sae.cfg.hook_layer,
                    sae.cfg.hook_name,
                    device,
                    artifacts_folder,
                )

            sae_eval_result = {
                "eval_instance_id": eval_instance_id,
                "sae_lens_release": sae_release,
                "sae_lens_id": sae_id,
                "eval_type_id": eval_type,
                "sae_lens_version": sae_lens_version,
                "sae_bench_version": sae_bench_version,
                "date_time": datetime.now().isoformat(),
                "eval_config": asdict(config),
                "eval_results": shift_or_tpp_results,
                "eval_artifacts": {"artifacts": os.path.relpath(artifacts_folder)},
            }

            results_dict[sae_id] = sae_eval_result

            # Save individual SAE results
            with open(sae_result_path, "w") as f:
                json.dump(sae_eval_result, f, indent=4)

    if clean_up_activations:
        shutil.rmtree(artifacts_folder)

    return results_dict


def setup_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
        spurious_corr=args.spurious_corr,
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
        "--output_folder", type=str, default="evals/shift_and_tpp/results", help="Output folder"
    )
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun of experiments")
    parser.add_argument(
        "--clean_up_activations", action="store_true", help="Clean up activations after evaluation"
    )

    def str_to_bool(value):
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise argparse.ArgumentTypeError("Boolean value expected.")

    parser.add_argument(
        "--spurious_corr",
        type=str_to_bool,
        required=True,
        help="If true, do Spurious Correlation Removal. If false, do TPP.",
    )

    return parser


if __name__ == "__main__":
    """
    Example pythia-70m usage:
    python evals/shift_and_tpp/main.py \
    --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
    --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
    --model_name pythia-70m-deduped \
    --spurious_corr true

    Example Gemma-2-2B SAE Bench usage:
    python evals/shift_and_tpp/main.py \
    --sae_regex_pattern "sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824" \
    --sae_block_pattern "blocks.19.hook_resid_post__trainer_2" \
    --model_name gemma-2-2b \
    --spurious_corr true

    Example Gemma-2-2B Gemma-Scope usage:
    python evals/shift_and_tpp/main.py \
    --sae_regex_pattern "gemma-scope-2b-pt-res" \
    --sae_block_pattern "layer_20/width_16k/average_l0_139" \
    --model_name gemma-2-2b \
    --spurious_corr true
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

    # For Gemma-2-2b
    sae_regex_patterns = [
        r"sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824",
        r"sae_bench_gemma-2-2b_sweep_standard_ctx128_ef8_0824",
        r"(gemma-scope-2b-pt-res)",
    ]
    sae_block_pattern = [
        r".*blocks\.19(?!.*step).*",
        r".*blocks\.19(?!.*step).*",
        r".*layer_(19).*(16k).*",
    ]

    sae_regex_patterns = None
    sae_block_pattern = None

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

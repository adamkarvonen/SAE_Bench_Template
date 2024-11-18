import gc
import os
import shutil
import random
import time
from dataclasses import asdict
from typing import Optional

import einops
from pydantic import TypeAdapter
import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer
import argparse
from datetime import datetime
import pickle

import evals.scr_and_tpp.dataset_creation as dataset_creation
from evals.scr_and_tpp.eval_config import ScrAndTppEvalConfig
from evals.scr_and_tpp.eval_output import (
    EVAL_TYPE_ID_SCR,
    EVAL_TYPE_ID_TPP,
    ScrEvalOutput,
    ScrMetricCategories,
    ScrResultDetail,
    ScrMetrics,
    TppEvalOutput,
    TppMetricCategories,
    TppResultDetail,
    TppMetrics,
)
import evals.sparse_probing.probe_training as probe_training
import sae_bench_utils.activation_collection as activation_collection
import sae_bench_utils.dataset_info as dataset_info
import sae_bench_utils.dataset_utils as dataset_utils
import sae_bench_utils.general_utils as general_utils
from sae_bench_utils import (
    get_eval_uuid,
    get_sae_lens_version,
    get_sae_bench_version,
)
from sae_bench_utils.sae_selection_utils import get_saes_from_regex

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
    perform_scr: bool,
    sae_batch_size: int,
) -> torch.Tensor:
    device = sae.device

    inputs_train_BLD, labels_train_B = probe_training.prepare_probe_data(
        precomputed_acts, class_idx, perform_scr
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

    if not perform_scr:
        # Only consider activations from the positive class
        average_acts_F.clamp_(min=0.0)

    effects_F = average_acts_F * dot_prod_F

    if perform_scr:
        effects_F = effects_F.abs()

    return effects_F


def get_all_node_effects_for_one_sae(
    sae: SAE,
    probes: dict[str, probe_training.Probe],
    chosen_class_indices: list[str],
    perform_scr: bool,
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
            perform_scr,
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
    perform_scr: bool,
) -> dict[str, float]:
    test_accuracies = {}
    for class_name in all_class_list:
        test_acts, test_labels = probe_training.prepare_probe_data(
            all_activations, class_name, perform_scr=perform_scr
        )

        test_acc_probe = probe_training.test_probe_gpu(
            test_acts,
            test_labels,
            probe_batch_size,
            probes[class_name],
        )
        test_accuracies[class_name] = test_acc_probe

    if perform_scr:
        scr_probe_accuracies = get_scr_probe_test_accuracy(
            probes, all_class_list, all_activations, probe_batch_size
        )
        test_accuracies.update(scr_probe_accuracies)

    return test_accuracies


def get_scr_probe_test_accuracy(
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
            all_activations, class_name, perform_scr=True
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
    perform_scr: bool,
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
                probes,
                chosen_classes,
                test_acts_ablated,
                probe_batch_size,
                perform_scr,
            )
    return ablated_class_accuracies


def get_scr_plotting_dict(
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

            scr_score = (changed_acc - original_acc) / (clean_acc - original_acc)

            print(
                f"dir: {dir}, original_acc: {original_acc}, clean_acc: {clean_acc}, changed_acc: {changed_acc}, scr_score: {scr_score}"
            )

            metric_key = f"scr_dir{dir}_threshold_{threshold}"

            results[metric_key] = scr_score

            scr_metric_key = f"scr_metric_threshold_{threshold}"
            if dir1_acc < dir2_acc and dir == 1:
                results[scr_metric_key] = scr_score
            elif dir1_acc > dir2_acc and dir == 2:
                results[scr_metric_key] = scr_score

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
            raise ValueError("This is SCR, shouldn't be here.")

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
    config: ScrAndTppEvalConfig,
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
        config.perform_scr,
        config.train_set_size,
        config.test_set_size,
        config.random_seed,
        column1_vals,
        column2_vals,
    )

    if not config.perform_scr:
        train_data = dataset_utils.filter_dataset(train_data, chosen_classes)
        test_data = dataset_utils.filter_dataset(test_data, chosen_classes)

    train_data = dataset_utils.tokenize_data_dictionary(
        train_data, model.tokenizer, config.context_length, device
    )
    test_data = dataset_utils.tokenize_data_dictionary(
        test_data, model.tokenizer, config.context_length, device
    )

    all_train_acts_BLD = activation_collection.get_all_llm_activations(
        train_data, model, llm_batch_size, layer, hook_point, mask_bos_pad_eos_tokens=True
    )
    all_test_acts_BLD = activation_collection.get_all_llm_activations(
        test_data, model, llm_batch_size, layer, hook_point, mask_bos_pad_eos_tokens=True
    )

    return all_train_acts_BLD, all_test_acts_BLD


def run_eval_single_dataset(
    dataset_name: str,
    config: ScrAndTppEvalConfig,
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

    if not config.perform_scr:
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
            perform_scr=config.perform_scr,
            early_stopping_patience=config.early_stopping_patience,
            l1_penalty=config.probe_l1_penalty,
        )

        torch.set_grad_enabled(False)

        llm_test_accuracies = get_probe_test_accuracy(
            llm_probes,
            chosen_classes,
            all_meaned_test_acts_BD,
            config.probe_test_batch_size,
            config.perform_scr,
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
        config.perform_scr,
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
        config.perform_scr,
    )

    return ablated_class_accuracies, llm_test_accuracies


def run_eval_single_sae(
    config: ScrAndTppEvalConfig,
    sae: SAE,
    model: HookedTransformer,
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

    os.makedirs(artifacts_folder, exist_ok=True)

    dataset_results = {}

    averaging_names = []

    for dataset_name in config.dataset_names:
        if config.perform_scr:
            column1_vals_list = config.column1_vals_lookup[dataset_name]
            for column1_vals in column1_vals_list:
                run_name = f"{dataset_name}_scr_{column1_vals[0]}_{column1_vals[1]}"
                raw_results, llm_clean_accs = run_eval_single_dataset(
                    dataset_name,
                    config,
                    sae,
                    model,
                    sae.cfg.hook_layer,
                    sae.cfg.hook_name,
                    device,
                    artifacts_folder,
                    save_activations,
                    column1_vals,
                )

                processed_results = get_scr_plotting_dict(raw_results, llm_clean_accs)

                dataset_results[f"{run_name}_results"] = processed_results

                averaging_names.append(run_name)

        else:
            run_name = f"{dataset_name}_tpp"
            raw_results, llm_clean_accs = run_eval_single_dataset(
                dataset_name,
                config,
                sae,
                model,
                sae.cfg.hook_layer,
                sae.cfg.hook_name,
                device,
                artifacts_folder,
                save_activations,
            )

            processed_results = create_tpp_plotting_dict(raw_results, llm_clean_accs)
            dataset_results[f"{run_name}_results"] = processed_results

            averaging_names.append(run_name)

    results_dict = general_utils.average_results_dictionaries(dataset_results, averaging_names)
    results_dict.update(dataset_results)

    return results_dict


def run_eval(
    config: ScrAndTppEvalConfig,
    selected_saes: list[tuple[str, SAE]] | list[tuple[str, str]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
    clean_up_activations: bool = False,
    save_activations: bool = True,
):
    """
    selected_saes is a list of either tuples of (sae_lens release, sae_lens id) or (sae_name, SAE object)

    If clean_up_activations is True, which means that the activations are deleted after the evaluation is done.
    You may want to use this because activations for all datasets can easily be 10s of GBs.
    Return dict is a dict of SAE name: evaluation results for that SAE."""
    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    if config.perform_scr:
        eval_type = EVAL_TYPE_ID_SCR
    else:
        eval_type = EVAL_TYPE_ID_TPP
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

        artifacts_folder = os.path.join(
            artifacts_base_folder, eval_type, config.model_name, sae.cfg.hook_name
        )

        sae_result_file = f"{sae_release}_{sae_id}_eval_results.json"
        sae_result_file = sae_result_file.replace("/", "_")
        sae_result_path = os.path.join(output_path, sae_result_file)

        if os.path.exists(sae_result_path) and not force_rerun:
            print(f"Loading existing results from {sae_result_path}")
            with open(sae_result_path, "r") as f:
                if eval_type == EVAL_TYPE_ID_SCR:
                    eval_output = TypeAdapter(ScrEvalOutput).validate_json(f.read())
                elif eval_type == EVAL_TYPE_ID_TPP:
                    eval_output = TypeAdapter(TppEvalOutput).validate_json(f.read())
                else:
                    raise ValueError(f"Invalid eval type: {eval_type}")
        else:
            scr_or_tpp_results = run_eval_single_sae(
                config,
                sae,
                model,
                device,
                artifacts_folder,
                save_activations,
            )
            if eval_type == EVAL_TYPE_ID_SCR:
                eval_output = ScrEvalOutput(
                    eval_type_id=eval_type,
                    eval_config=config,
                    eval_id=eval_instance_id,
                    datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
                    eval_result_metrics=ScrMetricCategories(
                        scr_metrics=ScrMetrics(
                            **{
                                k: v
                                for k, v in scr_or_tpp_results.items()
                                if not isinstance(v, dict)
                            }
                        )
                    ),
                    eval_result_details=[
                        ScrResultDetail(
                            dataset_name=dataset_name,
                            **result,
                        )
                        for dataset_name, result in scr_or_tpp_results.items()
                        if isinstance(result, dict)
                    ],
                    sae_bench_commit_hash=sae_bench_commit_hash,
                    sae_lens_id=sae_id,
                    sae_lens_release_id=sae_release,
                    sae_lens_version=sae_lens_version,
                )
            elif eval_type == EVAL_TYPE_ID_TPP:
                eval_output = TppEvalOutput(
                    eval_type_id=eval_type,
                    eval_config=config,
                    eval_id=eval_instance_id,
                    datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
                    eval_result_metrics=TppMetricCategories(
                        tpp_metrics=TppMetrics(
                            **{
                                k: v
                                for k, v in scr_or_tpp_results.items()
                                if not isinstance(v, dict)
                            }
                        )
                    ),
                    eval_result_details=[
                        TppResultDetail(
                            dataset_name=dataset_name,
                            **result,
                        )
                        for dataset_name, result in scr_or_tpp_results.items()
                        if isinstance(result, dict)
                    ],
                    sae_bench_commit_hash=sae_bench_commit_hash,
                    sae_lens_id=sae_id,
                    sae_lens_release_id=sae_release,
                    sae_lens_version=sae_lens_version,
                )
            else:
                raise ValueError(f"Invalid eval type: {eval_type}")

        results_dict[f"{sae_release}_{sae_id}"] = asdict(eval_output)

        eval_output.to_json_file(sae_result_path, indent=2)

    if clean_up_activations:
        if os.path.exists(artifacts_folder):
            shutil.rmtree(artifacts_folder)

    return results_dict


def create_config_and_selected_saes(
    args,
) -> tuple[ScrAndTppEvalConfig, list[tuple[str, str]]]:
    config = ScrAndTppEvalConfig(
        random_seed=args.random_seed,
        model_name=args.model_name,
        perform_scr=args.perform_scr,
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
        default="eval_results",
        help="Output folder",
    )
    parser.add_argument("--force_rerun", action="store_true", help="Force rerun of experiments")
    parser.add_argument(
        "--clean_up_activations",
        action="store_true",
        help="Clean up activations after evaluation",
    )
    parser.add_argument(
        "--save_activations",
        action="store_false",
        help="Save the generated LLM activations for later use",
    )

    def str_to_bool(value):
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise argparse.ArgumentTypeError("Boolean value expected.")

    parser.add_argument(
        "--perform_scr",
        type=str_to_bool,
        required=True,
        help="If true, do Spurious Correlation Removal (SCR). If false, do TPP.",
    )

    return parser


if __name__ == "__main__":
    """
    Example pythia-70m usage:
    python evals/scr_and_tpp/main.py \
    --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
    --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
    --model_name pythia-70m-deduped \
    --perform_scr true

    Example Gemma-2-2B SAE Bench usage:
    python evals/scr_and_tpp/main.py \
    --sae_regex_pattern "sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824" \
    --sae_block_pattern "blocks.19.hook_resid_post__trainer_2" \
    --model_name gemma-2-2b \
    --perform_scr true

    Example Gemma-2-2B Gemma-Scope usage:
    python evals/scr_and_tpp/main.py \
    --sae_regex_pattern "gemma-scope-2b-pt-res" \
    --sae_block_pattern "layer_20/width_16k/average_l0_139" \
    --model_name gemma-2-2b \
    --perform_scr true
    """
    args = arg_parser().parse_args()
    device = general_utils.setup_environment()

    start_time = time.time()

    config, selected_saes = create_config_and_selected_saes(args)

    print(selected_saes)

    config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
    config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

    # create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # run the evaluation on all selected SAEs
    results_dict = run_eval(
        config,
        selected_saes,
        device,
        args.output_folder,
        args.force_rerun,
        args.clean_up_activations,
        args.save_activations,
    )

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time} seconds")


# Use this code snippet to use custom SAE objects
# if __name__ == "__main__":
#     import baselines.identity_sae as identity_sae
#     import baselines.jumprelu_sae as jumprelu_sae

#     """
#     python evals/scr_and_tpp/main.py
#     """
#     device = general_utils.setup_environment()

#     start_time = time.time()

#     random_seed = 42
#     output_folder = "eval_results"
#     perform_scr = True

#     model_name = "gemma-2-2b"
#     hook_layer = 20

#     repo_id = "google/gemma-scope-2b-pt-res"
#     filename = f"layer_{hook_layer}/width_16k/average_l0_71/params.npz"
#     sae = jumprelu_sae.load_jumprelu_sae(repo_id, filename, hook_layer)
#     selected_saes = [(f"{repo_id}_{filename}_gemmascope_sae", sae)]

#     config = ScrAndTppEvalConfig(
#         random_seed=random_seed,
#         model_name=model_name,
#         perform_scr=perform_scr,
#     )

#     config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
#     config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

#     # create output folder
#     os.makedirs(output_folder, exist_ok=True)

#     # run the evaluation on all selected SAEs
#     results_dict = run_eval(
#         config,
#         selected_saes,
#         device,
#         output_folder,
#         force_rerun=True,
#         clean_up_activations=False,
#         save_activations=True,
#     )

#     end_time = time.time()

#     print(f"Finished evaluation in {end_time - start_time} seconds")

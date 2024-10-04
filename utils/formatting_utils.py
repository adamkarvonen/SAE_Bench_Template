import json
import pandas as pd
import re
from typing import Optional, Union
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory

from sae_lens.sae import SAE, TopK


def fix_topk_saes(sae: SAE, sae_release: str, sae_name: str, data_dir: str = "") -> SAE:
    """Temporary workaround as the TopK SAEs are currently being loaded as Standard SAEs.
    DEPRECATED, fixed in sae_lens version 3.23.1"""

    if isinstance(sae.activation_fn, TopK):
        print(f"SAE {sae_name} already has TopK activation function.")
        return sae

    sae_data_filename = f"{data_dir}sae_bench_data/{sae_release}_data.json"

    with open(sae_data_filename, "r") as f:
        sae_data = json.load(f)

    k = sae_data["sae_config_dictionary_learning"][sae_name]["trainer"]["k"]
    sae.activation_fn = TopK(k=k)

    return sae


def make_available_sae_df(for_printing: bool = False) -> pd.DataFrame:
    """Extract info on SAEs selected for our benchmarking project from SAE-Lens"""

    df = pd.DataFrame.from_records(
        {k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}
    ).T
    df.drop(
        columns=["expected_var_explained", "expected_l0", "config_overrides", "conversion_func"],
        inplace=True,
    )
    filtered_df = df[
        df.release.str.contains("bench")
    ]  # Each row is a "release" which has multiple SAEs which may have different configs / match different hook points in a model.

    if not for_printing:
        return filtered_df

    # Format df for readability
    def _analyze_dict_keys(data):
        hook_points = set()
        trainers = set()
        has_checkpoints = False

        for key in data.keys():
            # Extract hook point
            match = re.match(r"(.*?)__trainer", key)
            if match:
                hook_points.add(match.group(1))

            # Extract trainer number
            trainer_match = re.search(r"trainer_(\d+)", key)
            if trainer_match:
                trainers.add(int(trainer_match.group(1)))

            # Check for checkpoints
            if "_step_" in key:
                has_checkpoints = True

        return {
            "unique_hook_points": list(hook_points),
            "n_trainers": max(trainers) + 1 if trainers else 0,
            "has_checkpoints": has_checkpoints,
        }

    overview_df = filtered_df.reset_index(drop=True)
    analyzed_data = overview_df["neuronpedia_id"].apply(_analyze_dict_keys)
    overview_df["unique_hook_points"] = analyzed_data.apply(lambda x: x["unique_hook_points"])
    overview_df["n_saes_per_hook"] = analyzed_data.apply(lambda x: x["n_trainers"])
    overview_df["has_training_checkpoints"] = analyzed_data.apply(lambda x: x["has_checkpoints"])

    return overview_df


def extract_sae_info(sae_name: str) -> dict:
    """Extracts information from an SAE name string"""
    info = {}
    info["is_checkpoint"] = "checkpoints" in sae_name

    layer_match = re.search(r"layer_(\d+)", sae_name)
    if layer_match:
        info["layer"] = int(layer_match.group(1))
    else:
        raise ValueError(f"Could not find layer in {sae_name}")

    trainer_match = re.search(r"trainer_(\d+)", sae_name)
    if trainer_match:
        info["trainer_id"] = int(trainer_match.group(1))
    else:
        raise ValueError(f"Could not find trainer in {sae_name}")

    return info


def extract_saes_unique_info(sae_names: list[str], checkpoint_only: bool = False) -> dict:
    infos = {
        "layers": set(),
        "trainer_ids": set(),
    }
    for sae_name in sae_names:
        info = extract_sae_info(sae_name)
        if checkpoint_only and not info["is_checkpoint"]:
            continue
        infos["trainer_ids"].add(info["trainer_id"])
        infos["layers"].add(info["layer"])

    return infos


def find_gemmascope_average_l0_sae_names(
    layer_num: int, gemmascope_release_name: str = "gemma-scope-2b-pt-res", width_num: str = "16k"
) -> list[str]:
    df = pd.DataFrame.from_records(
        {k: v.__dict__ for k, v in get_pretrained_saes_directory().items()}
    ).T
    filtered_df = df[df.release == gemmascope_release_name]
    name_to_id_map = filtered_df.saes_map.item()

    pattern = rf"layer_{layer_num}/width_{width_num}/average_l0_\d+"

    matching_keys = [key for key in name_to_id_map.keys() if re.match(pattern, key)]

    return matching_keys


def filter_sae_names(
    sae_names: Union[list[str], str],
    layers: list[int],
    include_checkpoints: bool,
    trainer_ids: Optional[list[int]],
    drop_sae_bench_prefix: bool = True,
) -> list[str]:
    """Filter SAE names based on layer, trainer_id, and whether they are checkpoints
    Args:
        sae_names: List of SAE names or a string representing a release name
        layers: List of layer numbers to include
        trainer_ids: Optional list of trainer ids to include. If None, all trainers are included.
        include_checkpoints: Bool whether to include SAEs that are checkpoints. By default, this is True, as it isn't included in the sae names in SAE Lens.
    """
    filtered_sae_names = []

    if isinstance(sae_names, str):
        sae_df = make_available_sae_df()
        sae_df = sae_df[sae_df["release"] == sae_names]
        sae_names = sae_df.saes_map.item().values()

    for sae_name in sae_names:
        info = extract_sae_info(sae_name)
        if info["layer"] in layers:
            if not include_checkpoints and info["is_checkpoint"]:
                continue
            if trainer_ids:
                if info["trainer_id"] not in trainer_ids:
                    continue
            if drop_sae_bench_prefix:
                sae_name = sae_name.replace("sae_bench_", "")
            filtered_sae_names.append(sae_name)

    return filtered_sae_names


def get_sparsity_penalty(config: dict) -> float:
    trainer_class = config["trainer"]["trainer_class"]
    if trainer_class == "TrainerTopK":
        return config["trainer"]["k"]
    elif trainer_class == "PAnnealTrainer":
        return config["trainer"]["sparsity_penalty"]
    else:
        return config["trainer"]["l1_penalty"]


def ae_config_results(ae_paths: list[str], dictionaries_path: str) -> dict[str, dict[str, float]]:
    results = {}
    for ae_path in ae_paths:
        config_file = f"{ae_path}/config.json"

        with open(config_file, "r") as f:
            config = json.load(f)

        ae_name = ae_path.split(dictionaries_path)[1]

        results[ae_name] = {}

        trainer_class = config["trainer"]["trainer_class"]
        results[ae_name]["trainer_class"] = trainer_class
        results[ae_name]["l1_penalty"] = get_sparsity_penalty(config)

        results[ae_name]["lr"] = config["trainer"]["lr"]
        results[ae_name]["dict_size"] = config["trainer"]["dict_size"]
        if "steps" in config["trainer"]:
            results[ae_name]["steps"] = config["trainer"]["steps"]
        else:
            results[ae_name]["steps"] = -1

    return results


def add_custom_metric_results(
    ae_paths: list[str],
    results: dict[str, dict[str, float]],
    metric_filename: str,
    dictionaries_path: str,
    metric_dict_key: Optional[str] = None,
) -> dict[str, dict[str, float]]:
    for ae_path in ae_paths:
        config_file = f"{ae_path}/{metric_filename}"

        with open(config_file, "r") as f:
            custom_metric_results = json.load(f)

        ae_name = ae_path.split(dictionaries_path)[1]

        if metric_dict_key:
            results[ae_name]["custom_metric"] = custom_metric_results[metric_dict_key]
        else:
            for key, value in custom_metric_results.items():
                results[ae_name][key] = value

    return results


def filter_by_l0_threshold(results: dict, l0_threshold: Optional[int]) -> dict:
    if l0_threshold is not None:
        filtered_results = {
            path: data for path, data in results.items() if data["l0"] <= l0_threshold
        }

        # Optional: Print how many results were filtered out
        filtered_count = len(results) - len(filtered_results)
        print(f"Filtered out {filtered_count} results with L0 > {l0_threshold}")

        # Replace the original results with the filtered results
        results = filtered_results
    return results

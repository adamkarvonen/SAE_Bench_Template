import json
from typing import Optional

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
        results[ae_name]["l1_penalty"] = get_sparsity_penalty(config, trainer_class)

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
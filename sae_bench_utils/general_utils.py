import pandas as pd
import re
import os
import torch
from typing import Optional, Callable, Any, Union, Type
import time
import functools
import random
from sae_lens.toolkit.pretrained_saes_directory import get_pretrained_saes_directory
from sae_lens import SAE


def str_to_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str.lower())
    if dtype is None:
        raise ValueError(
            f"Unsupported dtype: {dtype_str}. Supported dtypes: {list(dtype_map.keys())}"
        )
    return dtype


def setup_environment():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    return device


def load_and_format_sae(
    sae_release_or_unique_id: str, sae_object_or_sae_lens_id: str | SAE, device: str
) -> tuple[str, SAE, Optional[torch.Tensor]]:
    """Handle both pretrained SAEs (identified by string) and custom SAEs (passed as objects)"""
    if isinstance(sae_object_or_sae_lens_id, str):
        sae, _, sparsity = SAE.from_pretrained(
            release=sae_release_or_unique_id,
            sae_id=sae_object_or_sae_lens_id,
            device=device,
        )
        sae_id = sae_object_or_sae_lens_id
    else:
        sae = sae_object_or_sae_lens_id
        sae_id = "custom_sae"
        sparsity = None

    return sae_id, sae, sparsity


def get_results_filepath(output_path: str, sae_release: str, sae_id: str) -> str:
    sae_result_file = f"{sae_release}_{sae_id}_eval_results.json"
    sae_result_file = sae_result_file.replace("/", "_")
    sae_result_path = os.path.join(output_path, sae_result_file)

    return sae_result_path


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


def get_sparsity_penalty(config: dict) -> float:
    trainer_class = config["trainer"]["trainer_class"]
    if trainer_class == "TrainerTopK":
        return config["trainer"]["k"]
    elif trainer_class == "PAnnealTrainer":
        return config["trainer"]["sparsity_penalty"]
    else:
        return config["trainer"]["l1_penalty"]


def average_results_dictionaries(
    results_dict: dict[str, dict[str, float]], dataset_names: list[str]
) -> dict[str, float]:
    """If we have multiple dicts of results from separate datasets, get an average performance over all datasets.
    Results_dict is dataset -> dict of metric_name : float result"""
    averaged_results = {}
    aggregated_results = {}

    for dataset_name in dataset_names:
        dataset_results = results_dict[f"{dataset_name}_results"]

        for metric_name, metric_value in dataset_results.items():
            if metric_name not in aggregated_results:
                aggregated_results[metric_name] = []

            aggregated_results[metric_name].append(metric_value)

    averaged_results = {}
    for metric_name, values in aggregated_results.items():
        average_value = sum(values) / len(values)
        averaged_results[metric_name] = average_value

    return averaged_results


def retry_with_exponential_backoff(
    retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: Union[Type[Exception], tuple[Type[Exception], ...]] = Exception,
) -> Callable:
    """
    Decorator for retrying a function with exponential backoff.

    Args:
        retries: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delay
        exceptions: Exception(s) to catch and retry on
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception = None

            for retry_count in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if retry_count == retries:
                        print(f"Failed after {retries} retries: {str(e)}")
                        raise

                    # Calculate delay with optional jitter
                    current_delay = min(delay * (exponential_base**retry_count), max_delay)
                    if jitter:
                        current_delay *= 1 + random.random() * 0.1  # 10% jitter

                    print(
                        f"Attempt {retry_count + 1}/{retries} failed: {str(e)}. "
                        f"Retrying in {current_delay:.2f} seconds..."
                    )
                    time.sleep(current_delay)

            if last_exception:
                raise last_exception
            return None

        return wrapper

    return decorator

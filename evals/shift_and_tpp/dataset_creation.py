import gc
import math
import os
import pickle
import random
import sys
from collections import defaultdict
from typing import Callable, Optional

import datasets
import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
from datasets import load_dataset
from sklearn.utils import shuffle
from torch import nn
from tqdm import tqdm

import sae_bench_utils.dataset_info as dataset_info
import sae_bench_utils.dataset_utils as dataset_utils


def get_spurious_corr_data(
    df: pd.DataFrame,
    column1_vals: tuple[str, str],
    column2_vals: tuple[str, str],
    dataset_name: str,
    min_samples_per_quadrant: int,
    random_seed: int,
) -> dict[str, list[str]]:
    """Returns a dataset of, in the case of bias_in_bios, a key that's something like `female_nurse_data_only`,
    and a value that's a list of bios (strs) of len min_samples_per_quadrant * 2."""
    balanced_data = {}

    text_column_name = dataset_info.dataset_metadata[dataset_name]["text_column_name"]
    column1_name = dataset_info.dataset_metadata[dataset_name]["column1_name"]
    column2_name = dataset_info.dataset_metadata[dataset_name]["column2_name"]

    column1_pos = column1_vals[0]
    column1_neg = column1_vals[1]
    column2_pos = column2_vals[0]
    column2_neg = column2_vals[1]

    # NOTE: This is a bit confusing. We select rows from the dataset based on column1_vals and column2_vals,
    # but below, we hardcode the keys as male / female, professor / nurse, etc
    column1_pos_idx = dataset_info.dataset_metadata[dataset_name]["column1_mapping"][column1_pos]
    column1_neg_idx = dataset_info.dataset_metadata[dataset_name]["column1_mapping"][column1_neg]
    column2_pos_idx = dataset_info.dataset_metadata[dataset_name]["column2_mapping"][column2_pos]
    column2_neg_idx = dataset_info.dataset_metadata[dataset_name]["column2_mapping"][column2_neg]

    pos_neg = df[(df[column1_name] == column1_neg_idx) & (df[column2_name] == column2_pos_idx)][
        text_column_name
    ].tolist()
    neg_neg = df[(df[column1_name] == column1_neg_idx) & (df[column2_name] == column2_neg_idx)][
        text_column_name
    ].tolist()

    pos_pos = df[(df[column1_name] == column1_pos_idx) & (df[column2_name] == column2_pos_idx)][
        text_column_name
    ].tolist()
    neg_pos = df[(df[column1_name] == column1_pos_idx) & (df[column2_name] == column2_neg_idx)][
        text_column_name
    ].tolist()

    min_count = min(
        len(pos_neg), len(neg_neg), len(pos_pos), len(neg_pos), min_samples_per_quadrant
    )

    assert min_count == min_samples_per_quadrant

    # For biased classes, we don't have two quadrants per label
    assert len(pos_pos) > min_samples_per_quadrant * 2
    assert len(neg_neg) > min_samples_per_quadrant * 2

    # Create and shuffle combinations
    combined_pos = pos_pos[:min_count] + pos_neg[:min_count]
    combined_neg = neg_pos[:min_count] + neg_neg[:min_count]
    pos_combined = pos_pos[:min_count] + neg_pos[:min_count]
    neg_combined = pos_neg[:min_count] + neg_neg[:min_count]
    pos_pos = pos_pos[: min_count * 2]
    neg_neg = neg_neg[: min_count * 2]

    # Shuffle each combination
    rng = np.random.default_rng(random_seed)

    rng.shuffle(combined_pos)
    rng.shuffle(combined_neg)
    rng.shuffle(pos_combined)
    rng.shuffle(neg_combined)
    rng.shuffle(pos_pos)
    rng.shuffle(neg_neg)

    # Assign to balanced_data
    balanced_data["male / female"] = combined_pos  # male data only, to be combined with female data
    balanced_data["female_data_only"] = combined_neg  # female data only
    balanced_data["professor / nurse"] = (
        pos_combined  # professor data only, to be combined with nurse data
    )
    balanced_data["nurse_data_only"] = neg_combined  # nurse data only
    balanced_data["male_professor / female_nurse"] = (
        pos_pos  # male_professor data only, to be combined with female_nurse data
    )
    balanced_data["female_nurse_data_only"] = neg_neg  # female_nurse data only

    for key in balanced_data.keys():
        balanced_data[key] = balanced_data[key][: min_samples_per_quadrant * 2]
        assert len(balanced_data[key]) == min_samples_per_quadrant * 2

    return balanced_data


def get_train_test_data(
    dataset_name: str,
    spurious_corr: bool,
    train_set_size: int,
    test_set_size: int,
    random_seed: int,
    column1_vals: Optional[tuple[str, str]] = None,
    column2_vals: Optional[tuple[str, str]] = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    if spurious_corr:
        assert "bias_in_bios" in dataset_name or "amazon_reviews" in dataset_name

        dataset_name = dataset_name.split("_class_set")[0]
        dataset = load_dataset(dataset_name)
        train_df = pd.DataFrame(dataset["train"])
        test_df = pd.DataFrame(dataset["test"])

        # 4 is because male / gender for each profession
        minimum_train_samples_per_quadrant = train_set_size // 4
        minimum_test_samples_per_quadrant = test_set_size // 4

        train_bios = get_spurious_corr_data(
            train_df,
            column1_vals,
            column2_vals,
            dataset_name,
            minimum_train_samples_per_quadrant,
            random_seed,
        )

        test_bios = get_spurious_corr_data(
            test_df,
            column1_vals,
            column2_vals,
            dataset_name,
            minimum_test_samples_per_quadrant,
            random_seed,
        )

    else:
        train_bios, test_bios = dataset_utils.get_multi_label_train_test_data(
            dataset_name, train_set_size, test_set_size, random_seed
        )

    train_bios, test_bios = dataset_utils.ensure_shared_keys(train_bios, test_bios)

    return train_bios, test_bios

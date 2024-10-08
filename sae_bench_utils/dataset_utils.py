from tqdm import tqdm
from typing import Callable, Optional
import torch
import pandas as pd

from transformers import AutoTokenizer
from datasets import load_dataset

import utils.dataset_info as dataset_info


# Load and prepare dataset
def load_huggingface_dataset(dataset_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if dataset_name == "bias_in_bios":
        dataset = load_dataset("LabHC/bias_in_bios")
        train_df = pd.DataFrame(dataset["train"])
        test_df = pd.DataFrame(dataset["test"])
    elif dataset_name == "amazon_reviews_all_ratings":
        dataset = load_dataset(
            "canrager/amazon_reviews_mcauley",
            config_name="dataset_all_categories_and_ratings_train1000_test250",
        )
    elif dataset_name == "amazon_reviews_1and5":
        dataset = load_dataset(
            "canrager/amazon_reviews_mcauley_1and5",
        )
        train_df = pd.DataFrame(dataset["train"])
        test_df = pd.DataFrame(dataset["test"])
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    return train_df, test_df


def get_balanced_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    min_samples_per_quadrant: int,
    random_seed: int,
) -> dict[str, list[str]]:
    """Returns a dataset of, in the case of bias_in_bios, a key of profession idx,
    and a value of a list of bios (strs) of len min_samples_per_quadrant * 2."""

    text_column_name = dataset_info.dataset_metadata[dataset_name]["text_column_name"]
    column1_name = dataset_info.dataset_metadata[dataset_name]["column1_name"]
    column2_name = dataset_info.dataset_metadata[dataset_name]["column2_name"]

    balanced_df_list = []

    for profession in tqdm(df[column1_name].unique()):
        prof_df = df[df[column1_name] == profession]
        min_count = prof_df[column2_name].value_counts().min()

        unique_groups = prof_df[column2_name].unique()
        if len(unique_groups) < 2:
            continue  # Skip professions with less than two groups

        if min_count < min_samples_per_quadrant:
            continue

        balanced_prof_df = pd.concat(
            [
                group.sample(n=min_samples_per_quadrant, random_state=random_seed)
                for _, group in prof_df.groupby(column2_name)
            ]
        ).reset_index(drop=True)
        balanced_df_list.append(balanced_prof_df)

    balanced_df = pd.concat(balanced_df_list).reset_index(drop=True)
    grouped = balanced_df.groupby(column1_name)[text_column_name].apply(list)

    str_data = {str(key): texts for key, texts in grouped.items()}

    balanced_data = {label: texts for label, texts in str_data.items()}

    for key in balanced_data.keys():
        balanced_data[key] = balanced_data[key][: min_samples_per_quadrant * 2]
        assert len(balanced_data[key]) == min_samples_per_quadrant * 2

    return balanced_data


def ensure_shared_keys(train_data: dict, test_data: dict) -> tuple[dict, dict]:
    # Find keys that are in test but not in train
    test_only_keys = set(test_data.keys()) - set(train_data.keys())

    # Find keys that are in train but not in test
    train_only_keys = set(train_data.keys()) - set(test_data.keys())

    # Remove keys from test that are not in train
    for key in test_only_keys:
        print(f"Removing {key} from test set")
        del test_data[key]

    # Remove keys from train that are not in test
    for key in train_only_keys:
        print(f"Removing {key} from train set")
        del train_data[key]

    return train_data, test_data


def get_multi_label_train_test_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    train_set_size: int,
    test_set_size: int,
    random_seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Returns a dict of [class_name, list[str]]"""
    # 4 is because male / gender for each profession
    minimum_train_samples_per_quadrant = train_set_size // 4
    minimum_test_samples_per_quadrant = test_set_size // 4

    train_data = get_balanced_dataset(
        train_df,
        dataset_name,
        minimum_train_samples_per_quadrant,
        random_seed=random_seed,
    )
    test_data = get_balanced_dataset(
        test_df,
        dataset_name,
        minimum_test_samples_per_quadrant,
        random_seed=random_seed,
    )

    train_data, test_data = ensure_shared_keys(train_data, test_data)

    return train_data, test_data


def tokenize_data(
    data: dict[str, list[str]], tokenizer: AutoTokenizer, max_length: int, device: str
) -> dict[str, dict]:
    tokenized_data = {}
    for key, texts in tqdm(data.items(), desc="Tokenizing data"):
        # .data so we have a dict, not a BatchEncoding
        tokenized_data[key] = (
            tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            .to(device)
            .data
        )
    return tokenized_data


def filter_dataset(
    data: dict[str, list[str]], chosen_class_indices: list[str]
) -> dict[str, list[str]]:
    filtered_data = {}
    for class_name in chosen_class_indices:
        filtered_data[class_name] = data[class_name]
    return filtered_data

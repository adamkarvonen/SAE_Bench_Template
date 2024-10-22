from typing import Callable, Optional
from collections import defaultdict
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
import random

import sae_bench_utils.dataset_info as dataset_info


def gather_dataset_from_df(
    df: pd.DataFrame,
    chosen_classes: list[str],
    min_samples_per_category: int,
    label_key: str,
    text_key: str,
    random_seed: int,
) -> dict[str, list[str]]:
    random.seed(random_seed)

    data = {}

    for chosen_class in chosen_classes:
        class_df = df[df[label_key] == chosen_class]

        sampled_texts = (
            class_df[text_key].sample(n=min_samples_per_category, random_state=random_seed).tolist()
        )
        assert len(sampled_texts) == min_samples_per_category

        data[str(chosen_class)] = sampled_texts

    return data


def get_ag_news_dataset(
    dataset_name: str,
    chosen_classes: list[str],
    train_set_size: int,
    test_set_size: int,
    random_seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    random.seed(random_seed)

    dataset = load_dataset(dataset_name, streaming=False)
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    # It's a binary classification task, so we need to halve the train and test sizes
    train_size = train_set_size // 2
    test_size = test_set_size // 2

    # convert str to int, as labels are stored as ints
    chosen_classes = [int(chosen_class) for chosen_class in chosen_classes]

    train_data = gather_dataset_from_df(
        train_df, chosen_classes, train_size, "label", "text", random_seed
    )
    test_data = gather_dataset_from_df(
        test_df, chosen_classes, test_size, "label", "text", random_seed
    )

    return train_data, test_data


def get_europarl_dataset(
    dataset_name: str,
    chosen_languages: list[str],
    train_size: int,
    test_size: int,
    random_seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    random.seed(random_seed)
    label_key = "translation"
    language_pairs = {
        "en": "en-fr",
        "fr": "fr-it",
        "de": "de-en",
        "es": "es-fr",
        "nl": "nl-pt",
    }

    # It's a binary classification task, so we need to halve the train and test sizes
    train_size = train_size // 2
    test_size = test_size // 2

    samples_per_language = train_size + test_size

    samples_by_language = defaultdict(list)

    print(f"Loading dataset {dataset_name}, this usually takes ~10 seconds")

    for language, language_pair in language_pairs.items():
        # Filter out languages that are not in the dataset
        dataset = load_dataset(
            dataset_name,
            language_pair,
            streaming=True,
            split="train",
        )

        # Collect samples for each language
        for sample in dataset:
            # Extract the text in the target language
            text = sample[label_key][language]
            samples_by_language[language].append(text)

            # Check if we have enough samples for all languages
            if len(samples_by_language[language]) > samples_per_language:
                break

    # Split samples into train and test sets
    train_samples = {}
    test_samples = {}

    for language in chosen_languages:
        lang_samples = samples_by_language[language]

        random.shuffle(lang_samples)
        train_samples[language] = lang_samples[:train_size]
        test_samples[language] = lang_samples[train_size : train_size + test_size]
        assert len(train_samples[language]) == train_size
        assert len(test_samples[language]) == test_size

    return train_samples, test_samples


def get_github_code_dataset(
    dataset_name: str,
    chosen_classes: list[str],
    train_size: int,
    test_size: int,
    random_seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Following the Neurons in a Haystack paper, we skip the first 50 tokens of each code snippet to avoid the license header.
    We use characters instead of tokens to avoid tokenization issues."""
    tokens_to_skip = 50
    ctx_len = 128
    chars_per_token = 3
    ctx_len_chars = ctx_len * chars_per_token
    chars_to_skip = tokens_to_skip * chars_per_token

    random.seed(random_seed)
    label_key = "language"

    # It's a binary classification task, so we need to halve the train and test sizes
    train_size = train_size // 2
    test_size = test_size // 2

    print(f"Loading dataset {dataset_name}, this usually takes ~30 seconds")

    # Filter out languages that are not in the dataset
    dataset = load_dataset(
        dataset_name,
        streaming=True,
        split="train",
        trust_remote_code=True,
        languages=chosen_classes,
    )

    total_size = train_size + test_size

    all_samples = defaultdict(list)

    # Collect samples for each language
    for sample in dataset:
        if sample[label_key] in chosen_classes:
            code = sample["code"]

            # In "Neurons in a Haystack", the authors skipped the first 50 tokens to avoid the license header
            # This is using characters so it's tokenizer agnostic
            if len(code) > (ctx_len_chars + chars_to_skip):
                code = code[chars_to_skip:]
                all_samples[sample[label_key]].append(code)

            # Check if we have collected enough samples for all languages
            if all(len(all_samples[lang]) > total_size for lang in chosen_classes):
                break

    # Split samples into train and test sets
    train_samples = {}
    test_samples = {}

    for lang in chosen_classes:
        lang_samples = all_samples[lang]

        random.shuffle(lang_samples)
        train_samples[lang] = lang_samples[:train_size]
        test_samples[lang] = lang_samples[train_size : train_size + test_size]
        assert len(train_samples[lang]) == train_size
        assert len(test_samples[lang]) == test_size

    return train_samples, test_samples


def get_balanced_dataset(
    df: pd.DataFrame,
    dataset_name: str,
    min_samples_per_quadrant: int,
    random_seed: int,
) -> dict[str, list[str]]:
    """This function is used for the amazon reviews dataset and the bias_in_bios dataset, which have two columns.

    Returns a balanced dataset as a dictionary, where each key corresponds to a unique value
    in one column, and each value is a list of text entries balanced across categories
    in the other column.

    Examples: For the 'bias_in_bios' dataset where `column1` is 'Profession' and `column2` is 'Gender':
        - If `balance_by_column1` is `True`:
            - Balances bios for each profession by gender.
            - Returns a dict with professions as keys and lists of bios as values.
    """

    text_column_name = dataset_info.dataset_metadata[dataset_name]["text_column_name"]
    column1_name = dataset_info.dataset_metadata[dataset_name]["column1_name"]
    column2_name = dataset_info.dataset_metadata[dataset_name]["column2_name"]

    balanced_data = {}

    for profession in tqdm(df[column1_name].unique()):
        prof_df = df[df[column1_name] == profession]
        unique_groups = prof_df[column2_name].unique()
        min_count = prof_df[column2_name].value_counts().min()

        if len(unique_groups) < 2:
            continue  # Skip professions with less than two groups

        if min_count < min_samples_per_quadrant:
            continue

        sampled_texts = []
        for _, group_df in prof_df.groupby(column2_name):
            sampled_group = group_df.sample(n=min_samples_per_quadrant, random_state=random_seed)
            sampled_texts.extend(sampled_group[text_column_name].tolist())

        balanced_data[str(profession)] = sampled_texts

        assert len(balanced_data[str(profession)]) == min_samples_per_quadrant * 2

    return balanced_data


def get_bias_in_bios_or_amazon_product_dataset(
    dataset_name: str, train_set_size: int, test_set_size: int, random_seed: int
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    dataset_name = dataset_name.split("_class_set")[0]

    dataset = load_dataset(dataset_name)
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    # 4 is because male / female split for each profession, 2 quadrants per profession, 2 professions for binary task
    minimum_train_samples_per_quadrant = train_set_size // 4
    minimum_test_samples_per_quadrant = test_set_size // 4

    train_data = get_balanced_dataset(
        train_df, dataset_name, minimum_train_samples_per_quadrant, random_seed
    )
    test_data = get_balanced_dataset(
        test_df, dataset_name, minimum_test_samples_per_quadrant, random_seed
    )

    return train_data, test_data


def get_amazon_sentiment_dataset(
    dataset_name: str, train_set_size: int, test_set_size: int, random_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_name = dataset_name.split("_sentiment")[0]
    dataset = load_dataset(dataset_name)
    train_df = pd.DataFrame(dataset["train"])
    test_df = pd.DataFrame(dataset["test"])

    minimum_train_samples_per_category = train_set_size // 2
    minimum_test_samples_per_category = test_set_size // 2

    train_data = get_balanced_amazon_sentiment_dataset(
        train_df, minimum_train_samples_per_category, random_seed
    )
    test_data = get_balanced_amazon_sentiment_dataset(
        test_df, minimum_test_samples_per_category, random_seed
    )

    return train_data, test_data


def get_balanced_amazon_sentiment_dataset(
    df: pd.DataFrame,
    min_samples_per_category: int,
    random_seed: int,
) -> dict[str, list[str]]:
    text_column_name = "text"
    column2_name = "rating"

    balanced_data = {}

    unique_ratings = df[column2_name].unique()

    for rating in unique_ratings:
        # Filter dataframe for current rating
        df_rating = df[df[column2_name] == rating]

        sampled_texts = (
            df_rating[text_column_name]
            .sample(n=min_samples_per_category, random_state=random_seed)
            .tolist()
        )
        assert len(sampled_texts) == min_samples_per_category

        balanced_data[str(rating)] = sampled_texts

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
    dataset_name: str,
    train_set_size: int,
    test_set_size: int,
    random_seed: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Returns a dict of [class_name, list[str]]"""

    if "bias_in_bios" in dataset_name or "canrager/amazon_reviews_mcauley_1and5" == dataset_name:
        train_data, test_data = get_bias_in_bios_or_amazon_product_dataset(
            dataset_name, train_set_size, test_set_size, random_seed
        )
    elif dataset_name == "canrager/amazon_reviews_mcauley_1and5_sentiment":
        train_data, test_data = get_amazon_sentiment_dataset(
            dataset_name, train_set_size, test_set_size, random_seed
        )
    elif dataset_name == "codeparrot/github-code":
        train_data, test_data = get_github_code_dataset(
            dataset_name,
            dataset_info.chosen_classes_per_dataset[dataset_name],
            train_set_size,
            test_set_size,
            random_seed,
        )
    elif dataset_name == "fancyzhx/ag_news":
        train_data, test_data = get_ag_news_dataset(
            dataset_name,
            dataset_info.chosen_classes_per_dataset[dataset_name],
            train_set_size,
            test_set_size,
            random_seed,
        )
    elif dataset_name == "Helsinki-NLP/europarl":
        train_data, test_data = get_europarl_dataset(
            dataset_name,
            dataset_info.chosen_classes_per_dataset[dataset_name],
            train_set_size,
            test_set_size,
            random_seed,
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")

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

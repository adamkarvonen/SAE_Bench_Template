from tqdm import tqdm
from transformers import AutoTokenizer


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

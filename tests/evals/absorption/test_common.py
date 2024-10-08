import pandas as pd
import torch
from transformers import PreTrainedTokenizerFast

from evals.absorption.common import (
    _parse_probe_data_split,
)


def test_parse_probe_data_split(gpt2_tokenizer: PreTrainedTokenizerFast):
    split_activations = torch.randn(4, 12)
    split_labels = [5, 1, 18, 22]
    df = pd.DataFrame(
        {
            "token": ["fish", "bird", "shark", "whale"],
        }
    )
    activations, vocab_with_labels = _parse_probe_data_split(
        gpt2_tokenizer, split_activations, split_labels, df
    )
    assert torch.allclose(activations, split_activations)
    assert vocab_with_labels == [
        ("fish", 5),
        ("bird", 1),
        ("shark", 18),
        ("whale", 22),
    ]


def test_parse_probe_data_split_removes_invalid_rows(
    gpt2_tokenizer: PreTrainedTokenizerFast,
):
    split_activations = torch.randn(5, 12)
    split_labels = [5, 1, 18, 22, 23]
    df = pd.DataFrame(
        {
            "token": [
                "fish",
                "bird",
                float("nan"),
                "whale",
                "<0x6A>",
            ],
        }
    )
    activations, vocab_with_labels = _parse_probe_data_split(
        gpt2_tokenizer, split_activations, split_labels, df
    )
    assert torch.allclose(activations, split_activations[[0, 1, 3]])
    assert vocab_with_labels == [
        ("fish", 5),
        ("bird", 1),
        ("whale", 22),
    ]


def test_parse_probe_data_split_replaces_special_token_chars(
    gpt2_tokenizer: PreTrainedTokenizerFast,
):
    split_activations = torch.randn(2, 12)
    split_labels = [18, 22]
    df = pd.DataFrame(
        {
            "token": [
                "Ġsculpt",
                "whale",
            ],
        }
    )
    activations, vocab_with_labels = _parse_probe_data_split(
        gpt2_tokenizer, split_activations, split_labels, df
    )
    assert torch.allclose(activations, split_activations)
    assert vocab_with_labels == [
        (" sculpt", 18),
        ("whale", 22),
    ]

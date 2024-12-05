"""Utility functions for training and evaluating interventions."""

import collections
import copy
import numpy as np
import re
from typing import List, Optional
from transformer_lens import HookedTransformer

from datasets import Dataset
# from methods.distributed_alignment_search import LowRankRotatedSpaceIntervention
# from methods.pca import PCARotatedSpaceIntervention
# from methods.sparse_autoencoder import AutoencoderIntervention
# import pyvene as pv
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
# from utils.dataset_utils import get_dataloader, get_label_offset

def find_positions_in_tokens(
    start: int,
    stop: int,
    tokens: Optional[List[str]] = None,
    has_bos: Optional[bool] = True,
    text: Optional[str] = None,
    model: Optional[HookedTransformer] = None,
    ) -> List[int]:
    """
    Given a text and two indices into it, find the indices of the *tokens*
    that correspond to the text between these indices INCLUSIVE.

    These indices are relative to the result of `model.to_str_tokens(text)`,
    which means they take into account the BOS token.

    This function should be used as a final step after you've done some string
    processing to decide where to start and stop in the text (because dealing
    with the tokenization is a bit tricky).
    """
    assert start <= stop
    if tokens is not None:
        if has_bos:
            text = ''.join(tokens[1:])
            tokens = tokens[1:] # get rid of BOS
        else:
            text = ''.join(tokens)

    else:
        assert text is not None
        assert model is not None
        tokens = model.to_str_tokens(text)[1:] # get rid of BOS
    ### by this point, `tokens` must hold the tokenization of the text WITHOUT a
    ### BOS token.

    assert start <= len(text) - 1
    assert stop <= len(text) - 1
    token_lengths = [len(token) for token in tokens]
    cum_lengths = np.cumsum([0] + token_lengths)
    tok_idx_to_char_idxs = {}
    for i, tok in enumerate(tokens):
        tok_idx_to_char_idxs[i] = list(range(cum_lengths[i], cum_lengths[i+1]))
    # now invert this
    char_idx_to_tok_idx = {}
    for tok_idx, char_idxs in tok_idx_to_char_idxs.items():
        for char_idx in char_idxs:
            char_idx_to_tok_idx[char_idx] = tok_idx
    # now find the token indices
    start_token_idx = char_idx_to_tok_idx[start]
    stop_token_idx = char_idx_to_tok_idx[stop]
    result_without_bos = list(range(start_token_idx, stop_token_idx + 1))
    return [idx + 1 for idx in result_without_bos] # add 1 to account for BOS
from dataclasses import dataclass, asdict
from typing import Optional, List
from enum import Enum
import torch


@dataclass
class EvalConfig:
    random_seed: int = 42
    model_dtype: torch.dtype = torch.bfloat16

    dataset_name: str = "bias_in_bios"
    chosen_classes = ["0", "1", "2", "6", "9"]

    probe_train_set_size: int = 4000
    probe_test_set_size: int = 1000
    context_length: int = 128

    probe_batch_size: int = 250
    epochs: int = 10
    lr: float = 1e-3

    sae_batch_size: int = 500

    sae_release: str = "sae_bench_pythia70m_sweep_topk_ctx128_0730"
    model_name: str = "pythia-70m-deduped"

    k_values = [1, 2, 5, 10, 20, 50, 100]

    saes = []
    layers = [4]
    trainer_ids = [2, 10, 18]

    for layer in layers:
        for i in trainer_ids:
            saes.append(f"blocks.{layer}.hook_resid_post__trainer_{i}")

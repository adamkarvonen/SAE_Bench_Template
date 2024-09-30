import re
import torch
from dataclasses import dataclass, asdict
from typing import Optional, List
from enum import Enum

# TODO make import from shared directory more robust
import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/share/u/can/SAE_Bench_Template')
from formatting_utils import filter_sae_names


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
    sae_batch_size: int = 125

    ## Uncomment to run Pythia SAEs
    # sae_release: str = "sae_bench_pythia70m_sweep_topk_ctx128_0730"
    # model_name: str = "pythia-70m-deduped"
    # layers = [4]
    # trainer_ids = list(range(20))
    # include_checkpoints = False


    ## Uncomment to run Gemma SAEs
    sae_release: str = "sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824"
    model_name: str = "gemma-2-2b"
    layers = [19]
    trainer_ids = list(range(6))
    include_checkpoints = True
    k_values = [1, 2, 5, 10, 20, 50, 100]

    # Find all sae_names that match the given criteria
    def __post_init__(self):
        self.saes = filter_sae_names(self.sae_release, self.layers, self.trainer_ids, self.include_checkpoints)
        print("SAEs: ", self.saes)

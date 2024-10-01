from dataclasses import dataclass, field
import torch

# TODO make import from shared directory more robust
# I wanted to avoid the pip install -e . in the shared directory, but maybe that's the best way to do it
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from formatting_utils import filter_sae_names


@dataclass
class EvalConfig:
    random_seed: int = 42

    dataset_name: str = "bias_in_bios"
    chosen_classes: list[str] = field(default_factory=lambda: ["0", "1", "2", "6", "9"])

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
    # layers: list[int] = field(default_factory=lambda: [4])
    # trainer_ids: list[int] = field(default_factory=lambda: list(range(20)))
    # include_checkpoints: bool = False

    ## Uncomment to run Gemma SAEs

    sae_release: str = "sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824"
    model_name: str = "gemma-2-2b"
    layers: list[int] = field(default_factory=lambda: [19])
    trainer_ids: list[int] = field(default_factory=lambda: list(range(6)))
    include_checkpoints: bool = True

    k_values: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100])

    saes: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.saes = filter_sae_names(
            self.sae_release,
            self.layers,
            self.trainer_ids,
            self.include_checkpoints,
            drop_sae_bench_prefix=True,
        )

        print("SAEs: ", self.saes)

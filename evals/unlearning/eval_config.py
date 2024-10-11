from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class EvalConfig:
    random_seed: int = 42

    # dataset_names: list[str] = field(
    #     default_factory=lambda: ["bias_in_bios", "amazon_reviews_1and5"]
    # )

    # probe_train_set_size: int = 4000
    # probe_test_set_size: int = 1000
    # context_length: int = 128

    # sae_batch_size: int = 125

    sae_releases: list[str] = field(
        default_factory=lambda: [
            "gemma-scope-2b-pt-res",
            "sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824",
            "sae_bench_gemma-2-2b_sweep_standard_ctx128_ef8_0824",
        ]
    )
    model_name: str = "gemma-2-2b-it"
    layer: int = 3
    trainer_ids: Optional[list[int]] = None
    include_checkpoints: bool = False

    # k_values: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100])

    selected_saes_dict: dict = field(default_factory=lambda: {})

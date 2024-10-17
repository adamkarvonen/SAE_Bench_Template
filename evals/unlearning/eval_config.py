from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class EvalConfig:
    random_seed: int = 42

    all_dataset_names: list[str] = field(
        default_factory=lambda: [
            "wmdp-bio",
            "high_school_us_history",
            "college_computer_science",
            "high_school_geography",
            "human_aging",
            "college_biology",
        ]
    )

    intervention_method: str = "clamp_feature_activation"

    retain_thresholds: list[str] = field(default_factory=lambda: [0.01])
    n_features_list: list[str] = field(default_factory=lambda: [10, 20])
    multipliers: list[str] = field(default_factory=lambda: [10, 25, 50, 100])

    llm_batch_size: int = 4
    mcq_batch_size: int = 4

    dataset_size: int = 256
    seq_len: int = 1024

    n_batch_loss_added: int = 50
    target_metric: str = "correct"
    save_metrics: bool = True

    sae_releases: list[str] = field(
        default_factory=lambda: [
            # "gemma-scope-2b-pt-res",
            "sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824",
            # "sae_bench_gemma-2-2b_sweep_standard_ctx128_ef8_0824",
        ]
    )
    model_name: str = "gemma-2-2b-it"
    layer: int = 3
    # trainer_ids: Optional[list[int]] = None
    trainer_ids: Optional[list[int]] = field(default_factory=lambda: [2])
    include_checkpoints: bool = False

    selected_saes_dict: dict = field(default_factory=lambda: {})

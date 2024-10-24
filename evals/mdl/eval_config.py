from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalConfig:
    k_values: list[Optional[int]] = field(default_factory=lambda: [16, 24, 32])
    num_bins_values: list[int] = field(default_factory=lambda: [4, 6, 8, 12, 16, 32])

    random_seed: int = 42
    dataset_names: list[str] = field(default_factory=lambda: ["HuggingFaceFW/fineweb"])

    context_length: int = 128

    sae_batch_size: int = 64
    # sae_batch_size: int = 4

    selected_saes_dict: dict = field(default_factory=lambda: {})

    sae_releases: list[str] = field(
        default_factory=lambda: [
            "sae_bench_pythia70m_sweep_standard_ctx128_0712",
            "sae_bench_pythia70m_sweep_topk_ctx128_0730",
        ]
    )
    model_name: str = "pythia-70m-deduped"
    layer: int = 4
    trainer_ids: Optional[list[int]] = field(default_factory=lambda: list(range(20)))
    trainer_ids: Optional[list[int]] = field(default_factory=lambda: [10])
    include_checkpoints: bool = False

    mse_epsilon_threshold: float = 0.01

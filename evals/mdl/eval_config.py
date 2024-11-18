from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MDLEvalConfig:
    k_values: list[Optional[int]] = field(default_factory=lambda: [16, 24, 32])
    num_bins_values: list[int] = field(default_factory=lambda: [4, 6, 8, 12, 16, 32])

    random_seed: int = 42
    dataset_name: str = "HuggingFaceFW/fineweb"

    context_length: int = 128

    sae_batch_size: int = 64

    model_name: str = "pythia-70m-deduped"
    llm_dtype: str = "float32"

    mse_epsilon_threshold: float = 0.01

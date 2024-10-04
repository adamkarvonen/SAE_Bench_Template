from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalConfig:
    random_seed: int = 42

    train_portion: float = 0.5
    context_length: int = 128

    sae_batch_size: int = 125

    ## Uncomment to run Pythia SAEs

    sae_releases: list[str] = field(
        default_factory=lambda: [
            "sae_bench_pythia70m_sweep_standard_ctx128_0712",
            "sae_bench_pythia70m_sweep_topk_ctx128_0730",
        ]
    )
    model_name: str = "pythia-70m-deduped"
    layer: int = 4
    include_checkpoints: bool = False

    ## Uncomment to run Gemma SAEs

    # sae_releases: list[str] = field(
    #     default_factory=lambda: [
    #         "gemma-scope-2b-pt-res",
    #         "sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824",
    #         "sae_bench_gemma-2-2b_sweep_standard_ctx128_ef8_0824",
    #     ]
    # )
    # model_name: str = "gemma-2-2b"
    # layer: int = 19
    # trainer_ids: Optional[list[int]] = None
    # include_checkpoints: bool = False

    max_k_value: int = 10

    selected_saes_dict: dict = field(default_factory=lambda: {})

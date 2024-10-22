from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class EvalConfig:
    random_seed: int = 42

    dataset_names: list[str] = field(
        default_factory=lambda: [
            "LabHC/bias_in_bios_class_set1",
            "LabHC/bias_in_bios_class_set2",
            "LabHC/bias_in_bios_class_set3",
            "canrager/amazon_reviews_mcauley_1and5",
            "canrager/amazon_reviews_mcauley_1and5_sentiment",
            "codeparrot/github-code",
            "fancyzhx/ag_news",
            "Helsinki-NLP/europarl",
        ]
    )

    probe_train_set_size: int = 4000
    probe_test_set_size: int = 1000
    context_length: int = 128

    sae_batch_size: int = 125

    ## Uncomment to run Pythia SAEs

    # sae_releases: list[str] = field(
    #     default_factory=lambda: [
    #         "sae_bench_pythia70m_sweep_standard_ctx128_0712",
    #         "sae_bench_pythia70m_sweep_topk_ctx128_0730",
    #     ]
    # )
    # model_name: str = "pythia-70m-deduped"
    # layer: int = 4
    # trainer_ids: Optional[list[int]] = field(default_factory=lambda: list(range(20)))
    # trainer_ids: Optional[list[int]] = field(default_factory=lambda: [10])
    # include_checkpoints: bool = False

    ## Uncomment to run Gemma SAEs

    sae_releases: list[str] = field(
        default_factory=lambda: [
            "gemma-scope-2b-pt-res",
            "sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824",
            "sae_bench_gemma-2-2b_sweep_standard_ctx128_ef8_0824",
        ]
    )
    model_name: str = "gemma-2-2b"
    layer: int = 19
    trainer_ids: Optional[list[int]] = None
    # trainer_ids: Optional[list[int]] = field(default_factory=lambda: [2])
    include_checkpoints: bool = True

    k_values: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50])

    selected_saes_dict: dict = field(default_factory=lambda: {})

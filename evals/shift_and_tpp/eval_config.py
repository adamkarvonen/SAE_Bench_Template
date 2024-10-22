from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalConfig:
    random_seed: int = 42

    # dataset_names: list[str] = field(
    #     default_factory=lambda: [
    #         "LabHC/bias_in_bios_class_set1",
    #         "canrager/amazon_reviews_mcauley_1and5",
    #     ]
    # )

    dataset_names: list[str] = field(default_factory=lambda: ["LabHC/bias_in_bios_class_set1"])
    column1_vals_list: list[tuple[str, str]] = field(
        default_factory=list
    )  # This will be populated in run_eval()
    spurious_corr: bool = False

    # Load datset and probes
    train_set_size: int = 4000
    test_set_size: int = 1000  # This is limited as the test set is smaller than the train set

    context_length: int = 128
    probe_train_batch_size: int = (
        16  # We don't want probe batch size to be close to the train set size
    )
    probe_test_batch_size: int = min(500, test_set_size)
    probe_epochs: int = 20
    probe_lr: float = 1e-3

    sae_batch_size: int = 250

    # This is for spurrious correlation removal
    chosen_class_indices = [
        "male / female",
        "professor / nurse",
        "male_professor / female_nurse",
    ]

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

    n_values: list[int] = field(default_factory=lambda: [2, 5, 10, 20, 50, 100, 500, 1000, 2000])

    selected_saes_dict: dict = field(default_factory=lambda: {})

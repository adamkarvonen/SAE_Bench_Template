from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalConfig:
    random_seed: int = 42  # NOTE: This will be overwritten by argparse

    dataset_names: list[str] = field(
        default_factory=lambda: [
            "LabHC/bias_in_bios_class_set1",
            "canrager/amazon_reviews_mcauley_1and5",
        ]
    )

    spurious_corr: bool = True
    # Load datset and probes
    train_set_size: int = 4000
    test_set_size: int = 1000  # This is limited as the test set is smaller than the train set

    context_length: int = 128
    probe_train_batch_size: int = 16  # DO NOT CHANGE without reading the paper appendix Section 1
    probe_test_batch_size: int = min(500, test_set_size)
    probe_epochs: int = 20
    probe_lr: float = 1e-3

    sae_batch_size: int = 125
    llm_batch_size: int = 32
    llm_dtype: str = "bfloat16"  # asdict() doesn't like to serialize torch.dtype

    model_name: str = "pythia-70m-deduped"

    n_values: list[int] = field(default_factory=lambda: [2, 5, 10, 20, 50, 100, 500])

    column1_vals_lookup: dict[str, list[tuple[str, str]]] = field(
        default_factory=lambda: {
            "LabHC/bias_in_bios_class_set1": [
                ("professor", "nurse"),
                ("architect", "journalist"),
                ("surgeon", "psychologist"),
                ("attorney", "teacher"),
            ],
            "canrager/amazon_reviews_mcauley_1and5": [
                ("Books", "CDs_and_Vinyl"),
                ("Software", "Electronics"),
                ("Pet_Supplies", "Office_Products"),
                ("Industrial_and_Scientific", "Toys_and_Games"),
            ],
        }
    )

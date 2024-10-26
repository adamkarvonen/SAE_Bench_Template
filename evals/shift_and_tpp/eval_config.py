from pydantic.dataclasses import dataclass
from pydantic import Field, field_validator
from evals.base_eval_output import BaseEvalConfig


@dataclass
class ShiftAndTppEvalConfig(BaseEvalConfig):
    random_seed: int = Field(
        default=42,
        title="Random Seed",
        description="NOTE: This will be overwritten by argparse",
    )

    dataset_names: list[str] = Field(
        default_factory=lambda: [
            "LabHC/bias_in_bios_class_set1",
            "canrager/amazon_reviews_mcauley_1and5",
        ],
        title="Dataset Names",
        description="List of dataset names",
    )

    spurious_corr: bool = Field(
        default=True,
        title="Spurious Correlation",
        description="",
    )

    # This reduces randomness in the SCR results
    early_stopping_patience: int = Field(
        default=40,
        title="Early Stopping Patience",
        description="This reduces randomness in the SCR results.",
    )

    # Load datset and probes
    train_set_size: int = Field(
        default=4000,
        title="Train Set Size",
        description="",
    )
    test_set_size: int = Field(
        default=1000,
        title="Test Set Size",
        description="This is limited as the test set is smaller than the train set",
    )

    context_length: int = Field(
        default=128,
        title="Context Length",
        description="",
    )
    probe_train_batch_size: int = Field(
        default=16,
        title="Probe Train Batch Size",
        description="DO NOT CHANGE without reading the paper appendix Section 1",
    )

    @field_validator("probe_test_batch_size")
    def ensure_min_probe_test_batch_size(cls, value: int) -> int:
        return min(value, 500)

    probe_test_batch_size: int = Field(
        default=500,
        title="Probe Test Batch Size",
        description="",
    )
    probe_epochs: int = Field(
        default=20,
        title="Probe Epochs",
        description="",
    )
    probe_lr: float = Field(default=1e-3, title="Probe LR", description="")

    sae_batch_size: int = Field(
        default=125,
        title="SAE Batch Size",
        description="",
    )
    llm_batch_size: int = Field(
        default=32,
        title="LLM Batch Size",
        description="",
    )
    llm_dtype: str = Field(
        default="bfloat16",
        title="LLM Dtype",
        description="asdict() doesn't like to serialize torch.dtype",
    )

    model_name: str = Field(
        default="pythia-70m-deduped",
        title="Model Name",
        description="",
    )

    n_values: list[int] = Field(
        default_factory=lambda: [2, 5, 10, 20, 50, 100, 500],
        title="N Values",
        description="",
    )

    column1_vals_lookup: dict[str, list[tuple[str, str]]] = Field(
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
        },
        title="Column 1 Values Lookup",
        description="",
    )

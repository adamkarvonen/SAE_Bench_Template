from pydantic.dataclasses import dataclass
from pydantic import Field
from evals.base_eval_output import BaseEvalConfig


@dataclass
class SparseProbingEvalConfig(BaseEvalConfig):
    random_seed: int = Field(
        default=42,
        title="Random Seed",
        description="Random seed",
    )

    dataset_names: list[str] = Field(
        default_factory=lambda: [
            "LabHC/bias_in_bios_class_set1",
            "LabHC/bias_in_bios_class_set2",
            "LabHC/bias_in_bios_class_set3",
            "canrager/amazon_reviews_mcauley_1and5",
            "canrager/amazon_reviews_mcauley_1and5_sentiment",
            "codeparrot/github-code",
            "fancyzhx/ag_news",
            "Helsinki-NLP/europarl",
        ],
        title="Dataset Names",
        description="List of dataset names. We have at most 5 class names in a single subset, which is why we have multiple bias_in_bios class subsets.",
    )

    probe_train_set_size: int = Field(
        default=4000,
        title="Probe Train Set Size",
        description="Probe train set size",
    )
    probe_test_set_size: int = Field(
        default=1000,
        title="Probe Test Set Size",
        description="Probe test set size",
    )
    context_length: int = Field(
        default=128,
        title="LLM Context Length",
        description="The maximum length of each input to the LLM. Any longer inputs will be truncated, keeping only the beginning.",
    )

    sae_batch_size: int = Field(
        default=125,
        title="SAE Batch Size",
        description="SAE batch size, inference only",
    )
    llm_batch_size: int = Field(
        default=None,
        title="LLM Batch Size",
        description="LLM batch size. This is set by default in the main script, or it can be set with a command line argument.",
    )
    llm_dtype: str = Field(
        default="",
        title="LLM Data Type",
        description="LLM data type. This is set by default in the main script, or it can be set with a command line argument.",
    )

    model_name: str = Field(
        default="",
        title="Model Name",
        description="Model name. Must be set with a command line argument.",
    )

    k_values: list[int] = Field(
        default_factory=lambda: [1, 2, 5, 10, 20, 50],
        title="K Values",
        description="K represents the number of SAE features or residual stream channels we train the linear probe on. We iterate over all values of K.",
    )

    lower_vram_usage: bool = Field(
        default=False,
        title="Lower Memory Usage",
        description="Lower GPU memory usage by doing more computation on the CPU. Useful on 1M width SAEs. Will be slower and require more system memory.",
    )

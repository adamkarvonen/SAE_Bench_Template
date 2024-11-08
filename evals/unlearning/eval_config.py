from pydantic.dataclasses import dataclass
from pydantic import Field
from evals.base_eval_output import BaseEvalConfig

@dataclass
class UnlearningEvalConfig(BaseEvalConfig):
    random_seed: int = Field(default=42, title="Random Seed", description="Random seed")

    dataset_names: list[str] = Field(
        default_factory=lambda: [
            "wmdp-bio",
            "high_school_us_history",
            "college_computer_science",
            "high_school_geography",
            "human_aging",
            "college_biology",
        ],
        title="Dataset Names",
        description="List of dataset names",
    )

    intervention_method: str = Field(
        default="clamp_feature_activation",
        title="Intervention Method",
        description="Intervention method",
    )

    retain_thresholds: list[float] = Field(
        default_factory=lambda: [0.001, 0.01],
        title="Retain Thresholds",
        description="Retain thresholds",
    )
    n_features_list: list[int] = Field(
        default_factory=lambda: [10, 20],
        title="N Features List",
        description="N features list",
    )
    multipliers: list[int] = Field(
        default_factory=lambda: [25, 50, 100, 200],
        title="Multipliers",
        description="Multipliers",
    )

    llm_batch_size: int = Field(
        default=4,
        title="LLM Batch Size",
        description="LLM batch size",
    )
    mcq_batch_size: int = Field(
        default=8,
        title="MCQ Batch Size",
        description="MCQ batch size. Multiple choice questions are shorter, so we can afford a larger batch size",
    )

    dataset_size: int = Field(
        default=1024,
        title="Dataset Size",
        description="Dataset size",
    )
    seq_len: int = Field(
        default=1024,
        title="Sequence Length",
        description="Sequence length",
    )

    n_batch_loss_added: int = Field(
        default=50,
        title="N Batch Loss Added",
        description="N batch loss added",
    )
    target_metric: str = Field(
        default="correct",
        title="Target Metric",
        description="Target metric",
    )
    save_metrics: bool = Field(
        default=True,
        title="Save Metrics",
        description="Save metrics",
    )

    model_name: str = Field(
        default="gemma-2-2b-it",
        title="Model Name",
        description="Model name",
    )
    llm_dtype: str = Field(
        default="bfloat16",
        title="LLM Data Type",
        description="LLM data type",
    )

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
        ],
        title="Dataset Names",
        description="List of dataset names. We want to unlearn wmdp-bio while retaining knowledge in other datasets",
    )

    intervention_method: str = Field(
        default="clamp_feature_activation",
        title="Intervention Method",
        description="Intervention method. We only support 'clamp_feature_activation' for now",
    )

    retain_thresholds: list[float] = Field(
        default_factory=lambda: [0.001, 0.01],
        title="Retain Thresholds",
        description="We ignore features that activate more than this threshold on the retain dataset",
    )
    n_features_list: list[int] = Field(
        default_factory=lambda: [10, 20],
        title="N Features List",
        description="Each N is the number of features we select and clamp to a negative value",
    )
    multipliers: list[int] = Field(
        default_factory=lambda: [25, 50, 100, 200],
        title="Multipliers",
        description="A list of negative values. We iterate over this list, clamping the selected features to each value",
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
        description="Dataset size we use when calculating feature sparsity",
    )
    seq_len: int = Field(
        default=1024,
        title="Sequence Length",
        description="Sequence length when calculating feature sparsity",
    )

    n_batch_loss_added: int = Field(
        default=50,
        title="N Batch Loss Added",
        description="Number of batches to use when calculating the loss added by an intervention (currently not supported).",
    )
    target_metric: str = Field(
        default="correct",
        title="Target Metric",
        description="Controls the type of `question_ids` we load. We support 'correct', `correct-iff-question`, and `correct-no-tricks",
    )
    save_metrics: bool = Field(
        default=True,
        title="Save Metrics Flag",
        description="If true, we save the metrics for each set of intervention hyperparameters. This is required to be true currently, as the unlearning score is calculated over all results.",
    )

    model_name: str = Field(
        default="gemma-2-2b-it",
        title="Model Name",
        description="Model name. Note that this should be a instruct model.",
    )
    llm_dtype: str = Field(
        default="bfloat16",
        title="LLM Data Type",
        description="LLM data type",
    )

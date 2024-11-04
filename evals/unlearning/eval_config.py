from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    random_seed: int = 42

    all_dataset_names: list[str] = field(
        default_factory=lambda: [
            "wmdp-bio",
            "high_school_us_history",
            "college_computer_science",
            "high_school_geography",
            "human_aging",
            "college_biology",
        ]
    )

    intervention_method: str = "clamp_feature_activation"

    retain_thresholds: list[str] = field(default_factory=lambda: [0.001, 0.01])
    n_features_list: list[str] = field(default_factory=lambda: [10, 20])
    multipliers: list[str] = field(default_factory=lambda: [25, 50, 100, 200])

    llm_batch_size: int = 4
    # multiple choice questions are shorter, so we can afford a larger batch size
    mcq_batch_size: int = llm_batch_size * 2

    dataset_size: int = 1024
    seq_len: int = 1024

    n_batch_loss_added: int = 50
    target_metric: str = "correct"
    save_metrics: bool = True

    model_name: str = "gemma-2-2b-it"
    llm_dtype: str = "bfloat16"

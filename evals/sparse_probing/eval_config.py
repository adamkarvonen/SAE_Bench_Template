from dataclasses import dataclass, field


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
    llm_batch_size: int = 32
    llm_dtype: str = "bfloat16"  # asdict() doesn't like to serialize torch.dtype

    model_name: str = "gemma-2-2b"

    k_values: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50])

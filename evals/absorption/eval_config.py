from pydantic.dataclasses import dataclass
from pydantic import Field
from evals.base_eval_output import BaseEvalConfig


# Define the eval config, which inherits from BaseEvalConfig, and include fields with titles and descriptions.
@dataclass
class AbsorptionEvalConfig(BaseEvalConfig):
    random_seed: int = Field(
        default=42,
        title="Random Seed",
        description="Random seed",
    )
    f1_jump_threshold: float = Field(
        default=0.03,
        title="F1 Jump Threshold",
        description="F1 jump threshold",
    )
    max_k_value: int = Field(
        default=10,
        title="Max K Value",
        description="Max k value",
    )

    # double-check token_pos matches prompting_template for other tokenizers
    prompt_template: str = Field(
        default="{word} has the first letter:",
        title="Prompt Template",
        description="Prompt template",
    )
    prompt_token_pos: int = Field(
        default=-6,
        title="Prompt Token Position",
        description="Prompt token position",
    )

    model_name: str = Field(
        default="",
        title="Model Name",
        description="Model name. Must be set with a command line argument. For this eval, we currently recommend to only use models >= 2B parameters.",
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

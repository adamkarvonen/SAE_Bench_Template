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
        default="pythia-70m-deduped",
        title="Model Name",
        description="Model name",
    )
    llm_batch_size: int = Field(
        default=32,
        title="LLM Batch Size",
        description="LLM batch size, inference only",
    )
    llm_dtype: str = Field(
        default="bfloat16",
        title="LLM Data Type",
        description="LLM data type",
    )

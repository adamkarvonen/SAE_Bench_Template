from pydantic.dataclasses import dataclass
from pydantic import Field
from evals.base_eval_output import BaseEvalConfig


@dataclass
class RAVELEvalConfig(BaseEvalConfig):
    random_seed: int = Field(
        default=42,
        title="Random Seed",
        description="Random seed",
    )

    # Dataset

    

    # Language model and SAE
    model_name: str = Field(
        default="gemma-2-2b",
        title="Model Name",
        description="Model name",
    )
    llm_dtype: str = Field(
        default="bfloat16",
        title="LLM Data Type",
        description="LLM data type",
    )
    llm_batch_size: int = Field(
        default=32,
        title="LLM Batch Size",
        description="LLM batch size, inference only",
    )
    sae_batch_size: int = Field(
        default=125,
        title="SAE Batch Size",
        description="SAE batch size, inference only",
    )

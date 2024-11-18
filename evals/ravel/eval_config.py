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
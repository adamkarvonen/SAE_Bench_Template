from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, Field
from evals.unlearning.eval_config import UnlearningEvalConfig
from evals.base_eval_output import (
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    DEFAULT_DISPLAY,
    BaseResultDetail,
)

EVAL_TYPE_ID_UNLEARNING = "unlearning"


@dataclass
class UnlearningMetrics(BaseMetrics):
    unlearning_score: float = Field(
        title="Unlearning Score",
        description="Unlearning score",
        json_schema_extra=DEFAULT_DISPLAY,
    )

# Define the categories themselves
@dataclass
class UnlearningMetricCategories(BaseMetricCategories):
    unlearning: UnlearningMetrics = Field(
        title="Unlearning",
        description="Metrics related to unlearning",
    )
        
# Define the eval output
@dataclass(config=ConfigDict(title="Unlearning"))
class UnlearningEvalOutput(
    BaseEvalOutput[UnlearningEvalConfig, UnlearningMetricCategories, BaseResultDetail]
):
    """
    The output of unlearning evaluations.
    """

    eval_config: UnlearningEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: UnlearningMetricCategories

    eval_type_id: str = Field(
        default=EVAL_TYPE_ID_UNLEARNING,
        title="Eval Type ID",
        description="The type of the evaluation",
    )

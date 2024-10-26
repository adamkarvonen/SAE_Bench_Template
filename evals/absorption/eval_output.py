from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, Field, field_validator
from evals.absorption.eval_config import AbsorptionEvalConfig
from evals.base_eval_output import (
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    BaseResultDetail,
)

EVAL_TYPE_ID_ABSORPTION = "absorption_first_letter"


# Define the metrics for each metric category, and include a title and description for each.
@dataclass
class AbsorptionMeanMetrics(BaseMetrics):

    mean_absorption_score: float = Field(
        title="Mean Absorption Score",
        description="Average of the absorption scores across all letters",
        json_schema_extra={"default_display": True},
    )
    mean_num_split_features: float = Field(
        title="Mean Number of Split Features",
        description="Average number of split features across all letters",
        json_schema_extra={"default_display": True},
    )


# Define the categories themselves, and include a title and description for each.
@dataclass
class AbsorptionMetricCategories(BaseMetricCategories):
    mean: AbsorptionMeanMetrics = Field(
        title="Mean",
        description="Mean metrics",
        json_schema_extra={"default_display": True},
    )


# Define a result detail, which in this case is an absorption result for a single letter.
@dataclass
class AbsorptionResultDetail(BaseResultDetail):

    first_letter: str = Field(title="First Letter", description="")

    @field_validator("first_letter")
    @classmethod
    def validate_single_letter(cls, value: str) -> str:
        if len(value) == 1 and value.isalpha():
            return value
        raise ValueError("First letter must be a single letter")

    absorption_rate: float = Field(title="Absorption Rate", description="")
    num_absorption: int = Field(title="Num Absorption", description="")
    num_probe_true_positives: int = Field(
        title="Num Probe True Positives", description=""
    )
    num_split_features: int = Field(title="Num Split Features", description="")


# Define the eval output, which includes the eval config, metrics, and result details.
# The title will end up being the title of the eval in the UI.
@dataclass(config=ConfigDict(title="Feature Absorption Evaluation - First Letter"))
class AbsorptionEvalOutput(
    BaseEvalOutput[
        AbsorptionEvalConfig, AbsorptionMetricCategories, AbsorptionResultDetail
    ]
):
    # This will end up being the description of the eval in the UI.
    """
    The output of a feature absorption evaluation looking at the first letter.
    """

    eval_config: AbsorptionEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: AbsorptionMetricCategories
    eval_result_details: list[AbsorptionResultDetail] = Field(
        default_factory=list,
        title="Per-Letter Absorption Results",
        description="Each object is a stat on the first letter of the absorption.",
    )
    eval_type_id: str = Field(
        default=EVAL_TYPE_ID_ABSORPTION,
        title="Eval Type ID",
        description="The type of the evaluation",
    )

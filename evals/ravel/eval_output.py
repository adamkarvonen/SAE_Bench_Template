from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, Field
from evals.base_eval_output import (
    DEFAULT_DISPLAY,
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    BaseResultDetail,
)
from evals.ravel.eval_config import RAVELEvalConfig

EVAL_TYPE_ID_RAVEL = "RAVEL"


@dataclass
class RAVELMetricResults(BaseMetrics):
    disentanglement: float = Field(
        title="Disentanglement",
        description="Mean of cause and isolation scores from RAVEL evaluation.",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    cause_score: float = Field(
        title="Mean Cause Score",
        description="Cause score: Patching attribute-related SAE latents. High cause accuracy indicates that the SAE latents are related to the attribute.",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    isolation_score: float = Field(
        title="Mean Isolation Score",
        description="Isolation score: Patching SAE latents related to another attribute. High isolation accuracy indicates that latents related to another attribute are not related to this attribute.",
        json_schema_extra=DEFAULT_DISPLAY,
    )


@dataclass
class RAVELMetricCategories(BaseMetricCategories):
    llm: RAVELMetricResults = Field(
        title="RAVEL",
        description="RAVEL metrics",
        json_schema_extra=DEFAULT_DISPLAY,
    )
# TODO revisit category split after inspecting results


@dataclass
class RAVELResultDetail(BaseResultDetail):
    dataset_name: str = Field(
        title="Dataset Name",
        description="Dataset name",
    )
    disentanglement: float = Field(
        title="Disentanglement",
        description="Mean of cause and isolation scores from RAVEL evaluation.",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    entity_class: str = Field(
        title="Entity Class",
        description="Entity Class",
    )
    attribute_A_name: str = Field(
        title="Attribute A Name",
        description="Attribute A name",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    attribute_B_name: str = Field(
        title="Attribute B Name",
        description="Attribute B name",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    cause_A_score: float = Field(
        title="Cause Score",
        description="Cause score: Patching attribute-related SAE latents. High cause accuracy indicates that the SAE latents are related to the attribute.",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    cause_B_score: float = Field(
        title="Cause Score",
        description="Cause score: Patching attribute-related SAE latents. High cause accuracy indicates that the SAE latents are related to the attribute.",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    isolation_AtoB_score: float = Field(
        title="Isolation Score",
        description="Isolation score: Patching SAE latents related to another attribute. High isolation accuracy indicates that latents related to another attribute are not related to this attribute.",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    isolation_BtoA_score: float = Field(
        title="Isolation Score AtoB",
        description="Isolation score: Patching SAE latents related to another attribute. High isolation accuracy indicates that latents related to another attribute are not related to this attribute.",
        json_schema_extra=DEFAULT_DISPLAY,
    )


@dataclass(config=ConfigDict(title="Sparse Probing"))
class RAVELEvalOutput(
    BaseEvalOutput[
        RAVELEvalConfig,
        RAVELMetricCategories,
        RAVELResultDetail
    ]
):
    # This will end up being the description of the eval in the UI.
    """
    An evaluation using SAEs to probe for supervised concepts in LLMs. We use sparse probing with the top K SAE latents and probe for over 30 different classes across 5 datasets.
    """

    eval_config: SparseProbingEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: SparseProbingMetricCategories
    eval_result_details: list[SparseProbingResultDetail] = Field(
        default_factory=list,
        title="Per-Dataset Sparse Probing Results",
        description="Each object is a stat on the sparse probing results for a dataset.",
    )
    eval_type_id: str = Field(default=EVAL_TYPE_ID_SPARSE_PROBING)
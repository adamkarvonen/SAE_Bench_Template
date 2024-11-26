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
    disentanglement_score: float = Field(
        title="Disentanglement Score",
        description="Mean of cause and isolation scores across RAVEL datasets.",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    cause_score: float = Field(
        title="Cause Score",
        description="Cause score: Patching attribute-related SAE latents. High cause accuracy indicates that the SAE latents are related to the attribute.",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    isolation_score: float = Field(
        title="Isolation Score",
        description="Isolation score: Patching SAE latents related to another attribute. High isolation accuracy indicates that latents related to another attribute are not related to this attribute.",
        json_schema_extra=DEFAULT_DISPLAY,
    )


@dataclass
class RAVELMetricCategories(BaseMetricCategories):
    sae: RAVELMetricResults = Field(
        title="RAVEL",
        description="RAVEL metrics",
        json_schema_extra=DEFAULT_DISPLAY,
    )
# TODO revisit category split after inspecting results


@dataclass
class RAVELResultDetail(BaseResultDetail):
    entity_class: str = Field(
        title="Entity Class",
        description="Entity Class",
    )
    attribute_classes: list[str] = Field(
        title="Attribute Classes",
        description="Attribute Classes",
    )
    latent_selection_thresholds: list[float] = Field(
        title="Latent Selection Threshold",
        description="Latent Selection Threshold",
    )
    cause_scores: dict[float, list[float]] = Field(
        title="Cause Scores",
        description="1D row of cause scores, ordered by attribute_classes. Cause score: Patching attribute-related SAE latents. High cause accuracy indicates that the SAE latents are related to the attribute.",
    )
    isolation_scores: dict[float, list[list[float]]] = Field(
        title="Isolation Scores",
        description="2D row of isolation scores, ordered by attribute_classes(base) x attribute_classes(source). Isolation score: Patching SAE latents related to another attribute. High isolation accuracy indicates that latents related to another attribute are not related to this attribute.",
    )
    mean_disentanglement: dict[float, float] = Field(
        title="Mean Disentanglement",
        description="Mean of cause and disentanglement with balanced weights: Mean(mean(cause_scores), mean(isolation_scores)).",
    )


@dataclass(config=ConfigDict(title="RAVEL"))
class RAVELEvalOutput(
    BaseEvalOutput[
        RAVELEvalConfig,
        RAVELMetricCategories,
        RAVELResultDetail
    ]
):
    # This will end up being the description of the eval in the UI.
    """
    An evaluation using SAEs for targeted modification of language model output. We leverage the RAVEL dataset of entity-attribute pairs. After filtering for known pairs, we identify attribute-related SAE latents and deterimine the effect on model predictions with activation patching experiments.
    """

    eval_config: RAVELEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: RAVELMetricCategories
    eval_result_details: list[RAVELResultDetail] = Field(
        default_factory=list,
        title="Per-Entity-Dataset RAVEL Results",
        description="Each object is a stat on the RAVEL results for an entity dataset.",
    )
    eval_type_id: str = Field(default=EVAL_TYPE_ID_RAVEL)
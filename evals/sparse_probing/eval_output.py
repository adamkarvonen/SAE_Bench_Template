from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, Field
from evals.base_eval_output import (
    DEFAULT_DISPLAY,
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    BaseResultDetail,
)
from evals.sparse_probing.eval_config import SparseProbingEvalConfig

EVAL_TYPE_ID_SPARSE_PROBING = "sparse_probing"


@dataclass
class SparseProbingLlmMetrics(BaseMetrics):
    llm_test_accuracy: float = Field(
        title="LLM Test Accuracy",
        description="Linear probe accuracy when training on the full LLM residual stream",
    )
    llm_top_1_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 1 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 1 residual stream channel test accuracy",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    llm_top_2_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 2 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 2 residual stream channels test accuracy",
    )
    llm_top_5_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 5 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 5 residual stream channels test accuracy",
    )
    llm_top_10_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 10 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 10 residual stream channels",
    )
    llm_top_20_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 20 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 20 residual stream channels",
    )
    llm_top_50_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 50 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 50 residual stream channels",
    )
    llm_top_100_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 100 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 100 residual stream channels",
    )


@dataclass
class SparseProbingSaeMetrics(BaseMetrics):
    sae_test_accuracy: float | None = Field(
        default=None,
        title="SAE Test Accuracy",
        description="Linear probe accuracy when trained on all SAE latents",
    )
    sae_top_1_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 1 Test Accuracy",
        description="Linear probe accuracy when trained on the top 1 SAE latents",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    sae_top_2_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 2 Test Accuracy",
        description="Linear probe accuracy when trained on the top 2 SAE latents",
    )
    sae_top_5_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 5 Test Accuracy",
        description="Linear probe accuracy when trained on the top 5 SAE latents",
    )
    sae_top_10_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 10 Test Accuracy",
        description="Linear probe accuracy when trained on the top 10 SAE latents",
    )
    sae_top_20_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 20 Test Accuracy",
        description="Linear probe accuracy when trained on the top 20 SAE latents",
    )
    sae_top_50_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 50 Test Accuracy",
        description="Linear probe accuracy when trained on the top 50 SAE latents",
    )
    sae_top_100_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 100 Test Accuracy",
        description="Linear probe accuracy when trained on the top 100 SAE latents",
    )


@dataclass
class SparseProbingMetricCategories(BaseMetricCategories):
    llm: SparseProbingLlmMetrics = Field(
        title="LLM",
        description="LLM metrics",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    sae: SparseProbingSaeMetrics = Field(
        title="SAE",
        description="SAE metrics",
        json_schema_extra=DEFAULT_DISPLAY,
    )


@dataclass
class SparseProbingResultDetail(BaseResultDetail):
    dataset_name: str = Field(
        title="Dataset Name",
        description="Dataset name",
    )

    llm_test_accuracy: float = Field(
        title="LLM Test Accuracy",
        description="Linear probe accuracy when trained on all LLM residual stream channels",
    )
    llm_top_1_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 1 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 1 residual stream channels",
    )
    llm_top_2_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 2 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 2 residual stream channels",
    )
    llm_top_5_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 5 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 5 residual stream channels",
    )
    llm_top_10_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 10 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 10 residual stream channels",
    )
    llm_top_20_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 20 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 20 residual stream channels",
    )
    llm_top_50_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 50 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 50 residual stream channels",
    )
    llm_top_100_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 100 Test Accuracy",
        description="Linear probe accuracy when trained on the LLM top 100 residual stream channels",
    )

    sae_test_accuracy: float | None = Field(
        default=None,
        title="SAE Test Accuracy",
        description="Linear probe accuracy when trained on all SAE latents",
    )
    sae_top_1_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 1 Test Accuracy",
        description="Linear probe accuracy when trained on the top 1 SAE latents",
    )
    sae_top_2_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 2 Test Accuracy",
        description="Linear probe accuracy when trained on the top 2 SAE latents",
    )
    sae_top_5_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 5 Test Accuracy",
        description="Linear probe accuracy when trained on the top 5 SAE latents",
    )
    sae_top_10_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 10 Test Accuracy",
        description="Linear probe accuracy when trained on the top 10 SAE latents",
    )
    sae_top_20_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 20 Test Accuracy",
        description="Linear probe accuracy when trained on the top 20 SAE latents",
    )
    sae_top_50_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 50 Test Accuracy",
        description="Linear probe accuracy when trained on the top 50 SAE latents",
    )
    sae_top_100_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 100 Test Accuracy",
        description="Linear probe accuracy when trained on the top 100 SAE latents",
    )


@dataclass(config=ConfigDict(title="Sparse Probing"))
class SparseProbingEvalOutput(
    BaseEvalOutput[
        SparseProbingEvalConfig,
        SparseProbingMetricCategories,
        SparseProbingResultDetail,
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

from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, Field
from evals.base_eval_output import (
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
        description="LLM test accuracy",
        json_schema_extra={"default_display": True},
    )
    llm_top_1_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 1 Test Accuracy",
        description="LLM top 1 test accuracy",
    )
    llm_top_2_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 2 Test Accuracy",
        description="LLM top 2 test accuracy",
    )
    llm_top_5_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 5 Test Accuracy",
        description="LLM top 5 test accuracy",
    )
    llm_top_10_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 10 Test Accuracy",
        description="LLM top 10 test accuracy",
    )
    llm_top_20_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 20 Test Accuracy",
        description="LLM top 20 test accuracy",
    )
    llm_top_50_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 50 Test Accuracy",
        description="LLM top 50 test accuracy",
    )
    llm_top_100_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 100 Test Accuracy",
        description="LLM top 100 test accuracy",
    )


@dataclass
class SparseProbingSaeMetrics(BaseMetrics):
    sae_test_accuracy: float | None = Field(
        default=None,
        title="SAE Test Accuracy",
        description="SAE test accuracy",
        json_schema_extra={"default_display": True},
    )
    sae_top_1_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 1 Test Accuracy",
        description="SAE top 1 test accuracy",
    )
    sae_top_2_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 2 Test Accuracy",
        description="SAE top 2 test accuracy",
    )
    sae_top_5_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 5 Test Accuracy",
        description="SAE top 5 test accuracy",
    )
    sae_top_10_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 10 Test Accuracy",
        description="SAE top 10 test accuracy",
    )
    sae_top_20_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 20 Test Accuracy",
        description="SAE top 20 test accuracy",
    )
    sae_top_50_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 50 Test Accuracy",
        description="SAE top 50 test accuracy",
    )
    sae_top_100_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 100 Test Accuracy",
        description="SAE top 100 test accuracy",
    )


@dataclass
class SparseProbingMetricCategories(BaseMetricCategories):
    llm: SparseProbingLlmMetrics = Field(
        title="LLM",
        description="LLM metrics",
        json_schema_extra={"default_display": True},
    )
    sae: SparseProbingSaeMetrics = Field(
        title="SAE",
        description="SAE metrics",
        json_schema_extra={"default_display": True},
    )


@dataclass
class SparseProbingResultDetail(BaseResultDetail):

    dataset_name: str = Field(
        title="Dataset Name",
        description="Dataset name",
    )

    llm_test_accuracy: float = Field(
        title="LLM Test Accuracy",
        description="LLM test accuracy",
    )
    llm_top_1_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 1 Test Accuracy",
        description="LLM top 1 test accuracy",
    )
    llm_top_2_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 2 Test Accuracy",
        description="LLM top 2 test accuracy",
    )
    llm_top_5_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 5 Test Accuracy",
        description="LLM top 5 test accuracy",
    )
    llm_top_10_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 10 Test Accuracy",
        description="LLM top 10 test accuracy",
    )
    llm_top_20_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 20 Test Accuracy",
        description="LLM top 20 test accuracy",
    )
    llm_top_50_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 50 Test Accuracy",
        description="LLM top 50 test accuracy",
    )
    llm_top_100_test_accuracy: float | None = Field(
        default=None,
        title="LLM Top 100 Test Accuracy",
        description="LLM top 100 test accuracy",
    )

    sae_test_accuracy: float | None = Field(
        default=None,
        title="SAE Test Accuracy",
        description="SAE test accuracy",
    )
    sae_top_1_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 1 Test Accuracy",
        description="SAE top 1 test accuracy",
    )
    sae_top_2_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 2 Test Accuracy",
        description="SAE top 2 test accuracy",
    )
    sae_top_5_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 5 Test Accuracy",
        description="SAE top 5 test accuracy",
    )
    sae_top_10_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 10 Test Accuracy",
        description="SAE top 10 test accuracy",
    )
    sae_top_20_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 20 Test Accuracy",
        description="SAE top 20 test accuracy",
    )
    sae_top_50_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 50 Test Accuracy",
        description="SAE top 50 test accuracy",
    )
    sae_top_100_test_accuracy: float | None = Field(
        default=None,
        title="SAE Top 100 Test Accuracy",
        description="SAE top 100 test accuracy",
    )


@dataclass(config=ConfigDict(title="Sparse Probing Evaluation"))
class SparseProbingEvalOutput(
    BaseEvalOutput[
        SparseProbingEvalConfig,
        SparseProbingMetricCategories,
        SparseProbingResultDetail,
    ]
):
    # This will end up being the description of the eval in the UI.
    """
    The output of a feature absorption evaluation looking at the first letter.
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

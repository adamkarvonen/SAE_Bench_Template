from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, Field
from evals.base_eval_output import (
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    BaseResultDetail,
)
from evals.shift_and_tpp.eval_config import ShiftAndTppEvalConfig

# ========= SHIFT Output


@dataclass
class ShiftUncategorizedMetrics(BaseMetrics):
    scr_dir1_threshold_10: float = Field(
        title="SCR Dir 10",
        description="SCR Dir 10",
    )
    scr_metric_threshold_10: float = Field(
        title="SCR Metric 10",
        description="SCR Metric 10",
    )
    scr_dir2_threshold_10: float = Field(
        title="SCR Dir2 10",
        description="SCR Dir2 10",
    )


@dataclass
class ShiftMetricCategories(BaseMetricCategories):
    uncategorized: ShiftUncategorizedMetrics = Field(
        title="Uncategorized",
        description="Uncategorized metrics",
    )


@dataclass
class ShiftResultDetail(BaseResultDetail):
    dataset_name: str = Field(title="Dataset Name", description="")
    scr_dir1_threshold_10: float = Field(
        title="SCR Dir 10",
        description="SCR Dir 10",
    )
    scr_metric_threshold_10: float = Field(
        title="SCR Metric 10",
        description="SCR Metric 10",
    )
    scr_dir2_threshold_10: float = Field(
        title="SCR Dir2 10",
        description="SCR Dir2 10",
    )


@dataclass(config=ConfigDict(title="SHIFT Evaluation"))
class ShiftEvalOutput(
    BaseEvalOutput[ShiftAndTppEvalConfig, ShiftMetricCategories, ShiftResultDetail]
):
    """
    The output of a SHIFT evaluation.
    """

    eval_config: ShiftAndTppEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: ShiftMetricCategories
    eval_result_details: list[ShiftResultDetail] = Field(
        default_factory=list,
        title="Per-Dataset SHIFT Results",
        description="Each object is a stat on the SHIFT results for a single dataset.",
    )
    eval_type_id: str = Field(
        default="shift",
        title="Eval Type ID",
        description="The type of the evaluation",
    )


# ========= TPP Output


@dataclass
class TppUncategorizedMetrics(BaseMetrics):
    tpp_threshold_10_total_metric: float = Field(
        title="TPP Threshold 10 Total Metric",
        description="TPP Threshold 10 Total Metric",
    )
    tpp_threshold_10_intended_diff_only: float = Field(
        title="TPP Threshold 10 Intended Diff Only",
        description="TPP Threshold 10 Intended Diff Only",
    )
    tpp_threshold_10_unintended_diff_only: float = Field(
        title="TPP Threshold 10 Unintended Diff Only",
        description="TPP Threshold 10 Unintended Diff Only",
    )


@dataclass
class TppMetricCategories(BaseMetricCategories):
    uncategorized: TppUncategorizedMetrics = Field(
        title="Uncategorized",
        description="Uncategorized metrics",
    )


@dataclass
class TppResultDetail(BaseResultDetail):
    dataset_name: str = Field(title="Dataset Name", description="")
    tpp_threshold_10_total_metric: float = Field(
        title="TPP Threshold 10 Total Metric",
        description="TPP Threshold 10 Total Metric",
    )
    tpp_threshold_10_intended_diff_only: float = Field(
        title="TPP Threshold 10 Intended Diff Only",
        description="TPP Threshold 10 Intended Diff Only",
    )
    tpp_threshold_10_unintended_diff_only: float = Field(
        title="TPP Threshold 10 Unintended Diff Only",
        description="TPP Threshold 10 Unintended Diff Only",
    )


@dataclass(config=ConfigDict(title="SHIFT Evaluation"))
class TppEvalOutput(
    BaseEvalOutput[ShiftAndTppEvalConfig, TppMetricCategories, TppResultDetail]
):
    """
    The output of a SHIFT evaluation.
    """

    eval_config: ShiftAndTppEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: TppMetricCategories
    eval_result_details: list[TppResultDetail] = Field(
        default_factory=list,
        title="Per-Dataset TPP Results",
        description="Each object is a stat on the TPP results for a single dataset.",
    )
    eval_type_id: str = Field(
        default="tpp",
        title="Eval Type ID",
        description="The type of the evaluation",
    )

from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, Field
from evals.base_eval_output import (
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    BaseResultDetail,
)
from evals.shift_and_tpp.eval_config import ShiftAndTppEvalConfig

EVAL_TYPE_ID_SHIFT = "scr"
EVAL_TYPE_ID_TPP = "tpp"

# ========= SHIFT Output


@dataclass
class ShiftUncategorizedMetrics(BaseMetrics):
    scr_dir1_threshold_2: float | None = Field(
        None,
        title="SCR Dir 2",
        description="SCR Dir 2",
    )
    scr_metric_threshold_2: float | None = Field(
        None,
        title="SCR Metric 2",
        description="SCR Metric 2",
    )
    scr_dir2_threshold_2: float | None = Field(
        None,
        title="SCR Dir2 2",
        description="SCR Dir2 2",
    )
    scr_dir1_threshold_5: float | None = Field(
        None,
        title="SCR Dir 5",
        description="SCR Dir 5",
    )
    scr_metric_threshold_5: float | None = Field(
        None,
        title="SCR Metric 5",
        description="SCR Metric 5",
    )
    scr_dir2_threshold_5: float | None = Field(
        None,
        title="SCR Dir2 5",
        description="SCR Dir2 5",
    )
    scr_dir1_threshold_10: float | None = Field(
        None,
        title="SCR Dir 10",
        description="SCR Dir 10",
        json_schema_extra={"default_display": True},
    )
    scr_metric_threshold_10: float | None = Field(
        None,
        title="SCR Metric 10",
        description="SCR Metric 10",
        json_schema_extra={"default_display": True},
    )
    scr_dir2_threshold_10: float | None = Field(
        None,
        title="SCR Dir2 10",
        description="SCR Dir2 10",
        json_schema_extra={"default_display": True},
    )
    scr_dir1_threshold_20: float | None = Field(
        None,
        title="SCR Dir 20",
        description="SCR Dir 20",
    )
    scr_metric_threshold_20: float | None = Field(
        None,
        title="SCR Metric 20",
        description="SCR Metric 20",
    )
    scr_dir2_threshold_20: float | None = Field(
        None,
        title="SCR Dir2 20",
        description="SCR Dir2 20",
    )
    scr_dir1_threshold_50: float | None = Field(
        None,
        title="SCR Dir 50",
        description="SCR Dir 50",
    )
    scr_metric_threshold_50: float | None = Field(
        None,
        title="SCR Metric 50",
        description="SCR Metric 50",
    )
    scr_dir2_threshold_50: float | None = Field(
        None,
        title="SCR Dir2 50",
        description="SCR Dir2 50",
    )
    scr_dir1_threshold_100: float | None = Field(
        None,
        title="SCR Dir 100",
        description="SCR Dir 100",
    )
    scr_metric_threshold_100: float | None = Field(
        None,
        title="SCR Metric 100",
        description="SCR Metric 100",
    )
    scr_dir2_threshold_100: float | None = Field(
        None,
        title="SCR Dir2 100",
        description="SCR Dir2 100",
    )
    scr_dir1_threshold_500: float | None = Field(
        None,
        title="SCR Dir 500",
        description="SCR Dir 500",
    )
    scr_metric_threshold_500: float | None = Field(
        None,
        title="SCR Metric 500",
        description="SCR Metric 500",
    )
    scr_dir2_threshold_500: float | None = Field(
        None,
        title="SCR Dir2 500",
        description="SCR Dir2 500",
    )


@dataclass
class ShiftMetricCategories(BaseMetricCategories):
    uncategorized: ShiftUncategorizedMetrics = Field(
        title="Uncategorized",
        description="Uncategorized metrics",
        json_schema_extra={"default_display": True},
    )


@dataclass
class ShiftResultDetail(BaseResultDetail):
    dataset_name: str = Field(title="Dataset Name", description="")

    scr_dir1_threshold_2: float | None = Field(
        None,
        title="SCR Dir 2",
        description="SCR Dir 2",
    )
    scr_metric_threshold_2: float | None = Field(
        None,
        title="SCR Metric 2",
        description="SCR Metric 2",
    )
    scr_dir2_threshold_2: float | None = Field(
        None,
        title="SCR Dir2 2",
        description="SCR Dir2 2",
    )
    scr_dir1_threshold_5: float | None = Field(
        None,
        title="SCR Dir 5",
        description="SCR Dir 5",
    )
    scr_metric_threshold_5: float | None = Field(
        None,
        title="SCR Metric 5",
        description="SCR Metric 5",
    )
    scr_dir2_threshold_5: float | None = Field(
        None,
        title="SCR Dir2 5",
        description="SCR Dir2 5",
    )
    scr_dir1_threshold_10: float | None = Field(
        None,
        title="SCR Dir 10",
        description="SCR Dir 10",
    )
    scr_metric_threshold_10: float | None = Field(
        None,
        title="SCR Metric 10",
        description="SCR Metric 10",
    )
    scr_dir2_threshold_10: float | None = Field(
        None,
        title="SCR Dir2 10",
        description="SCR Dir2 10",
    )
    scr_dir1_threshold_20: float | None = Field(
        None,
        title="SCR Dir 20",
        description="SCR Dir 20",
    )
    scr_metric_threshold_20: float | None = Field(
        None,
        title="SCR Metric 20",
        description="SCR Metric 20",
    )
    scr_dir2_threshold_20: float | None = Field(
        None,
        title="SCR Dir2 20",
        description="SCR Dir2 20",
    )
    scr_dir1_threshold_50: float | None = Field(
        None,
        title="SCR Dir 50",
        description="SCR Dir 50",
    )
    scr_metric_threshold_50: float | None = Field(
        None,
        title="SCR Metric 50",
        description="SCR Metric 50",
    )
    scr_dir2_threshold_50: float | None = Field(
        None,
        title="SCR Dir2 50",
        description="SCR Dir2 50",
    )
    scr_dir1_threshold_100: float | None = Field(
        None,
        title="SCR Dir 100",
        description="SCR Dir 100",
    )
    scr_metric_threshold_100: float | None = Field(
        None,
        title="SCR Metric 100",
        description="SCR Metric 100",
    )
    scr_dir2_threshold_100: float | None = Field(
        None,
        title="SCR Dir2 100",
        description="SCR Dir2 100",
    )
    scr_dir1_threshold_500: float | None = Field(
        None,
        title="SCR Dir 500",
        description="SCR Dir 500",
    )
    scr_metric_threshold_500: float | None = Field(
        None,
        title="SCR Metric 500",
        description="SCR Metric 500",
    )
    scr_dir2_threshold_500: float | None = Field(
        None,
        title="SCR Dir2 500",
        description="SCR Dir2 500",
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
        default=EVAL_TYPE_ID_SHIFT,
        title="Eval Type ID",
        description="The type of the evaluation",
    )


# ========= TPP Output


@dataclass
class TppUncategorizedMetrics(BaseMetrics):
    tpp_threshold_2_total_metric: float | None = Field(
        None,
        title="TPP Threshold 2 Total Metric",
        description="TPP Threshold 2 Total Metric",
    )
    tpp_threshold_2_intended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 2 Intended Diff Only",
        description="TPP Threshold 2 Intended Diff Only",
    )
    tpp_threshold_2_unintended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 2 Unintended Diff Only",
        description="TPP Threshold 2 Unintended Diff Only",
    )
    tpp_threshold_5_total_metric: float | None = Field(
        None,
        title="TPP Threshold 5 Total Metric",
        description="TPP Threshold 5 Total Metric",
    )
    tpp_threshold_5_intended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 5 Intended Diff Only",
        description="TPP Threshold 5 Intended Diff Only",
    )
    tpp_threshold_5_unintended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 5 Unintended Diff Only",
        description="TPP Threshold 5 Unintended Diff Only",
    )
    tpp_threshold_10_total_metric: float | None = Field(
        None,
        title="TPP Threshold 10 Total Metric",
        description="TPP Threshold 10 Total Metric",
        json_schema_extra={"default_display": True},
    )
    tpp_threshold_10_intended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 10 Intended Diff Only",
        description="TPP Threshold 10 Intended Diff Only",
        json_schema_extra={"default_display": True},
    )
    tpp_threshold_10_unintended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 10 Unintended Diff Only",
        description="TPP Threshold 10 Unintended Diff Only",
        json_schema_extra={"default_display": True},
    )
    tpp_threshold_20_total_metric: float | None = Field(
        None,
        title="TPP Threshold 20 Total Metric",
        description="TPP Threshold 20 Total Metric",
    )
    tpp_threshold_20_intended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 20 Intended Diff Only",
        description="TPP Threshold 20 Intended Diff Only",
    )
    tpp_threshold_20_unintended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 20 Unintended Diff Only",
        description="TPP Threshold 20 Unintended Diff Only",
    )
    tpp_threshold_50_total_metric: float | None = Field(
        None,
        title="TPP Threshold 50 Total Metric",
        description="TPP Threshold 50 Total Metric",
    )
    tpp_threshold_50_intended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 50 Intended Diff Only",
        description="TPP Threshold 50 Intended Diff Only",
    )
    tpp_threshold_50_unintended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 50 Unintended Diff Only",
        description="TPP Threshold 50 Unintended Diff Only",
    )
    tpp_threshold_100_total_metric: float | None = Field(
        None,
        title="TPP Threshold 100 Total Metric",
        description="TPP Threshold 100 Total Metric",
    )
    tpp_threshold_100_intended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 100 Intended Diff Only",
        description="TPP Threshold 100 Intended Diff Only",
    )
    tpp_threshold_100_unintended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 100 Unintended Diff Only",
        description="TPP Threshold 100 Unintended Diff Only",
    )
    tpp_threshold_500_total_metric: float | None = Field(
        None,
        title="TPP Threshold 500 Total Metric",
        description="TPP Threshold 500 Total Metric",
    )
    tpp_threshold_500_intended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 500 Intended Diff Only",
        description="TPP Threshold 500 Intended Diff Only",
    )
    tpp_threshold_500_unintended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 500 Unintended Diff Only",
        description="TPP Threshold 500 Unintended Diff Only",
    )


@dataclass
class TppMetricCategories(BaseMetricCategories):
    uncategorized: TppUncategorizedMetrics = Field(
        title="Uncategorized",
        description="Uncategorized metrics",
        json_schema_extra={"default_display": True},
    )


@dataclass
class TppResultDetail(BaseResultDetail):
    dataset_name: str = Field(title="Dataset Name", description="")

    tpp_threshold_2_total_metric: float | None = Field(
        None,
        title="TPP Threshold 2 Total Metric",
        description="TPP Threshold 2 Total Metric",
    )
    tpp_threshold_2_intended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 2 Intended Diff Only",
        description="TPP Threshold 2 Intended Diff Only",
    )
    tpp_threshold_2_unintended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 2 Unintended Diff Only",
        description="TPP Threshold 2 Unintended Diff Only",
    )
    tpp_threshold_5_total_metric: float | None = Field(
        None,
        title="TPP Threshold 5 Total Metric",
        description="TPP Threshold 5 Total Metric",
    )
    tpp_threshold_5_intended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 5 Intended Diff Only",
        description="TPP Threshold 5 Intended Diff Only",
    )
    tpp_threshold_5_unintended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 5 Unintended Diff Only",
        description="TPP Threshold 5 Unintended Diff Only",
    )
    tpp_threshold_10_total_metric: float | None = Field(
        None,
        title="TPP Threshold 10 Total Metric",
        description="TPP Threshold 10 Total Metric",
    )
    tpp_threshold_10_intended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 10 Intended Diff Only",
        description="TPP Threshold 10 Intended Diff Only",
    )
    tpp_threshold_10_unintended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 10 Unintended Diff Only",
        description="TPP Threshold 10 Unintended Diff Only",
    )
    tpp_threshold_20_total_metric: float | None = Field(
        None,
        title="TPP Threshold 20 Total Metric",
        description="TPP Threshold 20 Total Metric",
    )
    tpp_threshold_20_intended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 20 Intended Diff Only",
        description="TPP Threshold 20 Intended Diff Only",
    )
    tpp_threshold_20_unintended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 20 Unintended Diff Only",
        description="TPP Threshold 20 Unintended Diff Only",
    )
    tpp_threshold_50_total_metric: float | None = Field(
        None,
        title="TPP Threshold 50 Total Metric",
        description="TPP Threshold 50 Total Metric",
    )
    tpp_threshold_50_intended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 50 Intended Diff Only",
        description="TPP Threshold 50 Intended Diff Only",
    )
    tpp_threshold_50_unintended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 50 Unintended Diff Only",
        description="TPP Threshold 50 Unintended Diff Only",
    )
    tpp_threshold_100_total_metric: float | None = Field(
        None,
        title="TPP Threshold 100 Total Metric",
        description="TPP Threshold 100 Total Metric",
    )
    tpp_threshold_100_intended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 100 Intended Diff Only",
        description="TPP Threshold 100 Intended Diff Only",
    )
    tpp_threshold_100_unintended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 100 Unintended Diff Only",
        description="TPP Threshold 100 Unintended Diff Only",
    )
    tpp_threshold_500_total_metric: float | None = Field(
        None,
        title="TPP Threshold 500 Total Metric",
        description="TPP Threshold 500 Total Metric",
    )
    tpp_threshold_500_intended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 500 Intended Diff Only",
        description="TPP Threshold 500 Intended Diff Only",
    )
    tpp_threshold_500_unintended_diff_only: float | None = Field(
        None,
        title="TPP Threshold 500 Unintended Diff Only",
        description="TPP Threshold 500 Unintended Diff Only",
    )


@dataclass(config=ConfigDict(title="TPP Evaluation"))
class TppEvalOutput(
    BaseEvalOutput[ShiftAndTppEvalConfig, TppMetricCategories, TppResultDetail]
):
    """
    The output of a TPP evaluation.
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
        default=EVAL_TYPE_ID_TPP,
        title="Eval Type ID",
        description="The type of the evaluation",
    )

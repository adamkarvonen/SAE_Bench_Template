from pydantic.dataclasses import dataclass
from pydantic import ConfigDict, Field
from evals.base_eval_output import (
    DEFAULT_DISPLAY,
    BaseEvalOutput,
    BaseMetricCategories,
    BaseMetrics,
    BaseResultDetail,
)
from evals.shift_and_tpp.eval_config import ShiftAndTppEvalConfig

EVAL_TYPE_ID_SHIFT = "shift"
EVAL_TYPE_ID_TPP = "tpp"


@dataclass
class ShiftMetrics(BaseMetrics):
    scr_dir1_threshold_2: float | None = Field(
        None,
        title="SCR Dir 1, Top 2 SAE latents",
        description="Ablating the top 2 gender latents to increase profession accuracy",
    )
    scr_metric_threshold_2: float | None = Field(
        None,
        title="SCR Metric, Top 2 SAE latents",
        description="SCR Metric (selecting dir1 if inital profession accuracy is lower than initial gender accuracy, else dir2) ablating the top 2 SAE latents",
    )
    scr_dir2_threshold_2: float | None = Field(
        None,
        title="SCR Dir 2, Top 2 SAE latents",
        description="Ablating the top 2 profession latents to increase gender accuracy",
    )
    scr_dir1_threshold_5: float | None = Field(
        None,
        title="SCR Dir 1, Top 5 SAE latents",
        description="Ablating the top 5 gender latents to increase profession accuracy",
    )
    scr_metric_threshold_5: float | None = Field(
        None,
        title="SCR Metric, Top 5 SAE latents",
        description="SCR Metric (selecting dir1 if inital profession accuracy is lower than initial gender accuracy, else dir2) ablating the top 5 SAE latents",
    )
    scr_dir2_threshold_5: float | None = Field(
        None,
        title="SCR Dir 2, Top 5 SAE latents",
        description="Ablating the top 5 profession latents to increase gender accuracy",
    )
    scr_dir1_threshold_10: float | None = Field(
        None,
        title="SCR Dir 1, Top 10 SAE latents",
        description="Ablating the top 10 gender latents to increase profession accuracy",
    )
    scr_metric_threshold_10: float | None = Field(
        None,
        title="SCR Metric, Top 10 SAE latents",
        description="SCR Metric (selecting dir1 if inital profession accuracy is lower than initial gender accuracy, else dir2) ablating the top 10 SAE latents",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    scr_dir2_threshold_10: float | None = Field(
        None,
        title="SCR Dir 2, Top 10 SAE latents",
        description="Ablating the top 10 profession latents to increase gender accuracy",
    )
    scr_dir1_threshold_20: float | None = Field(
        None,
        title="SCR Dir 1, Top 20 SAE latents",
        description="Ablating the top 20 gender latents to increase profession accuracy",
    )
    scr_metric_threshold_20: float | None = Field(
        None,
        title="SCR Metric, Top 20 SAE latents",
        description="SCR Metric (selecting dir1 if inital profession accuracy is lower than initial gender accuracy, else dir2) ablating the top 20 SAE latents",
    )
    scr_dir2_threshold_20: float | None = Field(
        None,
        title="SCR Dir 2, Top 20 SAE latents",
        description="Ablating the top 20 profession latents to increase gender accuracy",
    )
    scr_dir1_threshold_50: float | None = Field(
        None,
        title="SCR Dir 1, Top 50 SAE latents",
        description="Ablating the top 50 gender latents to increase profession accuracy",
    )
    scr_metric_threshold_50: float | None = Field(
        None,
        title="SCR Metric, Top 50 SAE latents",
        description="SCR Metric (selecting dir1 if inital profession accuracy is lower than initial gender accuracy, else dir2) ablating the top 50 SAE latents",
    )
    scr_dir2_threshold_50: float | None = Field(
        None,
        title="SCR Dir 2, Top 50 SAE latents",
        description="Ablating the top 50 profession latents to increase gender accuracy",
    )
    scr_dir1_threshold_100: float | None = Field(
        None,
        title="SCR Dir 1, Top 100 SAE latents",
        description="Ablating the top 100 gender latents to increase profession accuracy",
    )
    scr_metric_threshold_100: float | None = Field(
        None,
        title="SCR Metric, Top 100 SAE latents",
        description="SCR Metric (selecting dir1 if inital profession accuracy is lower than initial gender accuracy, else dir2) ablating the top 100 SAE latents",
    )
    scr_dir2_threshold_100: float | None = Field(
        None,
        title="SCR Dir 2, Top 100 SAE latents",
        description="Ablating the top 100 profession latents to increase gender accuracy",
    )
    scr_dir1_threshold_500: float | None = Field(
        None,
        title="SCR Dir 1, Top 500 SAE latents",
        description="Ablating the top 500 gender latents to increase profession accuracy",
    )
    scr_metric_threshold_500: float | None = Field(
        None,
        title="SCR Metric, Top 500 SAE latents",
        description="SCR Metric (selecting dir1 if inital profession accuracy is lower than initial gender accuracy, else dir2) ablating the top 500 SAE latents",
    )
    scr_dir2_threshold_500: float | None = Field(
        None,
        title="SCR Dir 2, Top 500 SAE latents",
        description="Ablating the top 500 profession latents to increase gender accuracy",
    )


@dataclass
class ShiftMetricCategories(BaseMetricCategories):
    shift_metrics: ShiftMetrics = Field(
        title="Shift Metrics",
        description="SHIFT SCR metrics, calculated for different numbers of ablated features. Also includes the results for both correlation removal directions.",
        json_schema_extra=DEFAULT_DISPLAY,
    )


@dataclass
class ShiftResultDetail(BaseResultDetail):
    dataset_name: str = Field(title="Dataset Name", description="")

    scr_dir1_threshold_2: float | None = Field(
        None,
        title="SCR Dir 1, Top 2 SAE latents",
        description="Ablating the top 2 gender latents to increase profession accuracy",
    )
    scr_metric_threshold_2: float | None = Field(
        None,
        title="SCR Metric, Top 2 SAE latents",
        description="SCR Metric (selecting dir1 if inital profession accuracy is lower than initial gender accuracy, else dir2) ablating the top 2 SAE latents",
    )
    scr_dir2_threshold_2: float | None = Field(
        None,
        title="SCR Dir 2, Top 2 SAE latents",
        description="Ablating the top 2 profession latents to increase gender accuracy",
    )
    scr_dir1_threshold_5: float | None = Field(
        None,
        title="SCR Dir 1, Top 5 SAE latents",
        description="Ablating the top 5 gender latents to increase profession accuracy",
    )
    scr_metric_threshold_5: float | None = Field(
        None,
        title="SCR Metric, Top 5 SAE latents",
        description="SCR Metric (selecting dir1 if inital profession accuracy is lower than initial gender accuracy, else dir2) ablating the top 5 SAE latents",
    )
    scr_dir2_threshold_5: float | None = Field(
        None,
        title="SCR Dir 2, Top 5 SAE latents",
        description="Ablating the top 5 profession latents to increase gender accuracy",
    )
    scr_dir1_threshold_10: float | None = Field(
        None,
        title="SCR Dir 1, Top 10 SAE latents",
        description="Ablating the top 10 gender latents to increase profession accuracy",
    )
    scr_metric_threshold_10: float | None = Field(
        None,
        title="SCR Metric, Top 10 SAE latents",
        description="SCR Metric (selecting dir1 if inital profession accuracy is lower than initial gender accuracy, else dir2) ablating the top 10 SAE latents",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    scr_dir2_threshold_10: float | None = Field(
        None,
        title="SCR Dir 2, Top 10 SAE latents",
        description="Ablating the top 10 profession latents to increase gender accuracy",
    )
    scr_dir1_threshold_20: float | None = Field(
        None,
        title="SCR Dir 1, Top 20 SAE latents",
        description="Ablating the top 20 gender latents to increase profession accuracy",
    )
    scr_metric_threshold_20: float | None = Field(
        None,
        title="SCR Metric, Top 20 SAE latents",
        description="SCR Metric (selecting dir1 if inital profession accuracy is lower than initial gender accuracy, else dir2) ablating the top 20 SAE latents",
    )
    scr_dir2_threshold_20: float | None = Field(
        None,
        title="SCR Dir 2, Top 20 SAE latents",
        description="Ablating the top 20 profession latents to increase gender accuracy",
    )
    scr_dir1_threshold_50: float | None = Field(
        None,
        title="SCR Dir 1, Top 50 SAE latents",
        description="Ablating the top 50 gender latents to increase profession accuracy",
    )
    scr_metric_threshold_50: float | None = Field(
        None,
        title="SCR Metric, Top 50 SAE latents",
        description="SCR Metric (selecting dir1 if inital profession accuracy is lower than initial gender accuracy, else dir2) ablating the top 50 SAE latents",
    )
    scr_dir2_threshold_50: float | None = Field(
        None,
        title="SCR Dir 2, Top 50 SAE latents",
        description="Ablating the top 50 profession latents to increase gender accuracy",
    )
    scr_dir1_threshold_100: float | None = Field(
        None,
        title="SCR Dir 1, Top 100 SAE latents",
        description="Ablating the top 100 gender latents to increase profession accuracy",
    )
    scr_metric_threshold_100: float | None = Field(
        None,
        title="SCR Metric, Top 100 SAE latents",
        description="SCR Metric (selecting dir1 if inital profession accuracy is lower than initial gender accuracy, else dir2) ablating the top 100 SAE latents",
    )
    scr_dir2_threshold_100: float | None = Field(
        None,
        title="SCR Dir 2, Top 100 SAE latents",
        description="Ablating the top 100 profession latents to increase gender accuracy",
    )
    scr_dir1_threshold_500: float | None = Field(
        None,
        title="SCR Dir 1, Top 500 SAE latents",
        description="Ablating the top 500 gender latents to increase profession accuracy",
    )
    scr_metric_threshold_500: float | None = Field(
        None,
        title="SCR Metric, Top 500 SAE latents",
        description="SCR Metric (selecting dir1 if inital profession accuracy is lower than initial gender accuracy, else dir2) ablating the top 500 SAE latents",
    )
    scr_dir2_threshold_500: float | None = Field(
        None,
        title="SCR Dir 2, Top 500 SAE latents",
        description="Ablating the top 500 profession latents to increase gender accuracy",
    )


@dataclass(config=ConfigDict(title="SHIFT"))
class ShiftEvalOutput(
    BaseEvalOutput[ShiftAndTppEvalConfig, ShiftMetricCategories, ShiftResultDetail]
):
    """
    The SHIFT Spurious Correlation Removal (SCR) evaluation ablates SAE latents to shift the bias of a biased linear probe. The methodology is from `Evaluating Sparse Autoencoders on Targeted Concept Removal Tasks`.
    """

    eval_config: ShiftAndTppEvalConfig
    eval_id: str
    datetime_epoch_millis: int
    eval_result_metrics: ShiftMetricCategories
    eval_result_details: list[ShiftResultDetail] = Field(
        default_factory=list,
        title="Per-Dataset SHIFT Spurious Correlation Removal (SCR) Results",
        description="Each object is a stat on the SHIFT SCR results for a single dataset.",
    )
    eval_type_id: str = Field(
        default=EVAL_TYPE_ID_SHIFT,
        title="Eval Type ID",
        description="The type of the evaluation",
    )


# ========= TPP Output


@dataclass
class TppMetrics(BaseMetrics):
    tpp_threshold_2_total_metric: float | None = Field(
        None,
        title="TPP Metric, Top 2 SAE latents",
        description="TPP metric when ablating the top 2 SAE latents",
    )
    tpp_threshold_2_intended_diff_only: float | None = Field(
        None,
        title="TPP Intended Class, Top 2 SAE latents",
        description="TPP decrease to the intended class only when ablating the top 2 SAE latents",
    )
    tpp_threshold_2_unintended_diff_only: float | None = Field(
        None,
        title="TPP Unintended Class, Top 2 SAE latents",
        description="TPP decrease to all unintended classes when ablating the top 2 SAE latents",
    )
    tpp_threshold_5_total_metric: float | None = Field(
        None,
        title="TPP Metric, Top 5 SAE latents",
        description="TPP metric when ablating the top 5 SAE latents",
    )
    tpp_threshold_5_intended_diff_only: float | None = Field(
        None,
        title="TPP Intended Class, Top 5 SAE latents",
        description="TPP decrease to the intended class only when ablating the top 5 SAE latents",
    )
    tpp_threshold_5_unintended_diff_only: float | None = Field(
        None,
        title="TPP Unintended Class, Top 5 SAE latents",
        description="TPP decrease to all unintended classes when ablating the top 5 SAE latents",
    )
    tpp_threshold_10_total_metric: float | None = Field(
        None,
        title="TPP Metric, Top 10 SAE latents",
        description="TPP metric when ablating the top 10 SAE latents",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    tpp_threshold_10_intended_diff_only: float | None = Field(
        None,
        title="TPP Intended Class, Top 10 SAE latents",
        description="TPP decrease to the intended class only when ablating the top 10 SAE latents",
    )
    tpp_threshold_10_unintended_diff_only: float | None = Field(
        None,
        title="TPP Unintended Class, Top 10 SAE latents",
        description="TPP decrease to all unintended classes when ablating the top 10 SAE latents",
    )
    tpp_threshold_20_total_metric: float | None = Field(
        None,
        title="TPP Metric, Top 20 SAE latents",
        description="TPP metric when ablating the top 20 SAE latents",
    )
    tpp_threshold_20_intended_diff_only: float | None = Field(
        None,
        title="TPP Intended Class, Top 20 SAE latents",
        description="TPP decrease to the intended class only when ablating the top 20 SAE latents",
    )
    tpp_threshold_20_unintended_diff_only: float | None = Field(
        None,
        title="TPP Unintended Class, Top 20 SAE latents",
        description="TPP decrease to all unintended classes when ablating the top 20 SAE latents",
    )
    tpp_threshold_50_total_metric: float | None = Field(
        None,
        title="TPP Metric, Top 50 SAE latents",
        description="TPP metric when ablating the top 50 SAE latents",
    )
    tpp_threshold_50_intended_diff_only: float | None = Field(
        None,
        title="TPP Intended Class, Top 50 SAE latents",
        description="TPP decrease to the intended class only when ablating the top 50 SAE latents",
    )
    tpp_threshold_50_unintended_diff_only: float | None = Field(
        None,
        title="TPP Unintended Class, Top 50 SAE latents",
        description="TPP decrease to all unintended classes when ablating the top 50 SAE latents",
    )
    tpp_threshold_100_total_metric: float | None = Field(
        None,
        title="TPP Metric, Top 100 SAE latents",
        description="TPP metric when ablating the top 100 SAE latents",
    )
    tpp_threshold_100_intended_diff_only: float | None = Field(
        None,
        title="TPP Intended Class, Top 100 SAE latents",
        description="TPP decrease to the intended class only when ablating the top 100 SAE latents",
    )
    tpp_threshold_100_unintended_diff_only: float | None = Field(
        None,
        title="TPP Unintended Class, Top 100 SAE latents",
        description="TPP decrease to all unintended classes when ablating the top 100 SAE latents",
    )
    tpp_threshold_500_total_metric: float | None = Field(
        None,
        title="TPP Metric, Top 500 SAE latents",
        description="TPP metric when ablating the top 500 SAE latents",
    )
    tpp_threshold_500_intended_diff_only: float | None = Field(
        None,
        title="TPP Intended Class, Top 500 SAE latents",
        description="TPP decrease to the intended class only when ablating the top 500 SAE latents",
    )
    tpp_threshold_500_unintended_diff_only: float | None = Field(
        None,
        title="TPP Unintended Class, Top 500 SAE latents",
        description="TPP decrease to all unintended classes when ablating the top 500 SAE latents",
    )


@dataclass
class TppMetricCategories(BaseMetricCategories):
    tpp_metrics: TppMetrics = Field(
        title="TPP Metrics",
        description="Targeted Probe Perturbation (TPP) results",
        json_schema_extra=DEFAULT_DISPLAY,
    )


@dataclass
class TppResultDetail(BaseResultDetail):
    dataset_name: str = Field(title="Dataset Name", description="")

    tpp_threshold_2_total_metric: float | None = Field(
        None,
        title="TPP Metric, Top 2 SAE latents",
        description="TPP metric when ablating the top 2 SAE latents",
    )
    tpp_threshold_2_intended_diff_only: float | None = Field(
        None,
        title="TPP Intended Class, Top 2 SAE latents",
        description="TPP decrease to the intended class only when ablating the top 2 SAE latents",
    )
    tpp_threshold_2_unintended_diff_only: float | None = Field(
        None,
        title="TPP Unintended Class, Top 2 SAE latents",
        description="TPP decrease to all unintended classes when ablating the top 2 SAE latents",
    )
    tpp_threshold_5_total_metric: float | None = Field(
        None,
        title="TPP Metric, Top 5 SAE latents",
        description="TPP metric when ablating the top 5 SAE latents",
    )
    tpp_threshold_5_intended_diff_only: float | None = Field(
        None,
        title="TPP Intended Class, Top 5 SAE latents",
        description="TPP decrease to the intended class only when ablating the top 5 SAE latents",
    )
    tpp_threshold_5_unintended_diff_only: float | None = Field(
        None,
        title="TPP Unintended Class, Top 5 SAE latents",
        description="TPP decrease to all unintended classes when ablating the top 5 SAE latents",
    )
    tpp_threshold_10_total_metric: float | None = Field(
        None,
        title="TPP Metric, Top 10 SAE latents",
        description="TPP metric when ablating the top 10 SAE latents",
        json_schema_extra=DEFAULT_DISPLAY,
    )
    tpp_threshold_10_intended_diff_only: float | None = Field(
        None,
        title="TPP Intended Class, Top 10 SAE latents",
        description="TPP decrease to the intended class only when ablating the top 10 SAE latents",
    )
    tpp_threshold_10_unintended_diff_only: float | None = Field(
        None,
        title="TPP Unintended Class, Top 10 SAE latents",
        description="TPP decrease to all unintended classes when ablating the top 10 SAE latents",
    )
    tpp_threshold_20_total_metric: float | None = Field(
        None,
        title="TPP Metric, Top 20 SAE latents",
        description="TPP metric when ablating the top 20 SAE latents",
    )
    tpp_threshold_20_intended_diff_only: float | None = Field(
        None,
        title="TPP Intended Class, Top 20 SAE latents",
        description="TPP decrease to the intended class only when ablating the top 20 SAE latents",
    )
    tpp_threshold_20_unintended_diff_only: float | None = Field(
        None,
        title="TPP Unintended Class, Top 20 SAE latents",
        description="TPP decrease to all unintended classes when ablating the top 20 SAE latents",
    )
    tpp_threshold_50_total_metric: float | None = Field(
        None,
        title="TPP Metric, Top 50 SAE latents",
        description="TPP metric when ablating the top 50 SAE latents",
    )
    tpp_threshold_50_intended_diff_only: float | None = Field(
        None,
        title="TPP Intended Class, Top 50 SAE latents",
        description="TPP decrease to the intended class only when ablating the top 50 SAE latents",
    )
    tpp_threshold_50_unintended_diff_only: float | None = Field(
        None,
        title="TPP Unintended Class, Top 50 SAE latents",
        description="TPP decrease to all unintended classes when ablating the top 50 SAE latents",
    )
    tpp_threshold_100_total_metric: float | None = Field(
        None,
        title="TPP Metric, Top 100 SAE latents",
        description="TPP metric when ablating the top 100 SAE latents",
    )
    tpp_threshold_100_intended_diff_only: float | None = Field(
        None,
        title="TPP Intended Class, Top 100 SAE latents",
        description="TPP decrease to the intended class only when ablating the top 100 SAE latents",
    )
    tpp_threshold_100_unintended_diff_only: float | None = Field(
        None,
        title="TPP Unintended Class, Top 100 SAE latents",
        description="TPP decrease to all unintended classes when ablating the top 100 SAE latents",
    )
    tpp_threshold_500_total_metric: float | None = Field(
        None,
        title="TPP Metric, Top 500 SAE latents",
        description="TPP metric when ablating the top 500 SAE latents",
    )
    tpp_threshold_500_intended_diff_only: float | None = Field(
        None,
        title="TPP Intended Class, Top 500 SAE latents",
        description="TPP decrease to the intended class only when ablating the top 500 SAE latents",
    )
    tpp_threshold_500_unintended_diff_only: float | None = Field(
        None,
        title="TPP Unintended Class, Top 500 SAE latents",
        description="TPP decrease to all unintended classes when ablating the top 500 SAE latents",
    )


@dataclass(config=ConfigDict(title="TPP"))
class TppEvalOutput(BaseEvalOutput[ShiftAndTppEvalConfig, TppMetricCategories, TppResultDetail]):
    """
    The Targeted Probe Pertubation (TPP) evaluation ablates a set of SAE latents to damage a single targeted linear probe. The methodology is from `Evaluating Sparse Autoencoders on Targeted Concept Removal Tasks`.
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

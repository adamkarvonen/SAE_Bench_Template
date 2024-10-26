from dataclasses import asdict
import json
from typing import Any, Generic, TypeVar
from pydantic.dataclasses import dataclass
from pydantic import Field, field_validator, model_validator


@dataclass
class BaseEvalConfig:
    """
    Configuration for the evaluation.
    """

    def __init__(self):
        if type(self) is BaseEvalConfig:
            raise ValueError(
                "BaseEvalConfig is an abstract class and cannot be instantiated directly."
            )


BaseEvalConfigType = TypeVar("BaseEvalConfigType", bound=BaseEvalConfig)


# Metrics for a single eval category
@dataclass
class BaseMetrics:

    def __init__(self):
        if type(self) is BaseMetrics:
            raise ValueError(
                "BaseMetrics is an abstract class and cannot be instantiated directly."
            )

    @model_validator(mode="after")
    @classmethod
    def validate_dict(cls, data):
        for _, value in asdict(data).items():
            if isinstance(value, dict):
                raise ValueError(
                    "Metrics is designed to be a flat, one-level structure, so dicts are not allowed."
                )
        return data


BaseMetricsType = TypeVar("BaseMetricsType", bound=BaseMetrics)


@dataclass
class BaseMetricCategories:

    def __init__(self):
        if type(self) is BaseMetricCategories:
            raise ValueError(
                "BaseMetricCategories is an abstract class and cannot be instantiated directly."
            )

    @model_validator(mode="after")
    @classmethod
    def validate_base_metric_type(cls, data):
        for field_name, field_value in data.__dict__.items():
            if not isinstance(field_value, BaseMetrics):
                raise ValueError(
                    f"Field '{field_name}' in {cls.__name__} must inherit from BaseMetrics."
                )

        return data


BaseMetricCategoriesType = TypeVar(
    "BaseMetricCategoriesType", bound=BaseMetricCategories
)


@dataclass
class BaseResultDetail:
    pass


BaseResultDetailType = TypeVar("BaseResultDetailType", bound=BaseResultDetail)


@dataclass
class BaseEvalOutput(
    Generic[BaseEvalConfigType, BaseMetricCategoriesType, BaseResultDetailType]
):

    def to_json(self, indent: int = 2) -> str:
        """
        Dump the BaseEvalOutput object to a JSON string.

        Args:
            indent (int): The number of spaces to use for indentation in the JSON output. Default is 2.

        Returns:
            str: A JSON string representation of the BaseEvalOutput object.
        """
        return json.dumps(asdict(self), indent=indent, default=str)

    def to_json_file(self, file_path: str, indent: int = 2) -> None:
        """
        Dump the BaseEvalOutput object to a JSON file.
        """
        with open(file_path, "w") as f:
            json.dump(asdict(self), f, indent=indent, default=str)

    eval_type_id: str = Field(
        title="Eval Type ID",
        description="The type of the evaluation",
    )

    eval_config: BaseEvalConfigType = Field(
        title="Eval Config Type", description="The configuration of the evaluation."
    )

    eval_id: str = Field(
        title="ID",
        description="A unique UUID identifying this specific eval run",
    )

    datetime_epoch_millis: int = Field(
        title="DateTime (epoch ms)",
        description="The datetime of the evaluation in epoch milliseconds",
    )

    @field_validator("datetime_epoch_millis")
    @classmethod
    def validate_unix_time(cls, value: int) -> int:
        if value < 0:
            raise ValueError("Unix time must be a non-negative integer")
        if value > 9999999999999:
            raise ValueError("Unix time is unreasonably large")
        return value

    eval_result_metrics: BaseMetricCategoriesType = Field(
        title="Result Metrics Categorized",
        description="The metrics of the evaluation, organized by category. Define your own categories and the metrics that go inside them.",
    )

    eval_result_details: list[BaseResultDetailType] = Field(
        None,
        title="Result Details",
        description="Optional. The details of the evaluation. A list of objects that stores nested or more detailed data, such as details about the absorption of each letter.",
    )

    sae_bench_commit_hash: str = Field(
        title="SAE Bench Commit Hash",
        description="The commit hash of the SAE Bench that ran the evaluation.",
    )

    sae_lens_id: str | None = Field(
        title="SAE Lens ID",
        description="The ID of the SAE in SAE Lens.",
    )

    sae_lens_release_id: str | None = Field(
        title="SAE Lens Release ID",
        description="The release ID of the SAE in SAE Lens.",
    )

    sae_lens_version: str | None = Field(
        title="SAE Lens Version",
        description="The version of SAE Lens that ran the evaluation.",
    )

    eval_result_unstructured: Any | None = Field(
        default_factory=None,
        title="Unstructured Results",
        description="Optional. Any additional outputs that don't fit into the structured eval_result_metrics or eval_result_details fields. Since these are unstructured, don't expect this to be easily renderable in UIs, or contain any titles or descriptions.",
    )

    def __init__(self, eval_config: BaseEvalConfigType):
        if type(self) is BaseEvalOutput:
            raise ValueError(
                "BaseEvalOutput is an abstract class and cannot be instantiated directly."
            )
        self.eval_config = eval_config

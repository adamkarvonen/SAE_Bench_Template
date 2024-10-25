from datetime import datetime
import json
from pydantic import TypeAdapter
from evals.absorption.eval_config import (
    AbsorptionEvalConfig,
)
from evals.absorption.eval_output import (
    AbsorptionEvalOutput,
    AbsorptionMetricCategories,
    AbsorptionResultDetail,
    AbsorptionMeanMetrics,
)
from sae_bench_utils import get_sae_bench_version, get_sae_lens_version
from sae_bench_utils.testing_utils import validate_eval_output_format_str

EXAMPLE_ABSORPTION_METRIC_CATEGORIES = AbsorptionMetricCategories(
    mean=AbsorptionMeanMetrics(
        mean_absorption_score=2,
        mean_num_split_features=3.5,
    )
)

EXAMPLE_ABSORPTION_EVAL_CONFIG = AbsorptionEvalConfig(
    random_seed=42,
    f1_jump_threshold=0.03,
    max_k_value=10,
    prompt_template="{word} has the first letter:",
    prompt_token_pos=-6,
    model_name="pythia-70m-deduped",
)

EXAMPLE_ABSORPTION_RESULT_DETAILS = [
    AbsorptionResultDetail(
        first_letter="a",
        absorption_rate=0.5,
        num_absorption=1,
        num_probe_true_positives=2,
        num_split_features=3,
    ),
    AbsorptionResultDetail(
        first_letter="b",
        absorption_rate=0.6,
        num_absorption=2,
        num_probe_true_positives=3,
        num_split_features=4,
    ),
]


def test_absorption_eval_output_schema():

    main_model_schema = TypeAdapter(AbsorptionEvalOutput).json_schema()

    print(json.dumps(main_model_schema, indent=2))

    # test a few things to see that we got a sane schema
    assert main_model_schema["properties"]["eval_result_details"]["type"] == "array"
    assert (
        main_model_schema["$defs"]["AbsorptionEvalConfig"]["properties"]["random_seed"][
            "default"
        ]
        == 42
    )
    assert (
        main_model_schema["properties"]["eval_type_id"]["default"]
        == "absorption_first_letter"
    )


def test_absorption_eval_output():

    eval_output = AbsorptionEvalOutput(
        eval_config=EXAMPLE_ABSORPTION_EVAL_CONFIG,
        eval_id="abc-123",
        datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
        eval_result_metrics=EXAMPLE_ABSORPTION_METRIC_CATEGORIES,
        eval_result_details=EXAMPLE_ABSORPTION_RESULT_DETAILS,
        sae_bench_commit_hash=get_sae_bench_version(),
        sae_lens_id="some_sae_lens_id",
        sae_lens_release_id="some_sae_lens_release_id",
        sae_lens_version=get_sae_lens_version(),
    )

    assert eval_output.eval_type_id == "absorption_first_letter"
    assert eval_output.eval_config == EXAMPLE_ABSORPTION_EVAL_CONFIG
    assert eval_output.eval_result_metrics == EXAMPLE_ABSORPTION_METRIC_CATEGORIES
    assert eval_output.eval_result_details == EXAMPLE_ABSORPTION_RESULT_DETAILS


def test_absorption_eval_output_json():
    json_str = """
    {
        "eval_type_id": "absorption_first_letter",
        "eval_config": {
            "random_seed": 42,
            "f1_jump_threshold": 0.03,
            "max_k_value": 10,
            "prompt_template": "{word} has the first letter:",
            "prompt_token_pos": -6,
            "model_name": "pythia-70m-deduped"
        },
        "eval_id": "0c057d5e-973e-410e-8e32-32569323b5e6",
        "datetime_epoch_millis": "1729834113150",
        "eval_result_metrics": {
            "mean": {
                "mean_absorption_score": 2,
                "mean_num_split_features": 3.5
            }
        },
        "eval_result_details": [
            {
                "first_letter": "a",
                "num_absorption": 177,
                "absorption_rate": 0.28780487804878047,
                "num_probe_true_positives": 615.0,
                "num_split_features": 1
            },
            {
                "first_letter": "b",
                "num_absorption": 51,
                "absorption_rate": 0.1650485436893204,
                "num_probe_true_positives": 309.0,
                "num_split_features": 1
            }
        ],
        "sae_bench_commit_hash": "57e9be0ac9199dba6b9f87fe92f80532e9aefced",
        "sae_lens_id": "blocks.3.hook_resid_post__trainer_10",
        "sae_lens_release_id": "sae_bench_pythia70m_sweep_standard_ctx128_0712",
        "sae_lens_version": "4.0.0"
    }
    """

    validate_eval_output_format_str(json_str, eval_output_type=AbsorptionEvalOutput)

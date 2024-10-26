import json
import os
from evals.absorption.eval_output import AbsorptionEvalOutput
from sae_bench_utils.testing_utils import validate_eval_cli_interface
import torch

import evals.absorption.eval_config as eval_config
import evals.absorption.main as absorption
from sae_bench_utils.sae_selection_utils import get_saes_from_regex
from sae_bench_utils.testing_utils import validate_eval_output_format_file

test_data_dir = "tests/test_data/absorption"
expected_results_filename = os.path.join(
    test_data_dir, "absorption_expected_results.json"
)
expected_probe_results_filename = os.path.join(
    test_data_dir, "absorption_expected_probe_results.json"
)

TEST_RELEASE = "sae_bench_pythia70m_sweep_topk_ctx128_0730"
TEST_SAE_NAME = "blocks.4.hook_resid_post__trainer_10"
TEST_TOLERANCE = 0.02


def test_absorption_cli_interface():
    parser = absorption.arg_parser()

    # Additional required args specific to absorption eval (but aren't in the config)
    additional_required = {
        "force_rerun",
    }

    validate_eval_cli_interface(
        parser,
        eval_config_cls=eval_config.AbsorptionEvalConfig,
        additional_required_args=additional_required,
    )


def test_end_to_end_different_seed():
    """Estimated runtime: 2 minutes"""
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    test_config = eval_config.AbsorptionEvalConfig(
        model_name="pythia-70m-deduped",
        random_seed=44,
        f1_jump_threshold=0.03,
        max_k_value=10,
        prompt_template="{word} has the first letter:",
        prompt_token_pos=-6,
    )
    selected_saes_dict = get_saes_from_regex(TEST_RELEASE, TEST_SAE_NAME)
    print(f"Selected SAEs: {selected_saes_dict}")

    run_results = absorption.run_eval(
        config=test_config,
        selected_saes_dict=selected_saes_dict,
        device=device,
        output_path=test_data_dir,
        force_rerun=False,
    )

    path_to_eval_results = os.path.join(
        test_data_dir, f"{TEST_RELEASE}_{TEST_SAE_NAME}_eval_results.json"
    )
    validate_eval_output_format_file(
        path_to_eval_results, eval_output_type=AbsorptionEvalOutput
    )

    # New checks for the updated JSON structure
    assert isinstance(run_results, dict), "run_results should be a dictionary"

    # Find the correct key in the new structure
    actual_result_key = f"{TEST_RELEASE}_{TEST_SAE_NAME}"
    actual_mean_absorption_rate = run_results[actual_result_key]["eval_result_metrics"][
        "mean"
    ]["mean_absorption_score"]

    # Load expected results and compare
    with open(expected_results_filename, "r") as f:
        expected_results = json.load(f)

    expected_mean_absorption_rate = expected_results["eval_result_metrics"]["mean"][
        "mean_absorption_score"
    ]
    assert (
        abs(actual_mean_absorption_rate - expected_mean_absorption_rate)
        < TEST_TOLERANCE
    )

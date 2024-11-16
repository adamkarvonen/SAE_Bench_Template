import torch
import json
from evals.shift_and_tpp.eval_config import ShiftAndTppEvalConfig
import evals.shift_and_tpp.main as shift_and_tpp
import sae_bench_utils.testing_utils as testing_utils
from sae_bench_utils.sae_selection_utils import select_saes_multiple_patterns

tpp_results_filename = "tests/test_data/shift_and_tpp/tpp_expected_results.json"
scr_results_filename = "tests/test_data/shift_and_tpp/scr_expected_results.json"


def test_scr_end_to_end_different_seed():
    """Estimated runtime: 1 minute"""
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    test_config = ShiftAndTppEvalConfig()

    test_config.dataset_names = ["LabHC/bias_in_bios_class_set1"]
    test_config.model_name = "pythia-70m-deduped"
    test_config.random_seed = 48
    test_config.n_values = [10]
    test_config.sae_batch_size = 250
    test_config.llm_batch_size = 500
    test_config.llm_dtype = "float32"
    layer = 4
    tolerance = 0.08  # There can be significant variation in the strength of the correlation learned by a linear probe between random seeds
    # This causes large shifts in absolute values of the shift metrics, especially as this test only uses a single dataset

    test_config.perform_scr = True
    test_config.column1_vals_lookup = {
        "LabHC/bias_in_bios_class_set1": [
            ("professor", "nurse"),
        ],
    }

    sae_regex_patterns = [
        r"(sae_bench_pythia70m_sweep_topk_ctx128_0730).*",
    ]
    sae_block_pattern = [
        rf".*blocks\.([{layer}])\.hook_resid_post__trainer_(10)$",
    ]

    selected_saes = select_saes_multiple_patterns(sae_regex_patterns, sae_block_pattern)

    run_results = shift_and_tpp.run_eval(
        test_config,
        selected_saes,
        device,
        output_path="evals/shift_and_tpp/test_results/",
        force_rerun=True,
        clean_up_activations=True,
    )

    with open(scr_results_filename, "r") as f:
        expected_results = json.load(f)

    keys_to_compare = [
        "scr_metric_threshold_10",
    ]

    testing_utils.compare_dicts_within_tolerance(
        run_results[
            "sae_bench_pythia70m_sweep_topk_ctx128_0730_blocks.4.hook_resid_post__trainer_10"
        ]["eval_result_metrics"]["shift_metrics"],
        expected_results["eval_result_metrics"]["shift_metrics"],
        tolerance,
        keys_to_compare=keys_to_compare,
    )


def test_tpp_end_to_end_different_seed():
    """Estimated runtime: 1 minute"""
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    test_config = ShiftAndTppEvalConfig()

    test_config.dataset_names = ["LabHC/bias_in_bios_class_set1"]
    test_config.model_name = "pythia-70m-deduped"
    test_config.random_seed = 44
    test_config.n_values = [10]
    test_config.sae_batch_size = 250
    test_config.llm_batch_size = 500
    test_config.llm_dtype = "float32"
    layer = 4
    tolerance = 0.04

    test_config.perform_scr = False

    sae_regex_patterns = [
        r"(sae_bench_pythia70m_sweep_topk_ctx128_0730).*",
    ]
    sae_block_pattern = [
        rf".*blocks\.([{layer}])\.hook_resid_post__trainer_(10)$",
    ]

    selected_saes = select_saes_multiple_patterns(sae_regex_patterns, sae_block_pattern)

    run_results = shift_and_tpp.run_eval(
        test_config,
        selected_saes,
        device,
        output_path="evals/shift_and_tpp/test_results/",
        force_rerun=True,
        clean_up_activations=True,
    )

    with open(tpp_results_filename, "r") as f:
        expected_results = json.load(f)

    keys_to_compare = [
        "tpp_threshold_10_total_metric",
        "tpp_threshold_10_intended_diff_only",
        "tpp_threshold_10_unintended_diff_only",
    ]

    testing_utils.compare_dicts_within_tolerance(
        run_results[
            "sae_bench_pythia70m_sweep_topk_ctx128_0730_blocks.4.hook_resid_post__trainer_10"
        ]["eval_result_metrics"]["tpp_metrics"],
        expected_results["eval_result_metrics"]["tpp_metrics"],
        tolerance,
        keys_to_compare=keys_to_compare,
    )

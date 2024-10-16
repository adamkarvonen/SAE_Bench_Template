import json
import os
import uuid
import tempfile
import shutil

import torch

import evals.absorption.eval_config as eval_config
import evals.absorption.main as absorption
import sae_bench_utils.formatting_utils as formatting_utils
import sae_bench_utils.testing_utils as testing_utils
import sae_bench_utils

test_data_dir = "tests/test_data/absorption"
expected_results_filename = os.path.join(test_data_dir, "absorption_expected_results.json")
expected_probe_results_filename = os.path.join(test_data_dir, "absorption_expected_probe_results.json")

TEST_RELEASE = "sae_bench_pythia70m_sweep_topk_ctx128_0730"
TEST_SAE_NAME = "pythia70m_sweep_topk_ctx128_0730/resid_post_layer_4/trainer_10"

def test_end_to_end_different_seed():
    """Estimated runtime: 2 minutes"""
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Create a temporary directory for test outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        test_config = eval_config.EvalConfig()
        test_config.sae_releases = [
            "sae_bench_pythia70m_sweep_topk_ctx128_0730",
        ]

        test_config.model_name = "pythia-70m-deduped"
        test_config.layer = 4
        test_config.trainer_ids = [10]
        test_config.include_checkpoints = False
        test_config.random_seed = 44
        tolerance = 0.02

        # populate selected_saes_dict using config values
        for release in test_config.sae_releases:
            if "gemma-scope" in release:
                test_config.selected_saes_dict[release] = (
                    formatting_utils.find_gemmascope_average_l0_sae_names(test_config.layer)
                )
            else:
                test_config.selected_saes_dict[release] = formatting_utils.filter_sae_names(
                    sae_names=release,
                    layers=[test_config.layer],
                    include_checkpoints=test_config.include_checkpoints,
                    trainer_ids=test_config.trainer_ids,
                )

            print(f"SAE release: {release}, SAEs: {test_config.selected_saes_dict[release]}")

        run_results = absorption.run_eval(test_config, test_config.selected_saes_dict, device, test_data_dir)

        # New checks for the updated JSON structure
        assert isinstance(run_results, dict), "run_results should be a dictionary"
        
        for key, value in run_results.items():
            assert isinstance(value, dict), f"Each item in run_results should be a dictionary, but {key} is not"
            assert "eval_instance_id" in value, f"eval_instance_id missing in {key}"
            assert "sae_lens_release" in value, f"sae_lens_release missing in {key}"
            assert "sae_lens_id" in value, f"sae_lens_id missing in {key}"
            assert "eval_type_id" in value, f"eval_type_id missing in {key}"
            assert "sae_lens_version" in value, f"sae_lens_version missing in {key}"
            assert "sae_bench_version" in value, f"sae_bench_version missing in {key}"
            assert "date_time" in value, f"date_time missing in {key}"
            assert "eval_config" in value, f"eval_config missing in {key}"
            assert "eval_results" in value, f"eval_results missing in {key}"
            assert "eval_artifacts" in value, f"eval_artifacts missing in {key}"

            assert value["eval_type_id"] == "absorption", f"eval_type_id should be 'absorption', but got {value['eval_type_id']}"
            assert uuid.UUID(value["eval_instance_id"]), f"eval_instance_id should be a valid UUID, but got {value['eval_instance_id']}"

            # Check if k_sparse_probing_results artifact exists
            k_sparse_probing_path = os.path.join(test_data_dir, value["eval_artifacts"]["k_sparse_probing_results"])
            assert os.path.exists(k_sparse_probing_path), f"k_sparse_probing_results artifact not found at {k_sparse_probing_path}"

        # Find the correct key in the new structure
        actual_result_key = f"{TEST_RELEASE}_{TEST_SAE_NAME}"
        actual_mean_absorption_rate = run_results[actual_result_key]["eval_results"]["mean_absorption_rate"]

        # Load expected results and compare
        with open(expected_results_filename, "r") as f:
            expected_results = json.load(f)

        expected_mean_absorption_rate = expected_results["eval_results"]["mean_absorption_rate"]

        assert abs(actual_mean_absorption_rate - expected_mean_absorption_rate) < tolerance

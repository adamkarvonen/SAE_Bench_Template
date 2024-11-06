import json
import os
import argparse
from evals.core.eval_output import CoreEvalOutput
from sae_bench_utils.testing_utils import validate_eval_cli_interface
import torch
import evals.core.eval_config as eval_config
import evals.core.main as core
from sae_bench_utils.sae_selection_utils import get_saes_from_regex
from sae_bench_utils.testing_utils import validate_eval_output_format_file

test_data_dir = "tests/test_data/core"
expected_results_filename = os.path.join(
    test_data_dir, "core_expected_results.json"
)

TEST_RELEASE = "sae_bench_pythia70m_sweep_gated_ctx128_0730"
TEST_SAE_NAME = "blocks.3.hook_resid_post__trainer_5"
TEST_TOLERANCE = 0.02


def test_core_cli_interface():
    parser = core.arg_parser()

    # Additional required args specific to core eval (but aren't in the config)
    additional_required = {
        "force_rerun",
    }

    validate_eval_cli_interface(
        parser,
        eval_config_cls=eval_config.CoreEvalConfig,
        additional_required_args=additional_required,
    )


def test_end_to_end():
    """Estimated runtime: 2 minutes"""
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    test_config = eval_config.CoreEvalConfig(
        model_name="pythia-70m-deduped",
        batch_size_prompts=4,
        n_eval_reconstruction_batches=5,
        n_eval_sparsity_variance_batches=20,
        compute_kl=True,
        compute_ce_loss=True,
        compute_l2_norms=True,
        compute_sparsity_metrics=True,
        compute_variance_metrics=True,
        compute_featurewise_density_statistics=True,
        compute_featurewise_weight_based_metrics=True,
        dataset="Skylion007/openwebtext",
        context_size=128,
    )
    
    # Run evaluations
    eval_results = core.run_evaluations(
        argparse.Namespace(
            sae_regex_pattern=TEST_RELEASE,
            sae_block_pattern=TEST_SAE_NAME,
            n_eval_reconstruction_batches=test_config.n_eval_reconstruction_batches,
            n_eval_sparsity_variance_batches=test_config.n_eval_sparsity_variance_batches,
            batch_size_prompts=test_config.batch_size_prompts,
            dataset=test_config.dataset,
            context_size=test_config.context_size,
            output_folder=test_data_dir,
            verbose=False,
            force_rerun=True,
            compute_kl=test_config.compute_kl,
            compute_ce_loss=test_config.compute_ce_loss,
            compute_l2_norms=test_config.compute_l2_norms,
            compute_sparsity_metrics=test_config.compute_sparsity_metrics,
            compute_variance_metrics=test_config.compute_variance_metrics,
            compute_featurewise_density_statistics=test_config.compute_featurewise_density_statistics,
            compute_featurewise_weight_based_metrics=test_config.compute_featurewise_weight_based_metrics,
        )
    )
    
    path_to_eval_results = os.path.join(
        test_data_dir, f"{TEST_RELEASE}-{TEST_SAE_NAME}_{test_config.context_size}_{test_config.dataset.replace('/', '_')}.json"
    )
    print(f"Path to eval results: {path_to_eval_results}")
    validate_eval_output_format_file(path_to_eval_results, eval_output_type=CoreEvalOutput)
    
    # Verify results structure and key metrics
    assert isinstance(eval_results, list), "eval_results should be a list"
    assert len(eval_results) > 0, "eval_results should not be empty"
    
    result = eval_results[0]  # Get the first result
    
    # Load expected results and compare
    with open(expected_results_filename, "r") as f:
        expected_results = json.load(f)
    
    # Compare key metrics with expected values
    actual_metrics = result["metrics"]
    expected_metrics = expected_results["eval_result_metrics"]
    
    # Check reconstruction quality metrics
    if "reconstruction_quality" in actual_metrics:
        actual_exp_var = actual_metrics["reconstruction_quality"]["explained_variance"]
        expected_exp_var = expected_metrics["reconstruction_quality"]["explained_variance"]
        assert abs(actual_exp_var - expected_exp_var) < TEST_TOLERANCE
    
    # Check sparsity metrics
    if "sparsity" in actual_metrics:
        actual_l0 = actual_metrics["sparsity"]["l0"]
        expected_l0 = expected_metrics["sparsity"]["l0"]
        assert abs(actual_l0 - expected_l0) < TEST_TOLERANCE

def test_feature_metrics():
    """Test the feature-wise metrics computation"""
    test_config = eval_config.CoreEvalConfig(
        model_name="pythia-70m-deduped",
        batch_size_prompts=4,
        n_eval_reconstruction_batches=5,
        n_eval_sparsity_variance_batches=20,
        compute_featurewise_density_statistics=True,
        compute_featurewise_weight_based_metrics=True,
        dataset="Skylion007/openwebtext",
        context_size=128,
    )
    
    eval_results = core.run_evaluations(
        argparse.Namespace(
            sae_regex_pattern=TEST_RELEASE,
            sae_block_pattern=TEST_SAE_NAME,
            n_eval_reconstruction_batches=test_config.n_eval_reconstruction_batches,
            n_eval_sparsity_variance_batches=test_config.n_eval_sparsity_variance_batches,
            batch_size_prompts=test_config.batch_size_prompts,
            dataset=test_config.dataset,
            context_size=test_config.context_size,
            output_folder=test_data_dir,
            verbose=False,
            force_rerun=True,
            compute_featurewise_density_statistics=test_config.compute_featurewise_density_statistics,
            compute_featurewise_weight_based_metrics=test_config.compute_featurewise_weight_based_metrics,
        )
    )
    
    result = eval_results[0]
    feature_metrics = result["feature_metrics"]
    
    # Check that feature metrics contain the expected fields
    expected_fields = {
        "feature_density",
        "consistent_activation_heuristic",
        "encoder_bias",
        "encoder_norm",
        "encoder_decoder_cosine_sim"
    }
    
    assert all(field in feature_metrics for field in expected_fields)
    
    # Check that all feature metrics have the same length
    lengths = {len(feature_metrics[field]) for field in expected_fields}
    assert len(lengths) == 1, "All feature metrics should have the same length"
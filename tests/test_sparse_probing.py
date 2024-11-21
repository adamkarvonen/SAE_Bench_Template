import json
import torch
from evals.sparse_probing.eval_config import SparseProbingEvalConfig
import evals.sparse_probing.main as sparse_probing
import sae_bench_utils.testing_utils as testing_utils
from sae_bench_utils.sae_selection_utils import select_saes_multiple_patterns

results_filename = "tests/test_data/sparse_probing/sparse_probing_expected_results.json"


def test_end_to_end_different_seed():
    """Estimated runtime: 1 minute"""
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    test_config = SparseProbingEvalConfig()

    test_config.dataset_names = ["LabHC/bias_in_bios_class_set1"]
    test_config.model_name = "pythia-70m-deduped"
    test_config.random_seed = 44
    test_config.llm_batch_size = 512
    tolerance = 0.04
    test_config.k_values = [1, 2, 5, 10, 20, 50, 100]
    test_config.llm_dtype = "float32"

    layer = 4

    sae_regex_patterns = [
        r"(sae_bench_pythia70m_sweep_topk_ctx128_0730).*",
    ]
    sae_block_pattern = [
        rf".*blocks\.([{layer}])\.hook_resid_post__trainer_(10)$",
    ]

    selected_saes = select_saes_multiple_patterns(sae_regex_patterns, sae_block_pattern)

    run_results = sparse_probing.run_eval(
        test_config,
        selected_saes,
        device,
        output_path="evals/sparse_probing/test_results/",
        force_rerun=True,
        clean_up_activations=True,
    )

    with open(results_filename, "r") as f:
        expected_results = json.load(f)

    run_result_metrics = run_results[
        "sae_bench_pythia70m_sweep_topk_ctx128_0730_blocks.4.hook_resid_post__trainer_10"
    ]["eval_result_metrics"]

    keys_to_compare = ["llm_test_accuracy"]
    for k in test_config.k_values:
        keys_to_compare.append(f"llm_top_{k}_test_accuracy")

    testing_utils.compare_dicts_within_tolerance(
        run_result_metrics["llm"],
        expected_results["eval_result_metrics"]["llm"],
        tolerance,
        keys_to_compare=keys_to_compare,
    )

    keys_to_compare = []
    for k in test_config.k_values:
        keys_to_compare.append(f"sae_top_{k}_test_accuracy")
    testing_utils.compare_dicts_within_tolerance(
        run_result_metrics["sae"],
        expected_results["eval_result_metrics"]["sae"],
        tolerance,
        keys_to_compare=keys_to_compare,
    )

import json
import torch
from evals.unlearning.eval_config import UnlearningEvalConfig
import evals.unlearning.main as unlearning
import sae_bench_utils.testing_utils as testing_utils
from sae_bench_utils.sae_selection_utils import select_saes_multiple_patterns

results_filename = "tests/test_data/unlearning/sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824_blocks.3.hook_resid_post__trainer_2_eval_results.json"


def test_end_to_end_different_seed():
    """Estimated runtime: 5 minutes
    NOTE: Will require bio-forget-corpus.jsonl to be present in the data directory (see unlearning/README.md)"""
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    test_config = UnlearningEvalConfig()

    test_config.retain_thresholds = [0.01]
    test_config.n_features_list = [10]
    test_config.multipliers = [25]

    test_config.dataset_size = 256

    test_config.random_seed = 48
    tolerance = 0.04
    test_config.llm_dtype = "bfloat16"

    sae_regex_patterns = [
        r"sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824",
    ]
    sae_block_pattern = [
        r"blocks.3.hook_resid_post__trainer_2",
    ]

    selected_saes_dict = select_saes_multiple_patterns(sae_regex_patterns, sae_block_pattern)

    run_results = unlearning.run_eval(
        test_config,
        selected_saes_dict,
        device,
        output_path="evals/unlearning/test_results/",
        force_rerun=True,
        clean_up_artifacts=True,
    )

    with open(results_filename, "r") as f:
        expected_results = json.load(f)

    sae_name = "sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824_blocks.3.hook_resid_post__trainer_2"

    run_result_metrics = run_results[
        sae_name
    ]["eval_result_metrics"]

    testing_utils.compare_dicts_within_tolerance(
        run_result_metrics,
        expected_results["eval_result_metrics"],
        tolerance,
        keys_to_compare=["unlearning_score"],
    )

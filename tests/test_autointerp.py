import json
import torch
from evals.autointerp.eval_config import AutoInterpEvalConfig
import evals.autointerp.main as autointerp
import sae_bench_utils.testing_utils as testing_utils
from sae_bench_utils.sae_selection_utils import select_saes_multiple_patterns

results_filename = (
    "tests/test_data/autointerp/pythia-70m-deduped_layer_4_expected_eval_results.json"
)


def test_end_to_end_different_seed():
    """Estimated runtime: 1 minute.
    Note: Will require an OpenAI API key saved to openai_api_key.txt."""
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    with open("openai_api_key.txt", "r") as f:
        openai_api_key = f.read().strip()

    test_config = AutoInterpEvalConfig(model_name="pythia-70m-deduped")

    test_config.n_latents = 100

    test_config.random_seed = 48
    tolerance = 0.04
    test_config.llm_dtype = "float32"
    test_config.llm_batch_size = 512

    layer = 4

    sae_regex_patterns = [
        r"(sae_bench_pythia70m_sweep_topk_ctx128_0730).*",
    ]
    sae_block_pattern = [
        rf".*blocks\.([{layer}])\.hook_resid_post__trainer_(10)$",
    ]

    selected_saes = select_saes_multiple_patterns(sae_regex_patterns, sae_block_pattern)

    run_results = autointerp.run_eval(
        test_config,
        selected_saes,
        device,
        openai_api_key,
        output_path="evals/autointerp/test_results/",
        force_rerun=True,
    )

    with open("test.json", "w") as f:
        json.dump(run_results, f)

    with open(results_filename, "r") as f:
        expected_results = json.load(f)

    sae_name = "sae_bench_pythia70m_sweep_topk_ctx128_0730_blocks.4.hook_resid_post__trainer_10"

    run_result_metrics = run_results[sae_name]["eval_result_metrics"]

    testing_utils.compare_dicts_within_tolerance(
        run_result_metrics,
        expected_results[sae_name]["eval_result_metrics"],
        tolerance,
        keys_to_compare=["autointerp_score"],
    )

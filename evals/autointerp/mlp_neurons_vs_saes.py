import baselines.identity_sae as identity_sae
import evals.autointerp.main as autointerp
import sae_bench_utils.sae_selection_utils as sae_selection_utils
import sae_bench_utils.general_utils as general_utils

if __name__ == "__main__":
    model_name = "gemma-2-2b"
    layers = [5, 12, 19]
    d_model = 9216
    llm_dtype = "bfloat16"
    llm_batch_size = 32
    RANDOM_SEED = 42
    device = general_utils.setup_environment()
    force_rerun = False

    with open("openai_api_key.txt", "r") as f:
        api_key = f.read().strip()

    for layer in layers:
        mlp_hook = f"blocks.{layer}.mlp.hook_post"
        mlp_neurons = identity_sae.IdentitySAE(model_name, d_model, layer, hook_name=mlp_hook)

        selected_saes = [(f"identity_mlp_{model_name}_layer_{layer}", mlp_neurons)]

        autointerp.run_eval(
            autointerp.AutoInterpEvalConfig(
                model_name=model_name,
                random_seed=RANDOM_SEED,
                llm_batch_size=llm_batch_size,
                llm_dtype=llm_dtype,
            ),
            selected_saes,
            device,
            api_key,
            "evals/autointerp/results/",
            force_rerun,
        )

    for layer in layers:
        sae_regex_pattern = r"gemma-scope-2b-pt-mlp"
        sae_block_pattern = rf".*layer_{layer}.*(16k).*"

        selected_saes = sae_selection_utils.get_saes_from_regex(
            sae_regex_pattern, sae_block_pattern
        )

        autointerp.run_eval(
            autointerp.AutoInterpEvalConfig(
                model_name=model_name,
                random_seed=RANDOM_SEED,
                llm_batch_size=llm_batch_size,
                llm_dtype=llm_dtype,
            ),
            selected_saes,
            device,
            api_key,
            "evals/autointerp/results/",
            force_rerun,
        )

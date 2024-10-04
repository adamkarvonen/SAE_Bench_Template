This repo implements k-sparse probing, where k can be any integer less than the SAE's hidden dim.

Estimated runtime:

- Pythia-70M: ~10 seconds to collect activations per layer with SAEs, ~20 seconds per SAE to perform probing
- Gemma-2-2B: ~2 minutes to collect activations per layer with SAEs, ~20 seconds per SAE to perform probing

Using Gemma-2-2B, at current batch sizes, I see a peak GPU memory usage of 22 GB.

All configuration arguments and hyperparameters are located in `eval_config.py`. The full eval config is saved to the results json file.

If ran in the current state, `cd` in to `evals/sparse_probing/` and run `python main.py`. It should produce `evals/sparse_probing/results/sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824_eval_results.json`.

If interested, `utils/activation_collection.py` contains example functions to collect activations using TransformerLens.

If the random seed is set, it's fully deterministic and results match perfectly using `compare_run_results.ipynb`. If the random seed is not set, results will vary up to 0.03 for some outlier values, with a mean difference of 0.005.
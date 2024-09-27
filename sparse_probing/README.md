This repo implements k-sparse probing, where k can be any integer less than the SAE's hidden dim.

Estimated runtime:

- Pythia-70M: 10 seconds to collect activations per layer with SAEs, 10 seconds per SAE to perform probing
- Gemma-2-2B: 2 minutes to collect activations per layer with SAEs, 10 seconds per SAE to perform probing

All configuration arguments and hyperparameters are located in `eval_config.py`. The full eval config is saved to the results json file.

If ran in the current state, `cd` in to `src/` and run `python sparse_probing_eval.py`. It should produce `sparse_probing_results/sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824_eval_results.json`.

If interested, `activation_collection.py` contains example functions to collect activations using TransformerLens.

If the random seed is set, it's fully deterministic and results match perfectly using `compare_run_results.ipynb`. If the random seed is not set, results will vary up to 1.0 for some outlier values (TODO: Investigate), with a mean difference of 0.01.
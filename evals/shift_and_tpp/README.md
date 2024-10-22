This repo implements the SHIFT and TPP evals from "Evaluating Sparse Autoencoders on Targeted Concept Removal Tasks".

To run SHIFT, set eval_config.spurious_corr = True. To run TPP, set it to False.

Estimated runtime per dataset (currently there are 2 datasets, and for SHIFT we have 4 class pairs per dataset, so 2x4=8 iterations):

- Pythia-70M: ~10 seconds to collect activations per layer with SAEs, ~20 seconds per SAE to perform the evaluation
- Gemma-2-2B: ~2 minutes to collect activations per layer with SAEs, ~40 seconds per SAE to perform the evaluation

Using Gemma-2-2B, at current batch sizes, I see a peak GPU memory usage of 22 GB. This fits on a 3090.

All configuration arguments and hyperparameters are located in `eval_config.py`. The full eval config is saved to the results json file.

If ran in the current state, `cd` in to `evals/shift_and_tpp/` and run `python main.py`. It should produce `evals/shift_and_tpp/results/pythia-70m-deduped_tpp_layer_4_eval_results.json`.

`tests/test_shift_and_tpp.py` contains an end-to-end test of the evals. Running `pytest -s tests/test_shift_and_tpp` will verify that the actual results are within the specified tolerance of the expected results.

If the random seed is set, it's fully deterministic and results match perfectly using `compare_run_results.ipynb` or the end to end tests. For TPP, the maximum difference is 0.008. SHIFT's maximum difference is larger and requires investigation.
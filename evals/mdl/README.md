This folder implements an MDL-based eval from "[Interpretability as Compression: Reconsidering SAE Explanations of Neural Activations with MDL-SAEs](https://www.lesswrong.com/posts/G2oyFQFTE5eGEas6m/interpretability-as-compression-reconsidering-sae)".

Estimated runtime:

For a 16k width SAE, 2.5 minutes per `num_bins_value`. Runtime primarily scales with dictionary width, as there's a for loop over all SAE latents which can't be easily vectorized.

This eval fits on an RTX 3090 with Gemma-2-2B.

All configuration arguments and hyperparameters are located in `eval_config.py`. The full eval config is saved to the results json file.

Example output (including the eval config that generated the output) can be found in `test_data/mdl`.

TODO: Add tests

This repo contains the template we would like to use for the SAE Bench project. The `template.ipynb` explains the input to your custom eval (SAEs hosted on SAELens) and the output (a standardized results file).

The `sparse_probing` folder contains a full example implementation of a custom eval. In `main.py`, we have a function that takes a list of SAELens SAE names (defined in `eval_config.py`) and an sae release and returns a dictionary of results in a standard format. This folder contains some helper functions, like pre-computing model activations in `activation_collection.py`, that might be useful for you, too! We try to reuse functions as much as possible across evals to reduce bugs. Let Adam and Can know if you've implemented a helper function that might be useful for other evals as well (like autointerp, feature scoring). 

`python3 main.py` should run as is and demonstrate how to use our SAE Bench SAEs with Transformer Lens and SAE Lens. It will also generate a results file which can be graphed using `graphing.ipynb`.

Here is what we would like to see from each eval:

- Making sure we are returned both the results any config required for reproducibility (eg: eval config / function args).
- Ensuring the code meets some minimum bar (isn't missing anything, isn't abysmally slow etc).
- Ensuring we have example output to validate against.

the results dictionary of you custom eval can be loaded in to `graphing.ipynb`, to create a wide variety of plots. We also already have the basic `L0 / Loss Recovered` metrics for every SAE, as specified in `template.ipynb`.

For the purpose of validating evaluation outputs, we have `compare_run_results.ipynb`. Using this, you can run the same eval twice with the same input, and verify within a tolerance that it returns the same outputs. If the eval is fully deterministic, the results should be identical.

Once evaluations have been completed, please submit them as pull requests to this repo.

## Quick start:
1. Browse through `template.ipynb` to learn about input and output of your metric.
2. Execute `sparse_probing/sparse_probing_eval.py` to see an example implementation.
3. Inspect sparse probing results with `graphing.ipynb`.

## SAE Bench: Template for custom evals

This repo contains the template we would like to use for the SAE Bench project. The `template.ipynb` explains the input to your custom eval (SAEs hosted on SAELens) and the output (a standardized results file).

### Installation
Set up a virtual environment running on `python 3.11.9`.
```
git clone https://github.com/adamkarvonen/SAE_Bench_Template.git
cd SAE_Bench_Template
pip install -e .
```

### Quick start:

1. Browse through `template.ipynb` to learn about input and output of your metric.
2. Execute `main.py` in `evals/sparse_probing` to see an example implementation.
3. Inspect sparse probing results with `graphing.ipynb`.

## Overview

The `evals/sparse_probing` folder contains a full example implementation of a custom eval. In `main.py`, we have a function that takes a list of SAELens SAE names (defined in `eval_config.py`) and an sae release and returns a dictionary of results in a standard format. This folder contains some helper functions, like pre-computing model activations in `utils/activation_collection.py`, that might be useful for you, too! We try to reuse functions as much as possible across evals to reduce bugs. Let Adam and Can know if you've implemented a helper function that might be useful for other evals as well (like autointerp, feature scoring). 

`python3 main.py` (after `cd evals/sparse_probing/`) should run as is and demonstrate how to use our SAE Bench SAEs with Transformer Lens and SAE Lens. It will also generate a results file which can be graphed using `graphing.ipynb`.

Here is what we would like to see from each eval:

- Making sure we are returned both the results any config required for reproducibility (eg: eval config / function args).
- Ensuring the code meets some minimum bar (isn't missing anything, isn't abysmally slow etc).
- Ensuring we have example output to validate against.

the results dictionary of you custom eval can be loaded in to `graphing.ipynb`, to create a wide variety of plots. We also already have the basic `L0 / Loss Recovered` metrics for every SAE, as specified in `template.ipynb`.

For the purpose of validating evaluation outputs, we have `compare_run_results.ipynb`. Using this, you can run the same eval twice with the same input, and verify within a tolerance that it returns the same outputs. If the eval is fully deterministic, the results should be identical.

Once evaluations have been completed, please submit them as pull requests to this repo.

## Eval Format

Ideally, we would like to see something like `evals.sparse_probing.main.py`, which contains a `run_eval()` function. This function takes in hyperparameters and a `selected_saes_dict`, which is a dict of `sae_release` : `list[sae_names]` (as shown in template.ipynb).

All evals and submodules will share the same dependencies, which are set in pyproject.toml.

For a tutorial of using SAE Lens SAEs, including calculating L0 and Loss Recovered and getting a set of tokens from The Pile, refer to this notebook: https://github.com/jbloomAus/SAELens/blob/main/tutorials/basic_loading_and_analysing.ipynb

## Custom SAE Usage

For the sparse probing and SHIFT / TPP evals, we support evaluating any SAE object that has the following implemented, with inputs / outputs matching the SAELens SAE format:

```
sae.encode()
sae.decode()
sae.forward()
sae.W_dec # nn.Parameter(d_sae, d_in), required for SHIFT, TPP, and Feature Absorption
sae.device
sae.dtype
```

Just pass the appropriate inputs to `run_eval_single_sae()`, referring to individual eval READMEs as needed. If you match our output format you can reuse our graphing notebook.

To run our baselines in pythia-70m and gemma-2-2b, refer to `if __name__ == "__main__":` in `shift_and_tpp/main.py`.
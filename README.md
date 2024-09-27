This repo contains the template we would like to use for the SAE Bench project. The template is the `sparse_probing` folder. In it, we have a function that takes a list of SAELens SAE names and an sae release and returns a dictionary of results in a standard format.

`sparse_probing` should run as is and demonstrate how to use our SAE Bench SAEs with Transformer Lens and SAE Lens. It will also generate a results file which can be graphed using `graph_sae_results.ipynb`.

Here is what we would like to see from each eval:

- Making sure we are returned both the results any config required for reproducibility (eg: eval config / function args).
- Ensuring the code meets some minimum bar (isn't missing anything, isn't abysmally slow etc).
- Ensuring we have example output to validate against.

In `template.ipynb`, we show the structure that we would like to see in the results dictionary which an eval should return. This results dictionary can be loaded in to `graph_sae_results.ipynb`, to create a wide variety of plots. We also already have the basic `L0 / Loss Recovered` metrics for every SAE, as specified in `template.ipynb`.

For the purpose of validating evaluation outputs, we have `compare_run_results.ipynb`. Using this, you can run the same eval twice with the same input, and verify within a tolerance that it returns the same outputs. If the eval is fully deterministic, the results should be identical.

Once evaluations have been completed, please submit them as pull requests to this repo.
# SAE Bench

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Running Evaluations](#running-evaluations)
- [Custom SAE Usage](#custom-sae-usage)
- [Training Your Own SAEs](#training-your-own-saes)
- [Graphing Results](#graphing-results)

CURRENT REPO STATUS: SAE Bench is currently a beta release. This repo is still under development as we clean up some of the rough edges left over from the research process. However, it is usable in the current state for both SAE Lens SAEs and custom SAEs.


## Overview

SAE Bench is a comprehensive suite of 8 evaluations for Sparse Autoencoder (SAE) models:
- **[Feature Absorption](https://arxiv.org/abs/2409.14507)**
- **[AutoInterp](https://blog.eleuther.ai/autointerp/)**
- **L0 / Loss Recovered**
- **[RAVEL](https://arxiv.org/abs/2402.17700) (under development)**
- **[Spurious Correlation Removal (SCR)](https://arxiv.org/abs/2411.18895)**
- **[Targeted Probe Pertubation (TPP)](https://arxiv.org/abs/2411.18895)**
- **Sparse Probing**
- **[Unlearning](https://arxiv.org/abs/2410.19278)** (requires access to the WMDP dataset, see README)

### Supported Models and SAEs
- **SAE Lens Pretrained Models**: Supports evaluations on any SAE Lens pretrained model.
- **Custom SAEs**: Supports any general SAE object with `encode()` and `decode()` methods (see [Custom SAE Usage](#custom-sae-usage)).

### Installation
Set up a virtual environment with python >= 3.10.

```
git clone https://github.com/adamkarvonen/SAEBench.git
cd SAEBench
pip install -e .
```

All evals can be ran with current batch sizes on Gemma-2-2B on a 24GB VRAM GPU (e.g. a RTX 3090). By default, some evals cache LLM activations, which can require up to 100 GB of disk space. However, this can be disabled.

Autointerp requires the creation of `openai_api_key.txt`. Unlearning requires requesting access to the WMDP bio dataset (refer to `unlearning/README.md`).

## Getting Started

We recommend to get starting by going through the `sae_bench_demo.ipynb` notebook. In this notebook, we load both a custom SAE and an SAE Lens SAE, run both of them on multiple evaluations, and plot graphs of the results.

## Running Evaluations
Each evaluation has an example command located in its respective `main.py` file. To run all evaluations on a selection of SAE Lens SAEs, refer to `shell_scripts/README.md`. Here's an example of how to run a sparse probing evaluation on a single SAE Bench Pythia-70M SAE:

```
python evals/sparse_probing/main.py \
    --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
    --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
    --model_name pythia-70m-deduped
```

The results will be saved to the `eval_results/sparse_probing` directory.

We use regex patterns to select SAE Lens SAEs. For more examples of regex patterns, refer to `sae_regex_selection.ipynb`.

Every eval folder contains an `eval_config.py`, which contains all relevant hyperparamters for that evaluation. The values are currently set to the default recommended values.

For a tutorial of using SAE Lens SAEs, including calculating L0 and Loss Recovered and getting a set of tokens from The Pile, refer to this notebook: https://github.com/jbloomAus/SAELens/blob/main/tutorials/basic_loading_and_analysing.ipynb

## Custom SAE Usage

Our goal is to have first class support for custom SAEs as the field is rapidly evolving. Our evaluations can run on any SAE object with `encode()`, `decode()`, and a few config values. We recommend referring to `sae_bench_demo.ipynb`. In this notebook, we load a custom SAE and an SAE Bench baseline SAE, run them on two evals, and graph the results. There is additional information about custom SAE usage in `custom_saes/README.md`.

There are two ways to evaluate custom SAEs:

1. **Using Evaluation Templates**: 
   - Use the secondary `if __name__ == "__main__"` block in each `main.py`
   - Results are saved in SAE Bench format for easy visualization
   - Compatible with provided plotting tools

2. **Direct Function Calls**:
   - Use `run_eval_single_sae()` in each `main.py`
   - Simpler interface requiring only model, SAE, and config values
   - Graphing will require manual formatting

We currently have a suite of SAE Bench SAEs on layers 3 and 4 of Pythia-70M and layers 5, 12, and 19 of Gemma-2-2B, each trained on 200M tokens with checkpoints at various points. These SAEs can serve as baselines for any new custom SAEs. We also have baseline eval results, saved at TODO.

## Training Your Own SAEs

You can deterministically replicate the training of our SAEs using scripts provided [here](https://github.com/canrager/dictionary_training/), or implement your own SAE, or make a change to one of our SAE implementations. Once you train your new version, you can benchmark against our existing SAEs for a true apples to apples comparison.

## Graphing Results

If evaluating your own SAEs, we recommend using the graphing cells in `sae_bench_demo.ipynb`. To replicate all SAE Bench plots, refer to `graphing.ipynb`. In this notebook, we download all SAE Bench data and create a variety of plots.
### Setup
1. request the forget corpus from this [link](https://docs.google.com/forms/d/e/1FAIpQLSdnQc8Qn0ozSDu3VE8HLoHPvhpukX1t1dIwE5K5rJw9lnOjKw/viewform)
2. you will get one file: `bio-forget-corpus.jsonl`, place it the `evals/unlearning/data` directory
3. see [this page](https://huggingface.co/datasets/cais/wmdp-corpora) for more details

To run this eval, run `cd SAE_Bench_Template` and a command such as this one:

```
python evals/unlearning/main.py     --sae_regex_pattern "sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824"     --sae_block_pattern "blocks.3.hook_resid_post__trainer_2"     --model_name gemma-2-2b-it --force_rerun
```

Currently, the instruct prompt template is only added for Gemma-2-2B-it. Other prompt templates can be added in `evals/unlearning/utils/var.py`. This eval should only be ran on instruct models.

This eval fits on a RTX 3090 using Gemma-2-2B-it.

If running a new model, it takes around ~20 minutes to get `question_ids/` (the questions the LLM knows how to answer correctly). After that, it's around 10 minutes per SAE.
The unlearning score is evaluated by sweeping across a combination of `retain_thresholds`, `n_features`, and `multipliers`. We then find the best unlearning score where the MMLU accuracy is still > 99% of the original MMLU accuracy. This means that we can get a more accurate estimate of the unlearning score by sweeping across more hyperparameters at the cost of increased runtime.

The hyperparameters were set to obtain a good unlearning score on TopK and Standard SAEs on Gemma across layers 3, 11, and 19. It's possible that they may not represent the best hyperparameters on other LLMs. If evaluating a new LLM, it would require adding a instruct prompt template. You may also want to sweep a wider range of hyperparameters for initial SAEs, to see if the default hyperparameters capture the best unlearning score. This analysis can be done using `example.ipynb`.

### Eval
* after executing `main.py`, the following will happen:
    1. the feature sparsity for the forget and retain dataset will be saved at `artifacts/unlearning/{model_name}/{sae_name}/results/sparsities/`
    2. for each hyperparameter set, the eval results will be saved at `artifacts/unlearning/{model_name}/{sae_name}/results/metrics` as `.pkl` files
    3. The standard results json will be saved to `eval_results/unlearning/{sae_name}.json`, which contains the unlearning score.
* use `example.ipynb` to combine the sweeping metrics and retrieve one scalar (the unlearning score) for each SAE
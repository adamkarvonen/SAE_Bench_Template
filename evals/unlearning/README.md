### Setup
1. request the forget corpus from this [link](https://docs.google.com/forms/d/e/1FAIpQLSdnQc8Qn0ozSDu3VE8HLoHPvhpukX1t1dIwE5K5rJw9lnOjKw/viewform)
2. you will get one file: `bio-forget-corpus.jsonl`, place it the `evals/unlearning/data` directory
3. see [this page](https://huggingface.co/datasets/cais/wmdp-corpora) for more details

### Eval
* after executing `main.py`, the following will happen:
    1. the feature sparsity for the forget and retain dataset will be saved at `results/sparsities/`
    2. for each hyperparameter set, the eval results will be saved at `results/metrics` as `.pkl` files
* use `example.ipynb` to combine the sweeping metrics and retrieve one scalar (the unlearning score) for each SAE
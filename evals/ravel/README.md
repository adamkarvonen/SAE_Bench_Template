## RAVEL Benchmark, adapted for SAEs

#### TLDR 
RAVEL quantifies feature disentanglement. Given a dataset of entities (eg. cities) with multiple attributes (eg. country, language) we score an SAE's ability to have a precise causal effect on one of the attributes while leaving other attributes unaffected. The current form computes the disentanglement of two attributes `A` and `B` from a single file.

#### Structure
The scoring consists of three steps:
1. Create a `RAVELInstance`, a dataset of Entity-Attribute pairs filterd to only contain pairs the model actually knows. The `RAVELInstance.prompts` contain tokenized prompts and more metadata. See `instance.py` and `generation.py`.
2. Select attribute-sprecific SAE latens by choosing top weights from a single probe trained on SAE latents to predict the presence of the attribute. See `uniprobe.py`
3. Compute cause and isolation scores by intervening on attribute specific features
    - Cause evaluation: High accuracy if intervening with A_features is successful on base_A_template, ie. source_A_attribute_value is generated.
    - Isolation evaluation: High accuracy if intervening with B_features is unsuccessful on base_A_template, ie. base_A_attribute is generated regardless of intervention.
    - disentanglement_score is the mean: D = (cause_A[t] + cause_B[t] + isolation_AtoB[t] + isolation_BtoA[t]) / 4
    - see `intervention.py`

#### Open TODOs
- [ ] move hardcoded hyperparameters from all files to eval config
- [ ] test on multiple SAEs
- [ ] test on pythia
- [ ] test on bigger datset, multiple entities, multiple attributes
- [ ] format result file s.t. it’s easily graphable
- [ ] Doublecheck BOS token handling in `uniprobe.py` 
- [ ] Doublecheck the dataset selection used in the paper for training the probe in `uniprobe.py` 
- [ ] save artifacts
- [ ] write tests
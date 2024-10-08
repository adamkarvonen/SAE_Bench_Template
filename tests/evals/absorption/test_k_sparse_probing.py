import torch
from sae_lens import SAE

from evals.absorption.k_sparse_probing import (
    _get_sae_acts,
    eval_probe_and_sae_k_sparse_raw_scores,
    train_k_sparse_probes,
    train_sparse_multi_probe,
)
from evals.absorption.probing import LinearProbe
from evals.absorption.vocab import LETTERS


def test_train_sparse_multi_probe_results_in_many_zero_weights():
    x = torch.rand(1000, 100)
    y = torch.randint(2, (1000, 3))
    probe1 = train_sparse_multi_probe(x, y, l1_decay=0.01, device=torch.device("cpu"))
    probe2 = train_sparse_multi_probe(x, y, l1_decay=0.1, device=torch.device("cpu"))

    probe1_zero_weights = (probe1.weights.abs() < 1e-5).sum()
    probe2_zero_weights = (probe2.weights.abs() < 1e-5).sum()

    assert probe1_zero_weights > 0
    assert probe2_zero_weights > 0
    assert probe2_zero_weights > probe1_zero_weights


def test_train_k_sparse_probes_returns_reasonable_values(gpt2_l4_sae: SAE):
    train_labels = [("aaa", 0), ("bbb", 1), ("ccc", 2)]
    train_activations = torch.randn(3, 768)
    probes = train_k_sparse_probes(
        gpt2_l4_sae,
        train_labels,
        train_activations,
        ks=[1, 2, 3],
    )
    assert probes.keys() == {1, 2, 3}
    for k, k_probes in probes.items():
        assert k_probes.keys() == {0, 1, 2}
        for probe in k_probes.values():
            assert probe.weight.shape == (k,)
            assert probe.feature_ids.shape == (k,)
            assert probe.k == k


def test_get_sae_acts(gpt2_l4_sae: SAE):
    token_act = torch.randn(768)
    sae_acts = _get_sae_acts(gpt2_l4_sae, token_act.unsqueeze(0), True).squeeze()
    assert sae_acts.shape == (24576,)


def test_get_sae_acts_gives_same_results_batched_and_not_batched(gpt2_l4_sae: SAE):
    token_acts = torch.randn(10, 768)
    sae_acts_unbatched = _get_sae_acts(gpt2_l4_sae, token_acts, True, batch_size=1)
    sae_acts_batched = _get_sae_acts(gpt2_l4_sae, token_acts, True, batch_size=5)
    assert torch.allclose(sae_acts_unbatched, sae_acts_batched, atol=1e-3)


def test_eval_probe_and_sae_k_sparse_raw_scores_gives_sane_results(gpt2_l4_sae: SAE):
    fake_probe = LinearProbe(768, 26)
    eval_data = [(letter, i) for i, letter in enumerate(LETTERS)]
    eval_activations = torch.randn(26, 768)
    k_sparse_probes = train_k_sparse_probes(
        gpt2_l4_sae,
        eval_data,
        eval_activations,
        ks=[1, 2, 3],
    )
    df = eval_probe_and_sae_k_sparse_raw_scores(
        gpt2_l4_sae,
        fake_probe,
        k_sparse_probes,
        eval_data,
        eval_activations,
    )
    expected_columns = [
        "token",
        "answer_letter",
    ]
    for letter in LETTERS:
        expected_columns.append(f"score_probe_{letter}")
        for k in [1, 2, 3]:
            expected_columns.append(f"score_sparse_sae_{letter}_k_{k}")
            expected_columns.append(f"sum_sparse_sae_{letter}_k_{k}")
            expected_columns.append(f"sparse_sae_{letter}_k_{k}_acts")
    assert set(df.columns.values.tolist()) == set(expected_columns)

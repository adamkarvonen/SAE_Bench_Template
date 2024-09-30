from dataclasses import dataclass, field
import torch


@dataclass
class EvalConfig:
    random_seed: int = 42
    model_dtype: torch.dtype = torch.bfloat16

    dataset_name: str = "bias_in_bios"
    chosen_classes: list[str] = field(default_factory=lambda: ["0", "1", "2", "6", "9"])

    probe_train_set_size: int = 4000
    probe_test_set_size: int = 1000
    context_length: int = 128

    probe_batch_size: int = 250
    epochs: int = 10
    lr: float = 1e-3

    sae_batch_size: int = 125

    sae_release: str = "sae_bench_pythia70m_sweep_topk_ctx128_0730"
    model_name: str = "pythia-70m-deduped"
    layers: list[int] = field(default_factory=lambda: [4])
    trainer_ids: list[int] = field(default_factory=lambda: list(range(20)))

    # Uncomment to run Gemma SAEs

    sae_release: str = "sae_bench_gemma-2-2b_sweep_topk_ctx128_ef8_0824"
    model_name: str = "gemma-2-2b"
    layers: list[int] = field(default_factory=lambda: [19])
    trainer_ids: list[int] = field(default_factory=lambda: list(range(6)))

    k_values: list[int] = field(default_factory=lambda: [1, 2, 5, 10, 20, 50, 100])

    saes: list[str] = field(default_factory=list)

    def __post_init__(self):
        for layer in self.layers:
            for trainer_id in self.trainer_ids:
                sae_name = f"{self.sae_release}/resid_post_layer_{layer}/trainer_{trainer_id}"
                sae_name = sae_name.replace("sae_bench_", "")
                self.saes.append(sae_name)

        print("SAEs: ", self.saes)

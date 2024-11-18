There are a few requirements for the SAE object.

- It must have `encode()`, `decode()`, and `forward()` methods.
- The evals of SCR, TPP, and feature absorption require a `W_dec`, which is an nn.Parameter initialized with the following shape: `self.W_dec = nn.Parameter(torch.zeros(d_sae, d_model))`.
- The SAE must have a `dtype` and `device` attribute.
- The SAE must have a `.cfg` field, which contains attributes like `d_sae` and `d_in`. The core evals utilize SAE Lens internals, and require a handful of blank fields, which are already set in the `CustomSaeConfig` dataclass.
- In general, just pattern match to the `pca_sae` and `jump_relu` implementations for how to add the config and other fields.

If there are any pain points when using this repo with custom SAEs, please do not hesitate to reach out or raise an issue.
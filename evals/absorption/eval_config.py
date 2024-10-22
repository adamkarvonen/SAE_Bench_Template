from dataclasses import dataclass, field
from typing import Optional

@dataclass
class EvalConfig:
    random_seed: int = 42
    f1_jump_threshold: float = 0.03
    max_k_value: int = 10

    # double-check token_pos matches prompting_template for other tokenizers
    prompt_template: str = "{word} has the first letter:"
    prompt_token_pos: int = -6

    model_name: str = "pythia-70m-deduped"
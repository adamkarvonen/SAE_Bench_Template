from dataclasses import dataclass

@dataclass
class EvalConfig():
    model_name = 'pythia-70m-deduped'

    layer = 4
    hook_name = f'blocks.{layer}.hook_resid_post'
    
    context_length = 16
    sae_batch_size = 512
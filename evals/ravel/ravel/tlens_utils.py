from common_imports import *
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Literal
from collections import OrderedDict
from torch import Tensor
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint
from transformer_lens import utils as tl_utils
from functools import partial

class Node:
    """
    Mostly a copy of the one in path_patching.py, we'll see if it diverges
    """

    def __init__(
        self,
        component_name: Literal[
            "z",
            "attn_out",
            "pre",
            "post",
            "mlp_out",
            "resid_pre",
            "resid_post",
            "resid_mid",
            "q",
            "k",
            "v",
            "pattern",
            "attn_scores",
            "result",
            "q_input",
            "k_input",
            "v_input",
            'scale_ln1',
            'scale_ln2',
            'scale_final',
            "ln_final",
        ],
        layer: Optional[int] = None,
        head: Optional[int] = None,
        neuron: Optional[int] = None,
        seq_pos: Optional[Union[int, str]] = None, # string used for semantic indexing
    ):
        assert isinstance(component_name, str)
        self.component_name = component_name
        if layer is not None:
            assert isinstance(layer, int)
        self.layer = layer
        if head is not None:
            assert isinstance(head, int)
        self.head = head
        if neuron is not None:
            assert isinstance(neuron, int)
        self.neuron = neuron
        if seq_pos is not None:
            assert isinstance(seq_pos, (int, str))
        self.seq_pos = seq_pos
    
    def __get_mandala_dict__(self) -> Dict[str, Any]:
        return self.__dict__

    def __hash__(self) -> int:
        return hash((self.component_name, self.layer, self.head, self.neuron, self.seq_pos))
    
    def __lt__(self, other: 'Node') -> bool:
        return hash(self) < hash(other)
    
    def __eq__(self, other: 'Node') -> bool:
        return hash(self) == hash(other)
    
    def __le__(self, other: 'Node') -> bool:
        return hash(self) <= hash(other)

    @property
    def activation_name(self) -> str:
        if self.component_name == 'scale_ln1':
            return tl_utils.get_act_name('scale', layer=self.layer, layer_type='ln1')
        elif self.component_name == 'scale_ln2':
            return tl_utils.get_act_name('scale', layer=self.layer, layer_type='ln2')
        elif self.component_name == 'scale_final':
             return tl_utils.get_act_name('scale', layer=None)
        else:
            return tl_utils.get_act_name(self.component_name, layer=self.layer)

    @property
    def shape_type(self) -> List[str]:
        """
        List of the meaning of each dimension of the full activation for this
        node (i.e., what you'd get if you did `cache[self.activation_name]`).
        
        This is just for reference
        """
        if self.component_name in [
            "resid_pre",
            "resid_post",
            "resid_mid",
            "q_input",
            "k_input",
            "v_input",
        ]:
            return ["batch", "seq", "d_model"]
        elif self.component_name == 'pattern':
            return ["batch", "head", "query_pos", "key_pos"]
        elif self.component_name in ["q", "k", "v", "z"]:
            return ["batch", "seq", "head", "d_head"]
        elif self.component_name in ["result"]:
            return ["batch", "seq", "head", "d_model"]
        elif self.component_name == 'scale':
            return ['batch', 'seq']
        elif self.component_name == 'post':
            return ['batch', 'seq', 'd_mlp']
        else:
            raise NotImplementedError

    def idx(self) -> Tuple[Union[int, slice, Tensor, None], ...]:
        """
        Index into the full activation to restrict to layer / head / neuron /
        seq_pos
        """
        if isinstance(self.seq_pos, int):
            seq_pos_idx = self.seq_pos
            batch_idx = slice(None)
        elif self.seq_pos is None:
            seq_pos_idx = slice(None)
            batch_idx = slice(None)
        else:
            raise NotImplementedError

        if self.neuron is not None:
            raise NotImplementedError

        elif self.component_name in ['pattern', 'attn_scores']:
            assert self.head is not None
            return tuple([slice(None), self.head, slice(None), slice(None)])
        elif self.component_name in ["q", "k", "v", "z", "result"]:
            assert self.head is not None, "head must be specified for this component"
            return tuple([batch_idx, seq_pos_idx, self.head, slice(None)])
        elif self.component_name == 'scale':
            return tuple([slice(None), slice(None)])
        elif self.component_name == 'post':
            return tuple([batch_idx, seq_pos_idx, slice(None)])
        else:
            return tuple([batch_idx, seq_pos_idx, slice(None)])
    
    @property
    def names_filter(self) -> Callable:
        return lambda x: x in [self.activation_name]
    
    @staticmethod
    def get_names_filter(nodes: List['Node']) -> Callable:
        return lambda x: any(node.names_filter(x) for node in nodes)

    @property
    def needs_head_results(self) -> bool:
        return self.component_name in ['result']
    
    def get_value(self, cache: ActivationCache,) -> Tensor:
        return cache[self.activation_name][self.idx()]
    
    def __repr__(self) -> str:
        properties = OrderedDict({
            "component_name": self.component_name,
            "layer": self.layer,
            "head": self.head,
            "neuron": self.neuron,
            "seq_pos": self.seq_pos,
        })
        properties = ", ".join(f"{k}={v}" for k, v in properties.items() if v is not None)
        return f"Node({properties})"
    
    @property
    def displayname(self) -> str:
        if self.component_name in ('q', 'k', 'v', 'z'):
            return f'{self.component_name}@L{self.layer}H{self.head}@{self.seq_pos}'
        elif self.component_name == 'resid_post':
            if self.seq_pos is not None:
                return f'resid_post@L{self.layer}@{self.seq_pos}'
            else:
                return f'resid_post@L{self.layer}'
        else:
            raise NotImplementedError




def get_single_token_cf_hook(node: Node, positions: Tensor, A_cf: Tensor) -> Tuple[str, Callable]:
    """
    Get a hook that patches a counterfactual activation at a given position in
    the sequence. A_cf must be already indexed into, i.e. of shape (batch,
    d_node).
    """
    assert node.seq_pos is None, "This hook requires seq_pos=None"
    def hook_fn(activation: Tensor, hook: HookPoint) -> Tensor:
        # activation will be of shape (num_texts, seq_len, d_act)
        batch_size = activation.shape[0]
        activation[range(batch_size), positions, :] = A_cf.clone().to(activation.device).to(activation.dtype)
        return activation
    return (node.activation_name, hook_fn)

def run_with_cache(
    prompts_or_tokens: Union[List[str], Tensor],
    nodes: List[Node],
    batch_size: Optional[int],
    model: HookedTransformer,
    padding_side: Optional[Literal['left', 'right']] = None,
    verbose: bool = False,
    return_logits: bool = False,
    offload_to_cpu: bool = False,
    clear_cache: bool = False,
    out_device: Optional[Union[str, torch.device]] = None,
    prepend_bos: bool = True,
    required_length: Optional[int] = None,
) -> List[Tensor]:
    """
    Run the model on the given prompts, and return the activations for the
    given nodes.
    """
    model_device = next(model.parameters()).device
    if isinstance(prompts_or_tokens, Tensor):
        tokens = prompts_or_tokens
    else:
        tokens = model.to_tokens(prompts_or_tokens, prepend_bos=prepend_bos, padding_side=padding_side)
        tokens = tokens.to(model_device)
    if required_length is not None:
        # pad the tokens to the required length with BOS either on the left or right
        assert required_length >= tokens.shape[1]
        bos_token_id = model.tokenizer.bos_token_id
        if padding_side == 'left':
            tokens = torch.cat([torch.full((tokens.shape[0], required_length - tokens.shape[1]), bos_token_id, dtype=tokens.dtype, device=tokens.device), tokens], dim=1)
        elif padding_side in ['right', None]:
            tokens = torch.cat([tokens, torch.full((tokens.shape[0], required_length - tokens.shape[1]), bos_token_id, dtype=tokens.dtype, device=tokens.device)], dim=1)

    logits, cache = model.run_with_cache(tokens, names_filter=Node.get_names_filter(nodes))
    acts = [node.get_value(cache) for node in nodes]
    if offload_to_cpu:
        raise NotImplementedError()
    if clear_cache:
        raise NotImplementedError()
    if out_device is not None:
        acts = [act.to(out_device) for act in acts]
        logits = logits.to(out_device)
    if return_logits:
        return acts, logits
    else:
        return acts

def run_with_hooks(
    prompts: Any, 
    hooks: Optional[List[Tuple[str, Callable]]],
    batch_size: int,
    model: HookedTransformer,
    return_predictions: bool = False,
    semantic_nodes: Optional[List[Node]] = None,
    semantic_hooks: Optional[List[Tuple[str, Callable]]] = None,
    return_full_logits: bool = False,
    return_full_last_logits: bool = True,
    prepend_bos: bool = True,
    padding_side: Optional[Literal['left', 'right']] = None,
    verbose: bool = False,
    out_device: Optional[Union[str, torch.device]] = None,
) -> Tensor:
    tokens = model.to_tokens(prompts, prepend_bos=prepend_bos, padding_side=padding_side)
    assert (semantic_hooks is None) == (semantic_nodes is None)
    if semantic_nodes is not None:
        assert hooks is None
        assert semantic_hooks is not None
        hooks = []
        idxs_by_semantic_pos = {k: [p.semantic_pos[k] for p in prompts] for k in prompts[0].semantic_pos.keys()}
        for node, hook in zip(semantic_nodes, semantic_hooks):
            hooks.append((hook[0], partial(hook[1], idx=node.idx(prompts=prompts))))
    model.reset_hooks()
    logits = model.run_with_hooks(tokens, fwd_hooks=hooks)
    if return_full_logits:
        res = logits
    elif return_full_last_logits:
        res = logits[:, -1, :]
    elif return_predictions:
        res = logits[:, -1, :].argmax(dim=-1)
    else:
        raise NotImplementedError()
    if out_device is not None:
        res = res.to(out_device)
    return res


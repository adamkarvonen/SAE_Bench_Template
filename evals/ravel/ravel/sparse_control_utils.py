from common_imports import *
from collections import namedtuple
from typing import List, Tuple, NamedTuple
from sae_lens.sae import SAE as SAELensSAE


################################################################################
### RAVEL tools
################################################################################
from ravel_dataset_builder import Prompt
from utils.intervention_utils import find_positions_in_tokens

def get_entity_positions(prompt: Prompt, model: HookedTransformer) -> List[int]:
    t, e = prompt.text, prompt.entity
    start = t.find(e)
    stop = start + len(e) - 1

    positions = find_positions_in_tokens(
        start=start, stop=stop,
        tokens=model.to_str_tokens(t),
    )

    return positions





################################################################################
### generic tools for diffing SAE latents
################################################################################
class SAEDiffData(NamedTuple):
    """
    *Signed* data for the differences between two SAE latents, with positive signs
    w.r.t. the counterfactual prompt side. This convention is used to make it
    easy to construct the overall vector to add to the original activation 
    to steer it towards the counterfactual.
    """
    R_diff: torch.Tensor # cf - og reconstructions
    og_minus_cf_indices: List[torch.Tensor] # indices of features active in og but not cf
    og_minus_cf_summands: List[torch.Tensor] # - W_dec[i] * vals[i] for i in og_minus_cf_indices NOTE THE NEGATIVE SIGN
    cf_minus_og_indices: List[torch.Tensor] # indices of features active in cf but not og
    cf_minus_og_summands: List[torch.Tensor] # W_dec[i] * vals[i] for i in cf_minus_og_indices
    og_and_cf_indices: List[torch.Tensor] # indices of shared features ON THE OG SIDE
    og_and_cf_summands: List[torch.Tensor] # W_dec[i] * (vals_cf[i] - vals_og[i]) for i in og_and_cf_indices NOTE THE DIRECTION OF SUBTRACTION

    def get_concat_summands(self) -> List[Tensor]:
        """
        Return concatenated summands in the order (cf-og, og&cf, og-cf).

        This is useful when we want to run an intervention that can use any of
        these summands.
        """
        return [torch.cat([x, y, z], dim=0) for x, y, z in zip(self.cf_minus_og_summands, self.og_and_cf_summands, self.og_minus_cf_summands)]
    
    def check_scores(self, atol: float = 1e-3) -> None:
        # use this to test the object by checking that the scores sum to 1
        s = get_sae_diff_scores(diff=self, method='proj')
        # it must be that the coefficients sum to 1
        checksums = [x.sum() + y.sum() + z.sum() for x, y, z in zip(s.og_minus_cf_scores, s.cf_minus_og_scores, s.og_and_cf_scores)]
        assert all(torch.allclose(x, torch.ones_like(x), atol=atol) for x in checksums), checksums
    
    def __get_mandala_dict__(self) -> dict:
        return self.__dict__


def get_topk_sae_activations(
    encoder: SAELensSAE,
    A: Tensor,) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Return the indices and values of the topK features, and the overall
    reconstruction.
    """
    assert encoder.activation_fn.__class__.__name__ == 'TopK'
    A_enc = encoder.encode(A)
    k = encoder.activation_fn.k
    topk_outcome = torch.topk(A_enc, k, dim=-1)
    idxs, vals = topk_outcome.indices, topk_outcome.values
    R = encoder.decode(A_enc)
    return idxs, vals, R


@torch.no_grad()
def get_sae_diff_data(
    A_og: torch.Tensor, # (batch, d_act)
    A_cf: torch.Tensor, # (batch, d_act)
    encoder: SAELensSAE, # the encoder to use
) -> SAEDiffData:
    """
    Given the SAE latents for two activations, compute the data describing how
    the latents differ, in a way useful for downstream analysis & interventions.

    NOTE: this is tricky because we have to be careful about the shared features
    being at different indices in the two activations. This is why there's a 
    test added to the function to verify that the scores sum to 1.
    """
    ### prepare the things we need
    # out_og = encoder.forward(A_og)
    # out_cf = encoder.forward(A_cf)
    # idxs_og, vals_og, R_og = out_og.latent_indices, out_og.latent_acts, out_og.sae_out
    # idxs_cf, vals_cf, R_cf = out_cf.latent_indices, out_cf.latent_acts, out_cf.sae_out
    idxs_og, vals_og, R_og = get_topk_sae_activations(encoder, A_og)
    idxs_cf, vals_cf, R_cf = get_topk_sae_activations(encoder, A_cf)
    # A_diff = A_cf - A_og # we always subtract og from cf
    R_diff = R_cf - R_og
    W_dec = encoder.W_dec.detach().clone() # shape (d_hidden, d_act)

    ### find differing features and their values
    n_examples = A_og.shape[0]
    # masks for subsets
    og_minus_cf_masks = [~torch.isin(idxs_og[i], idxs_cf[i]) for i in range(n_examples)]
    cf_minus_og_masks = [~torch.isin(idxs_cf[i], idxs_og[i]) for i in range(n_examples)]
    # we need two masks for the shared features, one from the point of view of og and one from the point of view of cf
    og_and_cf_masks_og_side = [torch.isin(idxs_og[i], idxs_cf[i]) for i in range(n_examples)] 
    og_and_cf_masks_cf_side = [torch.isin(idxs_cf[i], idxs_og[i]) for i in range(n_examples)]
    # active indices for subsets
    og_minus_cf_indices = [idxs_og[i][og_minus_cf_masks[i]] for i in range(n_examples)]
    cf_minus_og_indices = [idxs_cf[i][cf_minus_og_masks[i]] for i in range(n_examples)]
    og_and_cf_indices = [idxs_og[i][og_and_cf_masks_og_side[i]] for i in range(n_examples)]
    # latent activations for subsets 
    og_minus_cf_vals = [vals_og[i][og_minus_cf_masks[i]] for i in range(n_examples)]
    cf_minus_og_vals = [vals_cf[i][cf_minus_og_masks[i]] for i in range(n_examples)]
    # here, we always subtract og from cf to keep with the convention
    # we have to be careful that we subtract corresponding values!!!
    # og_and_cf_vals = [vals_cf[i][og_and_cf_masks_cf_side[i]] - vals_og[i][og_and_cf_masks_og_side[i]] for i in range(n_examples)]
    og_and_cf_vals = []
    for i in range(n_examples):
        og_and_cf_vals_i = []
        for idx in og_and_cf_indices[i]:
            og_and_cf_vals_i.append(vals_cf[i][idxs_cf[i] == idx] - vals_og[i][idxs_og[i] == idx])
        if og_and_cf_vals_i:
            og_and_cf_vals.append(torch.cat(og_and_cf_vals_i, dim=0))
        else:
            og_and_cf_vals.append(torch.tensor([]).to(vals_og[i].device))
    # compute the vectors for the differing features
    og_minus_cf_summands = [- W_dec[indices, :] * vals.unsqueeze(-1) for indices, vals in zip(og_minus_cf_indices, og_minus_cf_vals)]
    cf_minus_og_summands = [W_dec[indices, :] * vals.unsqueeze(-1) for indices, vals in zip(cf_minus_og_indices, cf_minus_og_vals)]
    og_and_cf_summands = [W_dec[indices, :] * vals.unsqueeze(-1) for indices, vals in zip(og_and_cf_indices, og_and_cf_vals)]

    res = {
        'R_diff': R_diff,
        # 'og_minus_cf_scores': og_minus_cf_scores,
        'og_minus_cf_indices': og_minus_cf_indices,
        'og_minus_cf_summands': og_minus_cf_summands,
        # 'cf_minus_og_scores': cf_minus_og_scores,
        'cf_minus_og_indices': cf_minus_og_indices,
        'cf_minus_og_summands': cf_minus_og_summands,
        # 'og_and_cf_scores': og_and_cf_scores,
        'og_and_cf_indices': og_and_cf_indices,
        'og_and_cf_summands': og_and_cf_summands,
    }
    res = SAEDiffData(**res)
    res.check_scores()
    return res


class SAEDiffScores(NamedTuple):
    """
    NOTE: because the number of features in og_minus_cf, etc. can be different
    depending on the example, we keep them as lists of tensors.
    """

    og_minus_cf_scores: List[torch.Tensor]
    cf_minus_og_scores: List[torch.Tensor]
    og_and_cf_scores: List[torch.Tensor]

    def get_concat_scores(self) -> List[Tensor]:
        """
        Return concatenated scores in the order (cf-og, og&cf, og-cf)
        """
        return [torch.cat([x, y, z], dim=0) for x, y, z in zip(self.cf_minus_og_scores, self.og_and_cf_scores, self.og_minus_cf_scores)]

    @staticmethod
    def measure_sparsity(x: torch.Tensor, threshold=0.8) -> float:
        """
        1 minus fraction of coordinates we need to capture at least `threshold`
        of the sum of the absolute values of the tensor `x`.
        """
        fracs = x.abs() / x.abs().sum()
        sorted_fracs, sorted_idxs = fracs.sort(descending=True)
        cumsum = sorted_fracs.cumsum(dim=0)
        i = 0
        while cumsum[i] < threshold:
            i += 1
        return 1 - ((i + 1) / len(x))
    

@torch.no_grad()
def get_sae_diff_scores(diff: SAEDiffData, method: str = 'proj') -> SAEDiffScores:
    """
    Given the data of diffs between SAE latents on pairs of
    original/counterfactual activations, compute the contribution of each 
    latent to the difference of reconstructions.

    Return some quantities of interest about the differing features:
    - suppose the reconstructions are R_og and R_cf
    - the difference R_diff = R_cf - R_og = 
        sum_{i in og and cf} W_dec[i] * (vals_cf[i] - vals_og[i]) \  # shared features term
        + sum_{i in cf_minus_og} W_dec[i] * vals_cf[i] \ # cf minus og term
        - sum_{i in og_minus_cf} W_dec[i] * vals_og[i] - # og minus cf term
    - we can normalize by projecting on R_diff and dividing by ||R_diff||_2^2.
    - then all things in the sum add up to 1.
    - we can ask what part of the sum is due to the shared features, and what part
    is due to the differing features.
    - we can score each feature by the part of the sum it contributes to.
    """
    ### compute scores
    og_minus_cf_scores = [x @ r_diff / r_diff.norm()**2 for x, r_diff in zip(diff.og_minus_cf_summands, diff.R_diff)]
    cf_minus_og_scores = [x @ r_diff / r_diff.norm()**2 for x, r_diff in zip(diff.cf_minus_og_summands, diff.R_diff)]
    og_and_cf_scores = [x @ r_diff / r_diff.norm()**2 for x, r_diff in zip(diff.og_and_cf_summands, diff.R_diff)]
    res = {
        'og_minus_cf_scores': og_minus_cf_scores,
        'cf_minus_og_scores': cf_minus_og_scores,
        'og_and_cf_scores': og_and_cf_scores,
    }
    return SAEDiffScores(**res)

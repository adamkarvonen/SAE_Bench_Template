"""Define a featurizer and features with Differential Binary Masking."""

# import pyvene as pv
import torch
from torch import nn


class DifferentialBinaryMasking(nn.Module):
  """
  Generic re-implementation of the Differential Binary Masking (DBM) method
  used in RAVEL. The DBM method is a feature selection method that uses a
  binary mask to select features from a model representation. The mask is
  parameterized by a temperature parameter, which is used to control the
  sparsity of the mask. The mask is applied to the model representation using a
  simple interchange intervention, where the selected features are replaced by
  the corresponding features from a source representation.
  """

  def __init__(self, embed_dim,):
    super().__init__()
    self.mask = torch.nn.Parameter(torch.zeros(embed_dim), requires_grad=True)
    # self.temperature = torch.nn.Parameter(torch.tensor(1e-2)) # WHY is this a parameter if it never gets gradients?
    self.temperature = torch.tensor(1e-2)
    # The dimension of model representations.
    self.embed_dim = embed_dim
    self.rotate_layer = torch.nn.Linear(embed_dim, embed_dim, bias=False)
    self.rotate_layer.weight.requires_grad = False
    # The featurizer is equivalent to an identity matrix.
    with torch.no_grad():
      self.rotate_layer.weight.copy_(torch.eye(self.embed_dim))

  def get_temperature(self):
    return self.temperature

  def set_temperature(self, temp: torch.Tensor):
    self.temperature.data = temp

  def forward(self, base, source, subspaces=None):
    input_dtype, model_dtype = base.dtype, self.mask.dtype
    base, source = base.to(model_dtype), source.to(model_dtype)
    batch_size = base.shape[0]
    if self.training:
      mask_sigmoid = torch.sigmoid(self.mask / torch.tensor(self.temperature))
      # Save the selected features, i.e., where sigmoid(mask) > 0.5 as the
      # rotation matrix, so that at inference time, we only need to load the
      # rotation matrix.
      with torch.no_grad():
        if torch.any(mask_sigmoid > 0.5):
          rotate_matrix = torch.masked_select(
              torch.eye(self.embed_dim, device=base.device),
              (mask_sigmoid > 0.5).view([-1, 1])).view([-1, self.embed_dim])
          self.rotate_layer = torch.nn.Linear(self.embed_dim,
                                              rotate_matrix.shape[0],
                                              bias=False,
                                              device=base.device)
          self.rotate_layer.weight.copy_(rotate_matrix)
      # Apply interchange interventions.
      output = (1.0 - mask_sigmoid) * base + mask_sigmoid * source
    else:
      rotated_base = self.rotate_layer(base)
      rotated_source = self.rotate_layer(source)
      # Apply interchange interventions.
      output = base + torch.matmul(
          (rotated_source - rotated_base), self.rotate_layer.weight)
    return output.to(input_dtype)

  def get_sparsity_loss(self):
    mask_sigmoid = torch.sigmoid(self.mask / torch.tensor(self.temperature))
    return torch.norm(mask_sigmoid, p=1)
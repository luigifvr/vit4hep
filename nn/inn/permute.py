from FrEIA.modules import InvertibleModule

from typing import Union, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PermuteRandomS1(InvertibleModule):
    """Constructs a random permutation, that stays fixed during training.
    Permutes along the first (channel-) dimension for multi-dimenional tensors."""

    def __init__(self, dims_in, dims_c=None, seed: Union[int, None] = None):
        """Additional args in docstring of base class FrEIA.modules.InvertibleModule.

        Args:
          seed: Int seed for the permutation (numpy is used for RNG). If seed is
            None, do not reseed RNG.
        """
        super().__init__(dims_in, dims_c)

        self.in_channels = dims_in[0][1]

        if seed is not None:
            np.random.seed(seed)
        self.perm = np.random.permutation(self.in_channels)

        self.perm_inv = np.zeros_like(self.perm)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i

        self.perm = nn.Parameter(torch.LongTensor(self.perm), requires_grad=False)
        self.perm_inv = nn.Parameter(
            torch.LongTensor(self.perm_inv), requires_grad=False
        )

    def forward(self, x, rev=False, jac=True):
        if not rev:
            return [x[0][:, :, self.perm]], 0.0
        else:
            return [x[0][:, :, self.perm_inv]], 0.0

    def output_dims(self, input_dims):
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        return input_dims


class PermuteRandomS2(InvertibleModule):
    """Constructs a random permutation, that stays fixed during training.
    Permutes along the first (channel-) dimension for multi-dimenional tensors."""

    def __init__(self, dims_in, dims_c=None, seed: Union[int, None] = None):
        """Additional args in docstring of base class FrEIA.modules.InvertibleModule.

        Args:
          seed: Int seed for the permutation (numpy is used for RNG). If seed is
            None, do not reseed RNG.
        """
        super().__init__(dims_in, dims_c)

        self.in_channels = dims_in[0][2]

        if seed is not None:
            np.random.seed(seed)
        self.perm = np.random.permutation(self.in_channels)

        self.perm_inv = np.zeros_like(self.perm)
        for i, p in enumerate(self.perm):
            self.perm_inv[p] = i

        self.perm = nn.Parameter(torch.LongTensor(self.perm), requires_grad=False)
        self.perm_inv = nn.Parameter(
            torch.LongTensor(self.perm_inv), requires_grad=False
        )

    def forward(self, x, rev=False, jac=True):
        if not rev:
            return [x[0][:, :, :, self.perm]], 0.0
        else:
            return [x[0][:, :, :, self.perm_inv]], 0.0

    def output_dims(self, input_dims):
        if len(input_dims) != 1:
            raise ValueError(f"{self.__class__.__name__} can only use 1 input")
        return input_dims

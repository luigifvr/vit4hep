from collections.abc import Callable
from typing import Union

import torch
from FrEIA.modules import InvertibleModule


class BaseCouplingBlock(InvertibleModule):
    """Base class to implement various coupling schemes.  It takes care of
    checking the dimensions, conditions, clamping mechanism, etc.
    Each child class only has to implement the _coupling1 and _coupling2 methods
    for the left and right coupling operations.
    (In some cases below, forward() is also overridden)
    """

    def __init__(
        self,
        dims_in,
        dims_c=[],
        clamp: float = 2.0,
        clamp_activation: str | Callable = "ATAN",
        spatial: bool = False,
    ):
        """
        Additional args in docstring of base class.
        Args:
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        """

        super().__init__(dims_in, dims_c)

        # ndims means the rank of tensor strictly speaking.
        # i.e. 1D, 2D, 3D tensor, etc.
        self.ndims = len(dims_in[0])

        self.clamp = clamp

        # TODO add assertions

        self.conditional = len(dims_c) > 0

        if isinstance(clamp_activation, str):
            if clamp_activation == "ATAN":
                self.f_clamp = lambda u: 0.636 * torch.atan(u)
            elif clamp_activation == "TANH":
                self.f_clamp = torch.tanh
            elif clamp_activation == "SIGMOID":
                self.f_clamp = lambda u: 2.0 * (torch.sigmoid(u) - 0.5)
            else:
                raise ValueError(f'Unknown clamp activation "{clamp_activation}"')
        else:
            self.f_clamp = clamp_activation

    def forward(self, x1, x2, c=[], rev=False, jac=True):
        """See base class docstring"""
        # TODO update notation
        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # *_c: variable with condition concatenated
        # j1, j2: Jacobians of the two coupling operations
        # always the last vector is transformed
        if not rev:
            if self.conditional:
                y1, j1 = self._coupling1(x1, x2, c)
                y2, j2 = self._coupling2(x2, y1, c)
            else:
                y1, j1 = self._coupling1(x1, x2)
                y2, j2 = self._coupling2(x2, y1)
        else:
            # names of x and y are swapped for the reverse computation
            if self.conditional:
                y2, j2 = self._coupling2(x2, x1, c, rev=True)
                y1, j1 = self._coupling1(x1, y2, c, rev=True)
            else:
                y2, j2 = self._coupling2(x2, x1, rev=True)
                y1, j1 = self._coupling1(x1, y2, rev=True)

        return y1, y2, j1 + j2

    def _coupling1(self, x1, u2, c, rev=False):
        """The first/left coupling operation in a two-sided coupling block.
        Args:
          x1 (Tensor): the 'active' half being transformed.
          u2 (Tensor): the 'passive' half, including the conditions, from
            which the transformation is computed.
        Returns:
          y1 (Tensor): same shape as x1, the transformed 'active' half.
          j1 (float or Tensor): the Jacobian, only has batch dimension.
            If the Jacobian is zero of fixed, may also return float.
        """
        raise NotImplementedError()

    def _coupling2(self, x2, u1, c, rev=False):
        """The second/right coupling operation in a two-sided coupling block.
        Args:
          x2 (Tensor): the 'active' half being transformed.
          u1 (Tensor): the 'passive' half, including the conditions, from
            which the transformation is computed.
        Returns:
          y2 (Tensor): same shape as x1, the transformed 'active' half.
          j2 (float or Tensor): the Jacobian, only has batch dimension.
            If the Jacobian is zero of fixed, may also return float.
        """
        raise NotImplementedError()

    def output_dims(self, input_dims):
        """See base class for docstring"""
        if len(input_dims) != 1:
            raise ValueError("Can only use 1 input")
        return input_dims


class OneSidedBaseCouplingBlock(InvertibleModule):
    """Base class to implement various coupling schemes.  It takes care of
    checking the dimensions, conditions, clamping mechanism, etc.
    Each child class only has to implement the _coupling1 and _coupling2 methods
    for the left and right coupling operations.
    (In some cases below, forward() is also overridden)
    """

    def __init__(
        self,
        dims_in,
        dims_c=[],
        clamp: float = 2.0,
        clamp_activation: str | Callable = "ATAN",
    ):
        """
        Additional args in docstring of base class.
        Args:
          clamp: Soft clamping for the multiplicative component. The
            amplification or attenuation of each input dimension can be at most
            exp(±clamp).
          clamp_activation: Function to perform the clamping. String values
            "ATAN", "TANH", and "SIGMOID" are recognized, or a function of
            object can be passed. TANH behaves like the original realNVP paper.
            A custom function should take tensors and map -inf to -1 and +inf to +1.
        """

        super().__init__(dims_in, dims_c)

        # ndims means the rank of tensor strictly speaking.
        # i.e. 1D, 2D, 3D tensor, etc.
        self.ndims = len(dims_in[0])

        self.clamp = clamp

        # TODO add assertions

        self.conditional = len(dims_c) > 0

        if isinstance(clamp_activation, str):
            if clamp_activation == "ATAN":
                self.f_clamp = lambda u: 0.636 * torch.atan(u)
            elif clamp_activation == "TANH":
                self.f_clamp = torch.tanh
            elif clamp_activation == "SIGMOID":
                self.f_clamp = lambda u: 2.0 * (torch.sigmoid(u) - 0.5)
            else:
                raise ValueError(f'Unknown clamp activation "{clamp_activation}"')
        else:
            self.f_clamp = clamp_activation

    def forward(self, x1, x2, c=[], rev=False, jac=True):
        """See base class docstring"""
        # TODO update notation
        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # *_c: variable with condition concatenated
        # j1, j2: Jacobians of the two coupling operations
        # always the last vector is transformed
        if not rev:
            y, j = self._coupling(x1, x2, c)
        else:
            y, j = self._coupling(x1, x2, c, rev=True)

        return y, j

    def _coupling(self, x1, u2, c, rev=False):
        """The first/left coupling operation in a two-sided coupling block.
        Args:
          x1 (Tensor): the 'active' half being transformed.
          u2 (Tensor): the 'passive' half, including the conditions, from
            which the transformation is computed.
        Returns:
          y1 (Tensor): same shape as x1, the transformed 'active' half.
          j1 (float or Tensor): the Jacobian, only has batch dimension.
            If the Jacobian is zero of fixed, may also return float.
        """
        raise NotImplementedError()

    def output_dims(self, input_dims):
        """See base class for docstring"""
        if len(input_dims) != 1:
            raise ValueError("Can only use 1 input")
        return input_dims

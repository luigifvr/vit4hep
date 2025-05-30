from FrEIA.modules import InvertibleModule
from nn.base_calo import BaseCaloCouplingBlock
from nn.base_calo2 import BaseCalo2CouplingBlock
from nn.base_onesided_calo import BaseCalo2OneSideCouplingBlock

from typing import Callable, Union

import torch


class GLOWCaloCouplingBlock(BaseCaloCouplingBlock):
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
        subnet_constructor: Callable = None,
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "ATAN",
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
        super().__init__(dims_in, dims_c, clamp, clamp_activation)

        # No length of conditions needed?
        self.subnet1 = subnet_constructor(
            self.split_len1 + self.split_len2, self.split_len3 * 2
        )  # dims_in, dims_out
        self.subnet2 = subnet_constructor(
            self.split_len1 + self.split_len3, self.split_len2 * 2
        )
        self.subnet3 = subnet_constructor(
            self.split_len3 + self.split_len2, self.split_len1 * 2
        )

    def _coupling1(self, x1, x2, u3, c, rev=False):

        # notation (same for _coupling2):
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True)
        # y: outputs (same scheme)
        # *_c: variables with condition appended
        # *1, *2: left half, right half
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian

        x12 = torch.cat((x1, x2), dim=1)
        a3 = self.subnet1(x12, c)
        s3, t3 = a3[:, 0], a3[:, 1]
        s3 = self.clamp * self.f_clamp(s3)
        j3 = torch.sum(s3, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y3 = (u3 - t3) * torch.exp(-s3)
            return y3, -j3
        else:
            y3 = torch.exp(s3) * u3 + t3
            return y3, j3

    def _coupling2(self, x1, x3, u2, c, rev=False):

        x13 = torch.cat((x1, x3), dim=1)
        a2 = self.subnet2(x13, c)
        s2, t2 = a2[:, 0], a2[:, 1]
        s2 = self.clamp * self.f_clamp(s2)
        j2 = torch.sum(s2, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y2 = (u2 - t2) * torch.exp(-s2)
            return y2, -j2
        else:
            y2 = torch.exp(s2) * u2 + t2
            return y2, j2

    def _coupling3(self, x2, x3, u1, c, rev=False):

        x23 = torch.cat((x2, x3), dim=1)
        a1 = self.subnet3(x23, c)
        s1, t1 = a1[:, 0], a1[:, 1]
        s1 = self.clamp * self.f_clamp(s1)
        j1 = torch.sum(s1, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y1 = (u1 - t1) * torch.exp(-s1)
            return y1, -j1
        else:
            y1 = torch.exp(s1) * u1 + t1
            return y1, j1


class GLOWCalo2CouplingBlock(BaseCalo2CouplingBlock):
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
        subnet_constructor: Callable = None,
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "ATAN",
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
        super().__init__(dims_in, dims_c, clamp, clamp_activation, spatial)

        self.split_len1 = len(self.indices1)
        self.split_len2 = len(self.indices2)
        # No length of conditions needed?
        self.subnet1 = subnet_constructor(
            self.split_len1, self.split_len2 * 2
        )  # dims_in, dims_out
        self.subnet2 = subnet_constructor(self.split_len2, self.split_len1 * 2)

    def _coupling1(self, x1, u2, c, rev=False):

        # notation (same for _coupling2):
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True)
        # y: outputs (same scheme)
        # *_c: variables with condition appended
        # *1, *2: left half, right half
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian
        a2 = self.subnet1(u2, c)
        a2 *= 0.1
        s2, t2 = a2[:, 0], a2[:, 1]
        s2 = self.clamp * self.f_clamp(s2)
        j1 = torch.sum(s2, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y1 = (x1 - t2) * torch.exp(-s2)
            # y1 = x1 * torch.exp(-s2) - t2
            return y1, -j1  # TODO Why -j2 does not give errors??
        else:
            y1 = torch.exp(s2) * x1 + t2
            # y1 = (x1 + t2)*torch.exp(s2)
            return y1, j1

    def _coupling2(self, x2, u1, c, rev=False):

        a1 = self.subnet2(u1, c)
        a1 *= 0.1
        s1, t1 = a1[:, 0], a1[:, 1]
        s1 = self.clamp * self.f_clamp(s1)
        j2 = torch.sum(s1, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y2 = (x2 - t1) * torch.exp(-s1)
            return y2, -j2
        else:
            y2 = torch.exp(s1) * x2 + t1
            return y2, j2


class GLOWCalo2OneSideCouplingBlock(BaseCalo2OneSideCouplingBlock):
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
        subnet_constructor: Callable = None,
        clamp: float = 2.0,
        clamp_activation: Union[str, Callable] = "ATAN",
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
        super().__init__(dims_in, dims_c, clamp, clamp_activation, spatial)

        self.split_len1 = len(self.indices1)
        self.split_len2 = len(self.indices2)
        # No length of conditions needed?
        self.subnet1 = subnet_constructor(
            self.split_len1, self.split_len2 * 2
        )  # dims_in, dims_out

    def _coupling1(self, x1, u2, c, rev=False):

        # notation (same for _coupling2):
        # x: inputs (i.e. 'x-side' when rev is False, 'z-side' when rev is True)
        # y: outputs (same scheme)
        # *_c: variables with condition appended
        # *1, *2: left half, right half
        # a: all affine coefficients
        # s, t: multiplicative and additive coefficients
        # j: log det Jacobian
        a2 = self.subnet1(u2, c)
        a2 *= 0.1
        s2, t2 = a2[:, 0], a2[:, 1]
        s2 = self.clamp * self.f_clamp(s2)
        j1 = torch.sum(s2, dim=tuple(range(1, self.ndims + 1)))

        if rev:
            y1 = (x1 - t2) * torch.exp(-s2)
            # y1 = x1 * torch.exp(-s2) - t2
            return y1, -j1  # TODO Why -j2 does not give errors??
        else:
            y1 = torch.exp(s2) * x1 + t2
            # y1 = (x1 + t2)*torch.exp(s2)
            return y1, j1

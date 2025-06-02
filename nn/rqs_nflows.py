from typing import Callable

import math
import numpy as np
from scipy.stats import special_ortho_group
import torch
import torch.nn as nn
import torch.nn.functional as F
from nn.base_coupling import BaseCouplingBlock, OneSidedBaseCouplingBlock
from FrEIA.modules import InvertibleModule
from FrEIA import utils


class CaloRationalQuadraticSplineBlock(BaseCouplingBlock):
    def __init__(
        self,
        dims_in,
        dims_c,
        subnet_constructor,
        num_bins=10,
        bounds_init=1.0,
        tails="linear",
        bounds_type="SOFTPLUS",
        spatial=False,
        *args
    ):
        super().__init__(dims_in, dims_c, *args)

        self.spatial = spatial
        if spatial:
            self.channels = dims_in[0][1]
            self.patch_dim = self.channels // 2
            self.num_patches = dims_in[0][0]
        else:
            self.channels = dims_in[0][0]
            self.patch_dim = dims_in[0][1]
            self.num_patches = self.channels // 2

        # ndims means the rank of tensor strictly speaking.
        # i.e. 1D, 2D, 3D tensor, etc.
        self.ndims = len(dims_in[0])

        if self.spatial:
            # self.indices1 = [int(k) for k in range(self.channels // 2)]
            # self.indices2 = [
            #    int(k + self.channels // 2) for k in range(self.channels // 2)
            # ]
            self.indices1 = [int(2 * k) for k in range(self.channels // 2)]
            self.indices2 = [int(2 * k + 1) for k in range(self.channels // 2)]
        else:
            self.indices1 = [int(2 * k) for k in range(self.channels // 2)]
            self.indices2 = [int(2 * k + 1) for k in range(self.channels // 2)]
            # self.indices1 = [int(k) for k in range(self.channels // 2)]
            # self.indices2 = [
            #    int(k + self.channels // 2) for k in range(self.channels // 2)
            # ]
        len_splits = [len(self.indices1), len(self.indices2)]

        self._spline1 = CaloRationalQuadraticSpline(
            dims_in,
            dims_c,
            list(reversed(len_splits)),
            subnet_constructor=subnet_constructor,
            num_bins=num_bins,
            bounds_init=bounds_init,
            tails=tails,
            bounds_type=bounds_type,
            spatial=spatial,
        )

        self._spline2 = CaloRationalQuadraticSpline(
            dims_in,
            dims_c,
            len_splits,
            subnet_constructor=subnet_constructor,
            num_bins=num_bins,
            bounds_init=bounds_init,
            tails=tails,
            bounds_type=bounds_type,
            spatial=spatial,
        )

    def _coupling1(self, x1, u2, c, rev=False):
        y1, j1 = self._spline1(u2, x1, c, rev=rev)
        return y1, j1

    def _coupling2(self, x2, u1, c, rev=False):
        y2, j2 = self._spline2(u1, x2, c, rev=rev)
        return y2, j2

    def forward(self, x, c=[], rev=False, jac=True):
        """See base class docstring"""
        # TODO update notation
        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # *_c: variable with condition concatenated
        # j1, j2: Jacobians of the two coupling operations
        if self.spatial:
            x1, x2 = x[0][:, :, self.indices1], x[0][:, :, self.indices2]
        else:
            x1, x2 = x[0][:, self.indices1], x[0][:, self.indices2]

        # always the last vector is transformed
        y1, y2, j = super().forward(x1, x2, c)

        if self.spatial:
            y = x[0].clone()
            y[:, :, ::2] = y1
            y[:, :, 1::2] = y2
            # y = torch.cat((y1, y2), 2)
        else:
            y = x[0].clone()
            y[:, ::2] = y1
            y[:, 1::2] = y2
            # y = torch.cat((y1, y2), 1)
        return (y,), j


class OneSidedCaloRationalQuadraticSplineBlock(OneSidedBaseCouplingBlock):
    def __init__(
        self,
        dims_in,
        dims_c,
        subnet_constructor,
        num_bins=10,
        bounds_init=1.0,
        tails="linear",
        bounds_type="SOFTPLUS",
        spatial=False,
        *args
    ):
        super().__init__(dims_in, dims_c, *args)

        self.spatial = spatial
        if spatial:
            self.channels = dims_in[0][1]
            self.patch_dim = self.channels // 2
            self.num_patches = dims_in[0][0]
        else:
            self.channels = dims_in[0][0]
            self.patch_dim = dims_in[0][1]
            self.num_patches = self.channels // 2

        # ndims means the rank of tensor strictly speaking.
        # i.e. 1D, 2D, 3D tensor, etc.
        self.ndims = len(dims_in[0])

        if self.spatial:
            # self.indices1 = [int(k) for k in range(self.channels // 2)]
            # self.indices2 = [
            #    int(k + self.channels // 2) for k in range(self.channels // 2)
            # ]
            self.indices1 = [int(2 * k) for k in range(self.channels // 2)]
            self.indices2 = [int(2 * k + 1) for k in range(self.channels // 2)]
        else:
            self.indices1 = [int(2 * k) for k in range(self.channels // 2)]
            self.indices2 = [int(2 * k + 1) for k in range(self.channels // 2)]
            # self.indices1 = [int(k) for k in range(self.channels // 2)]
            # self.indices2 = [
            #    int(k + self.channels // 2) for k in range(self.channels // 2)
            # ]
        len_splits = [len(self.indices1), len(self.indices2)]

        self._spline = CaloRationalQuadraticSpline(
            dims_in,
            dims_c,
            len_splits,
            subnet_constructor=subnet_constructor,
            num_bins=num_bins,
            bounds_init=bounds_init,
            tails=tails,
            bounds_type=bounds_type,
            spatial=spatial,
        )

    def _coupling(self, x1, u2, c, rev=False):
        y1, j1 = self._spline(u2, x1, c, rev=rev)
        return y1, j1

    def forward(self, x, c=[], rev=False, jac=True):
        """See base class docstring"""
        # TODO update notation
        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # *_c: variable with condition concatenated
        # j1, j2: Jacobians of the two coupling operations
        if self.spatial:
            x1, x2 = x[0][:, :, self.indices1], x[0][:, :, self.indices2]
        else:
            x1, x2 = x[0][:, self.indices1], x[0][:, self.indices2]

        # always the last vector is transformed
        y2, j = super().forward(x2, x1, c)

        if self.spatial:
            y = x[0].clone()
            y[:, :, ::2] = x1
            y[:, :, 1::2] = y2
            # y = torch.cat((y1, y2), 2)
        else:
            y = x[0].clone()
            y[:, ::2] = x1
            y[:, 1::2] = y2
            # y = torch.cat((y1, y2), 1)
        return (y,), j


class SimpleRationalQuadraticSplineBlock(BaseCouplingBlock):
    def __init__(
        self,
        dims_in,
        dims_c,
        subnet_constructor,
        num_bins=10,
        bounds_init=1.0,
        tails="linear",
        bounds_type="SOFTPLUS",
        spatial=False,
        *args
    ):
        super().__init__(dims_in, dims_c, *args)

        self.spatial = spatial
        self.channels = dims_in[0][0]

        # ndims means the rank of tensor strictly speaking.
        # i.e. 1D, 2D, 3D tensor, etc.
        self.ndims = len(dims_in[0])

        # self.indices1 = [int(2 * k) for k in range(self.channels // 2)]
        # self.indices2 = [int(2 * k + 1) for k in range(self.channels // 2)]
        self.indices1 = [int(k) for k in range(self.channels // 2)]
        self.indices2 = [
            int(k + self.channels // 2)
            for k in range(self.channels - self.channels // 2)
        ]
        len_splits = [len(self.indices1), len(self.indices2)]

        self._spline1 = SimpleRationalQuadraticSpline(
            dims_in,
            dims_c,
            list(reversed(len_splits)),
            subnet_constructor=subnet_constructor,
            num_bins=num_bins,
            bounds_init=bounds_init,
            tails=tails,
            bounds_type=bounds_type,
            spatial=spatial,
        )

        self._spline2 = SimpleRationalQuadraticSpline(
            dims_in,
            dims_c,
            len_splits,
            subnet_constructor=subnet_constructor,
            num_bins=num_bins,
            bounds_init=bounds_init,
            tails=tails,
            bounds_type=bounds_type,
            spatial=spatial,
        )

    def _coupling1(self, x1, u2, c, rev=False):
        y1, j1 = self._spline1(u2, x1, c, rev=rev)
        return y1, j1

    def _coupling2(self, x2, u1, c, rev=False):
        y2, j2 = self._spline2(u1, x2, c, rev=rev)
        return y2, j2

    def forward(self, x, c=[], rev=False, jac=True):
        """See base class docstring"""
        # TODO update notation
        # notation:
        # x1, x2: two halves of the input
        # y1, y2: two halves of the output
        # *_c: variable with condition concatenated
        # j1, j2: Jacobians of the two coupling operations
        x1, x2 = x[0][:, self.indices1], x[0][:, self.indices2]

        # always the last vector is transformed
        y1, y2, j = super().forward(x1, x2, c)
        y = torch.cat((y1, y2), 1)

        return (y,), j


class SimpleRationalQuadraticSpline(InvertibleModule):
    DEFAULT_MIN_BIN_WIDTH = 1e-6
    DEFAULT_MIN_BIN_HEIGHT = 1e-6
    DEFAULT_MIN_DERIVATIVE = 1e-6

    def __init__(
        self,
        dims_in,
        dims_c,
        len_splits,
        subnet_constructor: Callable = None,
        num_bins: int = 10,
        bounds_init: float = 1.0,
        tails="linear",
        bounds_type="SOFTPLUS",
        spatial=False,
    ):

        super().__init__(dims_in, dims_c)

        self.spatial = spatial
        if spatial:
            channels = dims_in[0][1]
        else:
            channels = dims_in[0][0]
        # rank of the tensors means 1d, 2d, 3d tensor etc.
        self.input_rank = 1  # len(dims_in[0]) - 1
        # tuple containing all dims except for batch-dim (used at various points)
        self.sum_dims = tuple(range(1, 2 + self.input_rank))
        if len(dims_c) == 0:
            self.conditional = False
            self.condition_channels = 0
        else:
            self.conditional = True
            self.condition_channels = sum(dc[0] for dc in dims_c)

        self.splits = len_splits
        self.num_bins = num_bins
        if self.DEFAULT_MIN_BIN_WIDTH * self.num_bins > 1.0:
            raise ValueError("Minimal bin width too large for the number of bins")
        if self.DEFAULT_MIN_BIN_HEIGHT * self.num_bins > 1.0:
            raise ValueError("Minimal bin height too large for the number of bins")

        if bounds_type == "SIGMOID":
            bounds = 2.0 - np.log(10.0 / bounds_init - 1.0)
            self.bounds_activation = lambda a: 10 * torch.sigmoid(a - 2.0)
        elif bounds_type == "SOFTPLUS":
            bounds = 2.0 * np.log(np.exp(0.5 * 10.0 * bounds_init) - 1)
            self.softplus = nn.Softplus(beta=0.5)
            self.bounds_activation = lambda a: 0.1 * self.softplus(a)
        elif bounds_type == "EXP":
            bounds = np.log(bounds_init)
            self.bounds_activation = lambda a: torch.exp(a)
        elif bounds_type == "LIN":
            bounds = bounds_init
            self.bounds_activation = lambda a: a
        else:
            raise ValueError(
                'Global affine activation must be "SIGMOID", "SOFTPLUS" or "EXP"'
            )

        self.in_channels = channels
        self.bounds = self.bounds_activation(torch.tensor(bounds))
        self.tails = tails

        if subnet_constructor is None:
            raise ValueError(
                "Please supply a callable subnet_constructor"
                "function or object (see docstring)"
            )
        self.subnet = subnet_constructor(
            self.splits[0] + self.condition_channels,
            (3 * self.num_bins - 1) * self.splits[1],
        )

    def _unconstrained_rational_quadratic_spline(self, inputs, theta, rev=False):

        inside_interval_mask = torch.all(
            (inputs >= -self.bounds) & (inputs <= self.bounds), dim=-1
        )
        outside_interval_mask = ~inside_interval_mask

        masked_outputs = torch.zeros_like(inputs)
        masked_logabsdet = torch.zeros(inputs.shape[0], dtype=inputs.dtype).to(
            inputs.device
        )
        # masked_logabsdet = torch.zeros_like(inputs)

        min_bin_width = self.DEFAULT_MIN_BIN_WIDTH
        min_bin_height = self.DEFAULT_MIN_BIN_HEIGHT
        min_derivative = self.DEFAULT_MIN_DERIVATIVE

        if self.tails == "linear":
            masked_outputs[outside_interval_mask] = inputs[outside_interval_mask]
            masked_logabsdet[outside_interval_mask] = 0

        else:
            raise RuntimeError("{} tails are not implemented.".format(self.tails))
        inputs = inputs[inside_interval_mask]
        theta = theta[inside_interval_mask, :]
        bound = torch.min(self.bounds)

        left = -bound
        right = bound
        bottom = -bound
        top = bound

        unnormalized_widths = theta[..., : self.num_bins]
        unnormalized_heights = theta[..., self.num_bins : self.num_bins * 2]
        unnormalized_derivatives = theta[..., self.num_bins * 2 :]

        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        widths = F.softmax(unnormalized_widths, dim=-1)
        widths = min_bin_width + (1 - min_bin_width * self.num_bins) * widths
        cumwidths = torch.cumsum(widths, dim=-1)
        cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
        cumwidths = (right - left) * cumwidths + left
        cumwidths[..., 0] = left
        cumwidths[..., -1] = right
        widths = cumwidths[..., 1:] - cumwidths[..., :-1]

        derivatives = min_derivative + F.softplus(unnormalized_derivatives)

        heights = F.softmax(unnormalized_heights, dim=-1)
        heights = min_bin_height + (1 - min_bin_height * self.num_bins) * heights
        cumheights = torch.cumsum(heights, dim=-1)
        cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
        cumheights = (top - bottom) * cumheights + bottom
        cumheights[..., 0] = bottom
        cumheights[..., -1] = top
        heights = cumheights[..., 1:] - cumheights[..., :-1]

        if rev:
            bin_idx = self.searchsorted(cumheights, inputs)[..., None]
        else:
            bin_idx = self.searchsorted(cumwidths, inputs)[..., None]

        input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
        input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

        input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
        delta = heights / widths
        input_delta = delta.gather(-1, bin_idx)[..., 0]

        input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
        input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

        input_heights = heights.gather(-1, bin_idx)[..., 0]

        if rev:
            inputs = inputs.to(torch.float64)
            input_cumheights = input_cumheights.to(torch.float64)
            input_derivatives = input_derivatives.to(torch.float64)
            input_derivatives_plus_one = input_derivatives_plus_one.to(torch.float64)
            input_delta = input_delta.to(torch.float64)
            input_heights = input_heights.to(torch.float64)

            a = (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
            ) + input_heights * (input_delta - input_derivatives)
            b = input_heights * input_derivatives - (inputs - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
            )
            c = -input_delta * (inputs - input_cumheights)

            discriminant = b.pow(2) - 4 * a * c

            # inputs = inputs.to(torch.float32)
            # input_cumheights = input_cumheights.to(torch.float32)
            # input_derivatives = input_derivatives.to(torch.float32)
            # input_derivatives_plus_one = input_derivatives_plus_one.to(torch.float32)
            # input_delta = input_delta.to(torch.float32)
            # input_heights = input_heights.to(torch.float32)

            # a = a.to(torch.float32)
            # b = b.to(torch.float32)
            # c = c.to(torch.float32)
            # discriminant = discriminant.to(torch.float32)
            assert (discriminant >= 0).all()

            root = (2 * c) / (-b - torch.sqrt(discriminant))
            outputs = root * input_bin_widths + input_cumwidths

            theta_one_minus_theta = root * (1 - root)
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                * theta_one_minus_theta
            )
            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * root.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - root).pow(2)
            )
            logabsdet = -torch.log(derivative_numerator) + 2 * torch.log(denominator)

            outputs = outputs.to(torch.float32)
            logabsdet = logabsdet.to(torch.float32)
        else:
            theta = (inputs - input_cumwidths) / input_bin_widths
            theta_one_minus_theta = theta * (1 - theta)

            numerator = input_heights * (
                input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
            )
            denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                * theta_one_minus_theta
            )
            outputs = input_cumheights + numerator / denominator

            derivative_numerator = input_delta.pow(2) * (
                input_derivatives_plus_one * theta.pow(2)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * (1 - theta).pow(2)
            )
            logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        logabsdet = torch.sum(logabsdet, dim=-1)

        masked_outputs = masked_outputs.to(outputs.dtype)
        masked_logabsdet = masked_logabsdet.to(outputs.dtype)
        masked_outputs[inside_interval_mask], masked_logabsdet[inside_interval_mask] = (
            outputs,
            logabsdet,
        )

        return masked_outputs, masked_logabsdet

    def searchsorted(self, bin_locations, inputs, eps=1e-6):
        bin_locations[..., -1] += eps
        return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1

    def forward(self, x1, x2, c=[], rev=False):
        """See base class docstring"""
        self.bounds = self.bounds.to(x1[0].device)

        x1c = torch.cat([x1, *c], 1)
        if not rev:
            theta = self.subnet(x1c).reshape(
                x1c.shape[0], self.splits[1], 3 * self.num_bins - 1
            )
            x2, j2 = self._unconstrained_rational_quadratic_spline(x2, theta, rev=False)
        else:
            theta = self.subnet(x1c).reshape(
                x1c.shape[0], self.splits[1], 3 * self.num_bins - 1
            )
            x2, j2 = self._unconstrained_rational_quadratic_spline(x2, theta, rev=True)

        log_jac_det = j2
        x_out = x2

        return x_out, log_jac_det

    def output_dims(self, input_dims):
        return input_dims


class CaloRationalQuadraticSpline(SimpleRationalQuadraticSpline):
    def __init__(
        self,
        dims_in,
        dims_c,
        len_splits,
        subnet_constructor: Callable = None,
        num_bins: int = 10,
        bounds_init: float = 1.0,
        tails="linear",
        bounds_type="SOFTPLUS",
        spatial=False,
    ):

        super().__init__(
            dims_in,
            dims_c,
            len_splits,
            subnet_constructor,
            num_bins,
            bounds_init,
            tails,
            bounds_type,
        )

        self.spatial = spatial
        if spatial:
            channels = dims_in[0][1]
        else:
            channels = dims_in[0][0]
        self.in_channels = channels

        self.subnet = subnet_constructor(self.splits[0], (3 * self.num_bins - 1))

    def forward(self, x1, x2, c=[], rev=False):
        """See base class docstring"""
        self.bounds = self.bounds.to(x1[0].device)

        x1c = x1  # always conditional
        if not rev:
            theta = self.subnet(x1c, c).reshape(
                x1c.shape[0], -1, 3 * self.num_bins - 1
            )  # self.splits[1], -1, 3*self.num_bins - 1)
            x2 = x2.reshape(x2.shape[0], -1)
            x2, j2 = self._unconstrained_rational_quadratic_spline(x2, theta, rev=False)
            # j2 = utils.sum_except_batch(j2)
        else:
            theta = self.subnet(x1c, c).reshape(
                x1c.shape[0], -1, 3 * self.num_bins - 1
            )  # , self.splits[1], -1, 3*self.num_bins - 1)
            x2 = x2.reshape(x2.shape[0], -1)
            x2, j2 = self._unconstrained_rational_quadratic_spline(x2, theta, rev=True)
            # j2 = -utils.sum_except_batch(j2)
        if self.spatial:
            x2 = x2.reshape(x2.shape[0], -1, self.splits[1])
        else:
            x2 = x2.reshape(x2.shape[0], self.splits[1], -1)
        log_jac_det = j2
        x_out = x2  # torch.cat((x1, x2), 1)

        return x_out, log_jac_det

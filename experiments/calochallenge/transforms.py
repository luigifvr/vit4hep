import os
from itertools import pairwise

import numpy as np
import torch
import torch.nn.functional as F

from experiments.calo_utils.ugr_evaluation import XMLHandler


def logit(array, alpha=1.0e-6, inv=False):
    if inv:
        z = torch.sigmoid(array)
        z = (z - alpha) / (1 - 2 * alpha)
    else:
        z = array * (1 - 2 * alpha) + alpha
        z = torch.logit(z)
    return z


class GlobalStandardizeFromFile:
    """
    Standardize features
        mean_path: path to `.npy` file containing means of the features
        std_path: path to `.npy` file containing standard deviations of the features
        create: whether or not to calculate and save mean/std based on first call
    """

    def __init__(self, model_dir, exclude_zeros=True, eps=1.0e-6):

        self.model_dir = model_dir
        self.mean_path = os.path.join(model_dir, "means.npy")
        self.std_path = os.path.join(model_dir, "stds.npy")

        self.dtype = torch.get_default_dtype()
        self.u_transform = True
        self.exclude_zeros = exclude_zeros
        self.eps = torch.logit(torch.tensor(eps))
        try:
            # load from file
            self.mean = torch.from_numpy(np.load(self.mean_path)).to(self.dtype)
            self.std = torch.from_numpy(np.load(self.std_path)).to(self.dtype)
            self.written = True
        except FileNotFoundError:
            self.written = False

    def write(self):
        np.save(self.mean_path, self.mean.detach().cpu().numpy())
        np.save(self.std_path, self.std.detach().cpu().numpy())

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:
            transformed = shower * self.std.to(shower.device) + self.mean.to(
                shower.device
            )
        else:
            if not self.written:
                nonzero_mask = (shower > self.eps) & (shower < -self.eps)
                if not self.exclude_zeros:
                    nonzero_mask = nonzero_mask.fill_(True)
                self.mean = (shower[nonzero_mask]).mean()
                self.std = (shower[nonzero_mask]).std()
                if rank == 0:
                    self.write()
                self.written = True
            transformed = (shower - self.mean.to(shower.device)) / self.std.to(
                shower.device
            )
        return transformed, energy


class StandardizeUsFromFile:
    """
    Standardize features
        mean_path: path to `.npy` file containing means of the features
        std_path: path to `.npy` file containing standard deviations of the features
        create: whether or not to calculate and save mean/std based on first call
    """

    def __init__(self, n_us, model_dir):

        self.model_dir = model_dir
        self.mean_us_path = os.path.join(model_dir, "means_u.npy")
        self.std_us_path = os.path.join(model_dir, "stds_u.npy")

        self.dtype = torch.get_default_dtype()
        self.n_us = n_us
        self.u_transform = True
        try:
            # load from file
            self.mean_u = torch.from_numpy(np.load(self.mean_us_path)).to(self.dtype)
            self.std_u = torch.from_numpy(np.load(self.std_us_path)).to(self.dtype)
            self.written = True
        except FileNotFoundError:
            self.written = False

    def write(self):
        np.save(self.mean_us_path, self.mean_u.detach().cpu().numpy())
        np.save(self.std_us_path, self.std_u.detach().cpu().numpy())

    def __call__(self, shower, energy, rev=False, rank=0):
        us = shower[:, -self.n_us :]
        voxels = shower[:, : -self.n_us]
        if rev:
            trafo_us = us * self.std_u.to(shower.device) + self.mean_u.to(shower.device)
            transformed = torch.cat((voxels, trafo_us), dim=1)
        else:
            if not self.written:
                self.mean_u = us.mean(0)
                self.std_u = us.std(0)
                if rank == 0:
                    self.write()
                self.written = True
            trafo_us = (us - self.mean_u.to(shower.device)) / self.std_u.to(
                shower.device
            )
            transformed = torch.cat((voxels, trafo_us), dim=1)
        return transformed, energy


class SelectDims:
    """
    Selects a subset of the features
        start: start of range of indices to keep
        end:   end of range of indices to keep (exclusive)
    """

    def __init__(self, start, end):
        self.indices = torch.arange(start, end)

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:
            return shower, energy
        transformed = shower[..., self.indices]
        return transformed, energy


class AddFeaturesToCond:
    """
    Transfers a subset of the input features to the condition
        split_index: Index at which to split input. Features past the index will be moved
    """

    def __init__(self, split_index):
        self.split_index = split_index

    def __call__(self, x, c, rev=False, rank=0):

        if rev:
            c_, split = c[:, -1:], c[:, :-1]
            x_ = torch.cat([x, split], dim=1)
        else:
            x_, split = x[:, : self.split_index], x[:, self.split_index :]
            c_ = torch.cat([split, c], dim=1)
        return x_, c_


class LogEnergy:
    """
    Log transform incident energies
        alpha: Optional regularization for the log
    """

    def __init__(self, alpha=0.0):
        self.alpha = alpha
        self.cond_transform = True

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:
            transformed = torch.exp(energy) - self.alpha
        else:
            transformed = torch.log(energy + self.alpha)
        return shower, transformed


class ScaleVoxels:
    """
    Apply a multiplicative factor to the voxels.
        factor: Number to multiply voxels
    """

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:
            transformed = shower / self.factor
        else:
            transformed = shower * self.factor
        return transformed, energy


class ScaleTotalEnergy:
    """
    Scale only E_tot/E_inc by a factor f.
    The effect is the same of ScaleVoxels but
    it is applied in a different position in the
    preprocessing chain.
    """

    def __init__(self, factor, n_layers=45):
        self.factor = factor
        self.n_layers = n_layers
        self.u_transform = True

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:
            shower[..., -self.n_layers] /= self.factor
        else:
            shower[..., -self.n_layers] *= self.factor
        return shower, energy


class ScaleEnergy:
    """
    Scale incident energies to lie in the range [0, 1]
        e_min: Expected minimum value of the energy
        e_max: Expected maximum value of the energy
    """

    def __init__(self, e_min, e_max):
        self.e_min = e_min
        self.e_max = e_max
        self.cond_transform = True

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:
            transformed = energy * (self.e_max - self.e_min)
            transformed += self.e_min
        else:
            transformed = energy - self.e_min
            transformed /= self.e_max - self.e_min
        return shower, transformed


class ExclusiveLogitTransform:
    """
    Take log of input data
        delta: regularization
        exclusions: list of indices for features that should not be transformed
    """

    def __init__(self, delta, exclusions=None, rescale=False):
        self.delta = delta
        self.exclusions = exclusions
        self.rescale = rescale
        self.u_transform = True

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:
            if self.rescale:
                transformed = logit(shower, alpha=self.delta, inv=True)
            else:
                transformed = torch.special.expit(shower)
        else:
            if self.rescale:
                transformed = logit(shower, alpha=self.delta)
            else:
                transformed = torch.logit(shower, eps=self.delta)

        if self.exclusions is not None:
            transformed[..., self.exclusions] = shower[..., self.exclusions]
        return transformed, energy


class SelectiveUniformNoise:
    """
    Add noise to input data with the option to exlude some features
        func: torch distribution used to sample from
        width_noise: noise rescaling
        exclusions: list of indices for features that should not be transformed
    """

    def __init__(self, a, b, exclusions=None, cut=False):
        # self.func = func
        self.a = a
        self.b = b
        self.func = torch.distributions.Uniform(
            torch.tensor(self.a), torch.tensor(self.b)
        )
        self.exclusions = exclusions
        self.cut = cut  # apply cut if True

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:
            mask = shower < self.b
            if self.exclusions:
                mask[:, self.exclusions] = False
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0
        else:
            noise = self.func.sample(shower.shape).to(shower.dtype)
            mask = shower != 1
            if self.exclusions:
                noise[:, self.exclusions] = 0.0
            transformed = shower
            transformed[mask] = (shower + noise.to(shower.device))[mask]
        return transformed, energy


class CutValues:
    """
    Cut in Normalized space
        cut: threshold value for the cut
        n_layers: number of layers to avoid cutting on the us
    """

    def __init__(self, cut=0.0, n_layers=45):
        self.cut = cut
        self.n_layers = n_layers

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:
            mask = shower <= self.cut
            mask[:, -self.n_layers :] = False
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0
        else:
            transformed = shower
        return transformed, energy


class Reshape:
    """
    Reshape the shower as specified. Flattens batch in the reverse transformation.
        shape -- Tuple representing the desired shape of a single example
    """

    def __init__(self, shape):
        self.shape = torch.Size(shape)

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:
            shower = shower.reshape(-1, self.shape.numel())
        else:
            shower = shower.reshape(-1, *self.shape)
        return shower, energy


class NormalizeByElayer:
    """
    Normalize each shower by the layer energy
    This will change the shower shape to N_voxels+N_layers
    """

    def __init__(self, ptype, xml_file, cut=0.0, eps=1.0e-10):
        self.eps = eps
        self.xml = XMLHandler.XMLHandler(xml_file, ptype)
        self.layer_boundaries = np.unique(self.xml.GetBinEdges())
        self.n_layers = len(self.layer_boundaries) - 1
        self.cut = cut

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:

            # select u features
            us = shower[:, -self.n_layers :]

            # clip u_{i>0} into [0,1]
            us[:, (-self.n_layers + 1) :] = torch.clip(
                us[:, (-self.n_layers + 1) :],
                min=torch.tensor(0.0, device=shower.device),
                max=torch.tensor(1.0, device=shower.device),
            )

            # select voxels
            shower = shower[:, : -self.n_layers]

            # calculate unnormalised energies from the u's
            layer_Es = []
            total_E = torch.multiply(energy.flatten(), us[:, 0])  # Einc * u_0
            cum_sum = torch.zeros_like(total_E)
            for i in range(us.shape[-1] - 1):
                layer_E = (total_E - cum_sum) * us[:, i + 1]
                layer_Es.append(layer_E)
                cum_sum += layer_E
            layer_Es.append(total_E - cum_sum)
            layer_Es = torch.vstack(layer_Es).T

            # Normalize each layer and multiply it with its original energy
            transformed = torch.zeros_like(shower)
            for L, (start, end) in enumerate(pairwise(self.layer_boundaries)):
                layer = shower[:, start:end]  # select layer
                layer /= layer.sum(-1, keepdims=True) + self.eps  # normalize to unity
                mask = layer <= self.cut
                layer[mask] = 0.0  # apply normalized cut
                transformed[:, start:end] = (
                    layer * layer_Es[:, [L]]
                )  # scale to layer energy

        else:
            # compute layer energies
            layer_Es = []
            for start, end in pairwise(self.layer_boundaries):
                layer_E = torch.sum(shower[:, start:end], dim=1, keepdims=True)
                shower[:, start:end] /= layer_E + self.eps  # normalize to unity
                layer_Es.append(layer_E)  # store layer energy
            layer_Es = torch.cat(layer_Es, dim=1).to(shower.device)

            # compute generalized extra dimensions
            extra_dims = [torch.sum(layer_Es, dim=1, keepdim=True) / energy]
            for L in range(layer_Es.shape[1] - 1):
                remaining_E = torch.sum(layer_Es[:, L:], dim=1, keepdim=True)
                extra_dim = layer_Es[:, [L]] / (remaining_E + self.eps)
                extra_dims.append(extra_dim)
            extra_dims = torch.cat(extra_dims, dim=1)

            transformed = torch.cat((shower, extra_dims), dim=1)
        return transformed, energy


class AddAngularBins:
    """
    Add angular bins given an XML file. After the transformation the shower
    will have a regular geometry. The inverse step takes the maximum over the
    added angular bins.
    Args:
    xml_filename: path to the XML file
    ptype: particle type (e.g. "electron")
    num_bins: list with current number of angular bins per layer
    add_bins: list with desired number of angular bins per layer
    """

    def __init__(self, xml_filename, ptype, num_bins, add_bins):
        self.xml = XMLHandler.XMLHandler(xml_filename, ptype)
        self.layer_boundaries = np.unique(self.xml.GetBinEdges())
        self.num_bins = np.array(num_bins)
        self.add_bins = np.array(add_bins)
        self.n_voxels = self.layer_boundaries[-1]

    def __call__(self, shower, energy, rev=False, rank=0):

        if rev:
            new_n_voxels = self.new_layer_boundaries[-1]
            shower, us = shower[:, :new_n_voxels], shower[:, new_n_voxels:]
            transformed = []
            for L, (start, end) in enumerate(pairwise(self.new_layer_boundaries)):
                alpha_bins = self.num_bins[L]
                add_alpha_bins = self.add_bins[L] // alpha_bins
                layer = shower[:, start:end]
                layer, _ = layer.reshape(
                    shower.shape[0], -1, alpha_bins, add_alpha_bins
                ).max(-1)
                transformed.append(layer.reshape(layer.shape[0], -1))
            transformed = torch.cat(transformed, dim=-1).to(
                dtype=shower.dtype, device=shower.device
            )
            transformed = torch.cat((transformed, us), dim=-1)
        else:
            shower, us = shower[:, : self.n_voxels], shower[:, self.n_voxels :]
            transformed = []
            self.new_layer_boundaries = [
                0,
            ]
            for L, (start, end) in enumerate(pairwise(self.layer_boundaries)):
                alpha_bins = self.num_bins[L]
                add_alpha_bins = self.add_bins[L] // alpha_bins - 1
                layer = shower[:, start:end].reshape(shower.shape[0], -1, alpha_bins)
                pad_left = add_alpha_bins // 2
                pad_right = add_alpha_bins - add_alpha_bins // 2
                layer = F.pad(layer, (pad_left, pad_right), "constant", 0).reshape(
                    layer.shape[0], -1
                )

                transformed.append(layer)
                self.new_layer_boundaries.append(
                    self.new_layer_boundaries[L] + layer.shape[-1]
                )
            transformed = torch.cat(transformed, dim=-1).to(
                dtype=shower.dtype, device=shower.device
            )
            transformed = torch.cat((transformed, us), dim=-1)
        return transformed, energy


class AddLEMURSConditions:
    """
    Add global variables to match the LEMURS conditions.
    """

    def __init__(self, theta=0.5, phi=0.5, label=[1, 0, 0, 0, 0]):
        self.theta = theta
        self.phi = phi
        self.label = label
        self.n_conds = 2 + len(label)

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:
            energy, additional_conds = (
                energy[:, : -self.n_conds],
                energy[:, -self.n_conds :],
            )
            return shower, energy
        else:
            additional_conds = (
                torch.tensor(
                    [self.theta, self.phi] + self.label,
                    dtype=energy.dtype,
                    device=energy.device,
                )
                .unsqueeze(0)
                .repeat(energy.shape[0], 1)
            )
            energy = torch.cat((energy, additional_conds), dim=1)
            return shower, energy

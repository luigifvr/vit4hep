import torch
import torch.nn.functional as F
import numpy as np
import os
import torch.distributions as dist

from experiments.calochallenge.challenge_files import *
from experiments.calochallenge.challenge_files import XMLHandler
from itertools import pairwise


class LogUniform(dist.TransformedDistribution):
    def __init__(self, lb, ub):
        super(LogUniform, self).__init__(
            dist.Uniform(lb.log(), ub.log()), dist.ExpTransform()
        )


def logit(array, alpha=1.0e-6, inv=False):
    if inv:
        z = torch.sigmoid(array)
        z = (z - alpha) / (1 - 2 * alpha)
    else:
        z = array * (1 - 2 * alpha) + alpha
        z = torch.logit(z)
    return z


class Standardize(object):
    """
    Standardize features
        mean: vector of means
        std: vector of stds
    """

    def __init__(self, means, stds):
        self.means = means
        self.stds = stds

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:
            transformed = shower * self.stds + self.means
        else:
            transformed = (shower - self.means) / self.stds
        return transformed, energy


class GlobalStandardizeFromFileGAN(object):
    """
    Standardize features
        mean_path: path to `.npy` file containing means of the features
        std_path: path to `.npy` file containing standard deviations of the features
        create: whether or not to calculate and save mean/std based on first call
    """

    def __init__(self, model_dir):

        self.model_dir = model_dir
        self.mean_path = os.path.join(model_dir, "means.npy")
        self.std_path = os.path.join(model_dir, "stds.npy")

        self.dtype = torch.get_default_dtype()
        self.u_transform = True
        self.layer_keys = ["layer_0", "layer_1", "layer_2", "extra_dims"]
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

    def __call__(self, data_dict, rev=False, rank=0):
        if rev:
            for key in self.layer_keys:
                data_dict[key] = data_dict[key] * self.std + self.mean
        else:
            if not self.written:
                shower = torch.cat([data_dict[key] for key in self.layer_keys], dim=1)
                self.mean = shower.mean()
                self.std = shower.std()
                if rank == 0:
                    self.write()
                self.written = True
            for key in self.layer_keys:
                data_dict[key] = (data_dict[key] - self.mean) / self.std
        return data_dict


class StandardizeVoxelsFromFile(object):
    """
    Standardize features
        mean_path: path to `.npy` file containing means of the features
        std_path: path to `.npy` file containing standard deviations of the features
        create: whether or not to calculate and save mean/std based on first call
    """

    def __init__(self, n_voxels, model_dir):

        self.model_dir = model_dir
        self.mean_path = os.path.join(model_dir, "means.npy")
        self.std_path = os.path.join(model_dir, "stds.npy")

        self.dtype = torch.get_default_dtype()
        self.n_voxels = n_voxels
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
        voxels = shower[:, : self.n_voxels]
        us = shower[:, self.n_voxels :]
        if rev:
            trafo_voxels = voxels * self.std.to(shower.device) + self.mean.to(
                shower.device
            )
            transformed = torch.cat((trafo_voxels, us), dim=1)
        else:
            if not self.written:
                self.mean = voxels.mean(0)
                self.std = voxels.std(0)
                if rank == 0:
                    self.write()
                self.written = True
            trafo_voxels = (voxels - self.mean.to(shower.device)) / self.std.to(
                shower.device
            )
            transformed = torch.cat((trafo_voxels, us), dim=1)
        return transformed, energy


class StandardizeUsFromFile(object):
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


class SelectDims(object):
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


class AddFeaturesToCond(object):
    """
    Transfers a subset of the input features to the condition
        split_index: Index at which to split input. Features past the index will be moved
    """

    def __init__(self, split_index):
        self.split_index = split_index

    def __call__(self, x, c, rev=False, rank=0):

        if rev:
            c_, split = c[:, :1], c[:, 1:]
            x_ = torch.cat([x, split], dim=1)
        else:
            x_, split = x[:, : self.split_index], x[:, self.split_index :]
            c_ = torch.cat([c, split], dim=1)
        return x_, c_


class AddEmptyLayer(object):
    """
    Add an empty layer.
    """

    def __init__(self, noise_width=0.0, shape=(1, 1, 16, 9)):
        self.shape = shape
        self.noise_width = noise_width

    def __call__(self, x, c, rev=False, rank=0):
        if rev:
            c_ = c
            x_ = x[:, :, :-1]
        else:
            layer = torch.rand((x.shape[0],) + self.shape) * self.noise_width
            c_ = c
            x_ = torch.cat((x, layer), dim=2)
        return x_, c_


class LogEnergyGAN(object):
    """
    Log transform incident energies
        alpha: Optional regularization for the log
    """

    def __init__(self, alpha=0.0):
        self.alpha = alpha
        self.cond_transform = True

    def __call__(self, data_dict, rev=False, rank=0):
        if rev:
            energy = data_dict["energy"]
            data_dict["energy"] = torch.exp(energy) - self.alpha
        else:
            energy = data_dict["energy"]
            data_dict["energy"] = torch.log(energy + self.alpha)
        return data_dict


class ScaleVoxels(object):
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


class ScaleTotalEnergy(object):
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


class ScaleEnergyGAN(object):
    """
    Scale incident energies to lie in the range [0, 1]
        e_min: Expected minimum value of the energy
        e_max: Expected maximum value of the energy
    """

    def __init__(self, e_min, e_max):
        self.e_min = e_min
        self.e_max = e_max
        self.cond_transform = True

    def __call__(self, data_dict, rev=False, rank=0):
        if rev:
            energy = data_dict["energy"]
            transformed = energy * (self.e_max - self.e_min)
            transformed += self.e_min
            data_dict["energy"] = transformed
        else:
            energy = data_dict["energy"]
            transformed = energy - self.e_min
            transformed /= self.e_max - self.e_min
            data_dict["energy"] = transformed
        return data_dict


class ExclusiveLogTransform(object):
    """
    Take log of input data
        delta: regularization
        exclusions: list of indices for features that should not be transformed
    """

    def __init__(self, delta, exclusions=None):
        self.delta = delta
        self.exclusions = exclusions
        self.u_transform = True

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:
            transformed = torch.exp(shower) - self.delta
        else:
            transformed = torch.log(shower + self.delta)
        if self.exclusions is not None:
            transformed[..., self.exclusions] = shower[..., self.exclusions]
        return transformed, energy


class ExclusiveLogitTransformGAN(object):
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
        self.layer_keys = ["layer_0", "layer_1", "layer_2", "extra_dims"]

    def __call__(self, data_dict, rev=False, rank=0):
        if rev:
            for key in self.layer_keys:
                if self.rescale:
                    data_dict[key] = logit(data_dict[key], alpha=self.delta, inv=True)
                else:
                    data_dict[key] = torch.special.expit(data_dict[key])
        else:
            for key in self.layer_keys:
                if self.rescale:
                    data_dict[key] = logit(data_dict[key], alpha=self.delta)
                else:
                    data_dict[key] = torch.logit(data_dict[key], eps=self.delta)
        return data_dict


class RegularizeLargeLogit(object):
    def __init__(self, a, b, exclusions=None, cut=False):
        self.a = a
        self.b = b
        self.func = torch.distributions.Uniform(
            torch.tensor(self.a), torch.tensor(self.b)
        )
        self.exclusions = exclusions
        self.cut = cut
        self.u_transform = True

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:
            mask = shower > 1 - self.b
            if self.exclusions:
                mask[:, self.exclusions] = False
            transformed = shower
            if self.cut:
                transformed[mask] = 1.0
        else:
            transformed = shower
            mask = shower >= 1.0
            noise = self.func.sample(shower.shape).to(shower.dtype)
            if self.exclusions:
                noise[:, self.exclusions] = 0.0
            transformed[mask] = (shower - noise.to(shower.device))[mask]
        return transformed, energy


class AddNoise(object):
    """
    Add noise to input data
        func: torch distribution used to sample from
        width_noise: noise rescaling
    """

    def __init__(self, noise_width, cut=False):
        # self.func = func
        self.func = torch.distributions.Uniform(torch.tensor(0.0), torch.tensor(1.0))
        self.noise_width = noise_width
        self.cut = cut  # apply cut if True

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:
            mask = shower < self.noise_width
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0
        else:
            noise = self.func.sample(shower.shape) * self.noise_width
            transformed = shower + noise.reshape(shower.shape).to(shower.device)
        return transformed, energy


class SelectiveLogUniformNoise(object):
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
        self.func = LogUniform(torch.tensor(self.a), torch.tensor(self.b))
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


class SelectiveUniformNoise(object):
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


class SelectiveNormalNoise(object):
    """
    Add noise to input data with the option to exlude some features
        func: torch distribution used to sample from
        exclusions: list of indices for features that should not be transformed
    """

    def __init__(self, mean, std, exclusions=None, cut=None):
        # self.func = func
        self.mean = mean
        self.std = std
        self.exclusions = exclusions
        self.cut = 0.0 if cut is None else cut  # apply cut if True

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:
            mask = shower < self.cut
            if self.exclusions:
                mask[:, self.exclusions] = False
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0
        else:
            noise = torch.zeros(shower.shape[0], shower.shape[1]).normal_(
                mean=self.mean, std=self.std
            )
            if self.exclusions:
                noise[:, self.exclusions] = 0.0
            transformed = shower + noise.to(shower.device)
        return transformed, energy


class SelectiveLogNormalNoise(object):
    """
    Add noise to input data with the option to exlude some features
        func: torch distribution used to sample from
        width_noise: noise rescaling
        exclusions: list of indices for features that should not be transformed
    """

    def __init__(self, mean, std, exclusions=None, cut=None):
        # self.func = func
        self.mean = mean
        self.std = std
        self.exclusions = exclusions
        self.cut = 0.0 if cut is None else cut  # apply cut if True

    def __call__(self, shower, energy, rev=False, rank=0):
        if rev:
            mask = shower < self.cut
            if self.exclusions:
                mask[:, self.exclusions] = False
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0
        else:
            noise = torch.zeros(shower.shape[0], shower.shape[1]).log_normal_(
                mean=self.mean, std=self.std
            )
            if self.exclusions:
                noise[:, self.exclusions] = 0.0
            transformed = shower + noise.to(shower.device)
        return transformed, energy


class CutValues(object):
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


class Reshape(object):
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


class NormalizeLayerEnergyGAN(object):
    """
    Normalize each shower by the layer energy
    This will change the shower shape to N_voxels+N_layers
       layer_boundaries: ''
       eps: numerical epsilon
    """

    def __init__(self, cut=0.0, eps=1.0e-10):
        self.bin_edges = [0, 288, 432, 504]
        self.eps = eps
        self.cut = cut
        self.layer_keys = ["layer_0", "layer_1", "layer_2"]
        self.n_layers = 3

    def __call__(self, data_dict, rev=False, rank=0):
        energy = data_dict["energy"]
        if rev:

            # select u features
            us = data_dict["extra_dims"]

            # clip u_{i>0} into [0,1]
            us[:, (-self.n_layers + 1) :] = torch.clip(
                us[:, (-self.n_layers + 1) :],
                min=torch.tensor(0.0, device=energy.device),
                max=torch.tensor(1.0, device=energy.device),
            )

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

            for l, key in enumerate(self.layer_keys):
                layer = data_dict[key].clone()  # select layer
                layer /= layer.sum(-1, keepdims=True) + self.eps  # normalize to unity
                mask = layer <= self.cut
                layer[mask] = 0.0  # apply normalized cut
                data_dict[key] = layer * layer_Es[:, [l]]  # scale to layer energy
        else:
            # compute layer energies
            layer_Es = []
            for key in self.layer_keys:
                layer_E = torch.sum(data_dict[key], dim=1, keepdims=True)
                data_dict[key] /= layer_E + self.eps  # normalize to unity
                layer_Es.append(layer_E)  # store layer energy
            layer_Es = torch.cat(layer_Es, dim=1).to(energy.device)

            # compute generalized extra dimensions
            extra_dims = [torch.sum(layer_Es, dim=1, keepdim=True) / energy]
            for l in range(layer_Es.shape[1] - 1):
                remaining_E = torch.sum(layer_Es[:, l:], dim=1, keepdim=True)
                extra_dim = layer_Es[:, [l]] / (remaining_E + self.eps)
                extra_dims.append(extra_dim)
            extra_dims = torch.cat(extra_dims, dim=1)
            data_dict["extra_dims"] = extra_dims
        return data_dict


class AddAngularBins(object):
    def __init__(self, xml_filename, ptype, num_bins, n_voxels):
        self.xml = XMLHandler.XMLHandler(xml_filename, ptype)
        self.layer_boundaries = np.unique(self.xml.GetBinEdges())
        self.num_bins = np.array(num_bins)
        self.n_voxels = self.layer_boundaries[-1]

    def __call__(self, shower, energy, rev=False, rank=0):

        if rev:
            new_n_voxels = self.new_layer_boundaries[-1]
            shower, us = shower[:, :new_n_voxels], shower[:, new_n_voxels:]
            transformed = []
            for l, (start, end) in enumerate(pairwise(self.new_layer_boundaries)):
                alpha_bins = self.num_bins[l]
                add_alpha_bins = self.num_bins.max() // alpha_bins
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
            for l, (start, end) in enumerate(pairwise(self.layer_boundaries)):
                alpha_bins = self.num_bins[l]
                add_alpha_bins = self.num_bins.max() // alpha_bins - 1
                layer = shower[:, start:end].reshape(shower.shape[0], -1, alpha_bins)
                pad_left = add_alpha_bins // 2
                pad_right = add_alpha_bins - add_alpha_bins // 2
                layer = F.pad(layer, (pad_left, pad_right), "constant", 0).reshape(
                    layer.shape[0], -1
                )

                transformed.append(layer)
                self.new_layer_boundaries.append(
                    self.new_layer_boundaries[l] + layer.shape[-1]
                )
            transformed = torch.cat(transformed, dim=-1).to(
                dtype=shower.dtype, device=shower.device
            )
            transformed = torch.cat((transformed, us), dim=-1)
        return transformed, energy

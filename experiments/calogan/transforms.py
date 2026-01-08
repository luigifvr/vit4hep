import os

import numpy as np
import torch

from experiments.calochallenge.transforms import logit


class GlobalStandardizeFromFileGAN:
    """
    Standardize features
        mean_path: path to `.npy` file containing means of the features
        std_path: path to `.npy` file containing standard deviations of the features
        create: whether or not to calculate and save mean/std based on first call
    """

    def __init__(self, model_dir, eps=1.0e-6):

        self.model_dir = model_dir
        self.mean_path = os.path.join(model_dir, "means.npy")
        self.std_path = os.path.join(model_dir, "stds.npy")

        self.dtype = torch.get_default_dtype()
        self.u_transform = True
        self.layer_keys = ["layer_0", "layer_1", "layer_2", "extra_dims"]
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

    def __call__(self, data_dict, rev=False, rank=0):
        if rev:
            for key in self.layer_keys:
                data_dict[key] = data_dict[key] * self.std + self.mean
        else:
            if not self.written:
                shower = torch.cat([data_dict[key] for key in self.layer_keys], dim=1)
                nonzero_mask = (shower > self.eps) & (shower < -self.eps)
                self.mean = (shower[nonzero_mask]).mean()
                self.std = (shower[nonzero_mask]).std()
                if rank == 0:
                    self.write()
                self.written = True
            for key in self.layer_keys:
                data_dict[key] = (data_dict[key] - self.mean) / self.std
        return data_dict


class LogEnergyGAN:
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


class ScaleEnergyGAN:
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


class ExclusiveLogitTransformGAN:
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


class NormalizeLayerEnergyGAN:
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

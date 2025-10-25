import torch
import numpy as np
import os


def logit(array, alpha=1.0e-6, inv=False):
    if inv:
        z = torch.sigmoid(array)
        z = (z - alpha) / (1 - 2 * alpha)
    else:
        z = array * (1 - 2 * alpha) + alpha
        z = torch.logit(z)
    return z


class LEMURSGlobalStandardizeFromFile(object):
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
        self.keys = ["showers", "extra_dims"]
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
            for key in self.keys:
                data_dict[key] = data_dict[key] * self.std + self.mean
        else:
            if not self.written:
                shower = torch.cat([data_dict[key].flatten() for key in self.keys])
                self.mean = shower.mean()
                self.std = shower.std()
                if rank == 0:
                    self.write()
                self.written = True
            for key in self.keys:
                data_dict[key] = (data_dict[key] - self.mean) / self.std
        return data_dict


class LEMURSStandardizeUsFromFile(object):
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

    def __call__(self, data_dict, rev=False, rank=0):
        us = data_dict["extra_dims"]
        if rev:
            trafo_us = us * self.std_u.to(us.device) + self.mean_u.to(us.device)
        else:
            if not self.written:
                self.mean_u = us.mean(0)
                self.std_u = us.std(0)
                if rank == 0:
                    self.write()
                self.written = True
            trafo_us = (us - self.mean_u.to(us.device)) / self.std_u.to(us.device)
        data_dict["extra_dims"] = trafo_us
        return data_dict


class LEMURSPreprocessConds(object):
    """
    Apply preprocessing steps to the conditions.
    Scale all conditions to [0,1]. Incident energy is in linear scale.
    """

    def __init__(
        self, scale_E=[1e3, 1e6], scale_theta=[0.87, 2.27], scale_phi=[0, 3.1416]
    ):
        self.cond_transform = True
        self.keys = ["incident_energy", "incident_theta", "incident_phi"]
        self.rescaling = [scale_E, scale_theta, scale_phi]

    def __call__(self, data_dict, rev=False, rank=0):
        if rev:
            # rescale all conditions back to original range
            for n, key in enumerate(self.keys):
                min = self.rescaling[n][0]
                max = self.rescaling[n][1]
                data_dict[key] = data_dict[key] * (max - min) + min
        else:
            # Rescale all conditions
            for n, key in enumerate(self.keys):
                min = self.rescaling[n][0]
                max = self.rescaling[n][1]
                data_dict[key] = (data_dict[key] - min) / (max - min)
        return data_dict


class LEMURSScaleTotalEnergy(object):
    """
    Scale only E_tot/E_inc by a factor f.
    The effect is the same of scaling the voxels but
    it is applied in a different position in the
    preprocessing chain.
    """

    def __init__(self, factor):
        self.factor = factor
        self.u_transform = True

    def __call__(self, data_dict, rev=False, rank=0):
        u_0 = data_dict["extra_dims"][..., 0]
        if rev:
            u_0 /= self.factor
        else:
            u_0 *= self.factor
        data_dict["extra_dims"][..., 0] = u_0
        return data_dict


class LEMURSExclusiveLogitTransform(object):
    """
    Take log of input data
        delta: regularization
        exclusions: list of indices for features that should not be transformed
    """

    def __init__(self, delta, rescale=False):
        self.delta = delta
        self.rescale = rescale
        self.u_transform = True
        self.keys = ["showers", "extra_dims"]

    def __call__(self, data_dict, rev=False, rank=0):
        for key in self.keys:
            if rev:
                if self.rescale:
                    data_dict[key] = logit(data_dict[key], alpha=self.delta, inv=True)
                else:
                    data_dict[key] = torch.special.expit(data_dict[key])
            else:
                if self.rescale:
                    data_dict[key] = logit(data_dict[key], alpha=self.delta)
                else:
                    data_dict[key] = torch.logit(data_dict[key], eps=self.delta)
        return data_dict


class LEMURSCutValues(object):
    """
    Cut in Normalized space
        cut: threshold value for the cut
        n_layers: number of layers to avoid cutting on the us
    """

    def __init__(self, cut=0.0):
        self.cut = cut

    def __call__(self, data_dict, rev=False, rank=0):
        shower = data_dict["showers"]
        if rev:
            mask = shower <= self.cut
            transformed = shower
            if self.cut:
                transformed[mask] = 0.0
        else:
            transformed = shower
        data_dict["showers"] = transformed
        return data_dict


class LEMURSNormalizeByElayer(object):
    """
    Normalize each shower by the layer energy.
    This will change the shower shape to N_voxels+N_layers.
    This version is fully vectorized to avoid Python loops for performance.
    """

    def __init__(self, cut=0.0, eps=1.0e-10):
        self.eps = eps
        self.cut = cut

    def __call__(self, data_dict, rev=False, rank=0):
        shower = data_dict["showers"]
        B, H, W, L = shower.shape  # Batch, Height, Width, Layers

        if rev:
            us = data_dict["extra_dims"]
            energy = data_dict["incident_energy"]

            # Clip u_{i>0} into [0,1]
            us[:, 1:] = torch.clamp(us[:, 1:], min=0.0, max=1.0)

            layer_Es = []
            total_E = energy.flatten() * us[:, 0]
            remaining_E = total_E.clone()
            for i in range(L - 1):
                layer_E = remaining_E * us[:, i + 1]
                layer_Es.append(layer_E)
                remaining_E -= layer_E
            layer_Es.append(remaining_E)  # The last layer's energy
            layer_Es = torch.stack(layer_Es, dim=1)

            layer_Es_reshaped = layer_Es.view(B, 1, 1, L)

            # Normalize each layer and multiply it with its original energy
            layer_sums = shower.sum(dim=(1, 2), keepdim=True) + self.eps
            shower /= layer_sums

            if self.cut > 0.0:
                mask = shower <= self.cut
                shower[mask] = 0.0  # apply normalized cut

            shower *= layer_Es_reshaped  # scale all layers at once

        else:
            layer_Es = shower.sum(dim=(1, 2))
            shower /= layer_Es.view(B, 1, 1, L) + self.eps
            u_0 = layer_Es.sum(dim=1, keepdim=True) / (
                data_dict["incident_energy"] + self.eps
            )

            # For u_1, u_2, ... u_{L-1}
            # We need E_l / (E_l + E_{l+1} + ... + E_{L-1})
            # The denominator is a cumulative sum from right to left.
            remaining_E = torch.cumsum(layer_Es.flip(dims=[1]), dim=1).flip(dims=[1])

            # Calculate u_i for i > 0. Shape: (B, L-1)
            us_rest = layer_Es[:, :-1] / (remaining_E[:, :-1] + self.eps)
            extra_dims = torch.cat([u_0, us_rest], dim=1)

            data_dict["extra_dims"] = extra_dims

        data_dict["showers"] = shower
        return data_dict

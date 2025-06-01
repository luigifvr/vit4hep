import math
import torch
import torch.nn as nn


class BaseModel(nn.Module):
    def __init__(self, shape):
        super().__init__()

        self.shape = shape

    def forward(self, x, c, rev=False, jac=True):
        z, log_jac = self.net.forward(x, c, rev=rev, jac=jac)
        return z, log_jac

    def _batch_loss(self, x):
        return -self.log_prob(x[0], x[1])  # neg log prob

    def log_prob(self, x, c):
        """
        evaluate conditional log-likelihoods for given samples and conditions

        Parameters:
        x (tensor): Samples
        c (tensor): Conditions

        Returns:
        tensor: Log-likelihoods
        """
        z, log_jac_det = self.forward(x, c, rev=False)
        z = z.reshape(-1, math.prod(self.shape))
        log_prob = (
            -0.5 * torch.sum(z**2, 1)
            + log_jac_det
            - z.shape[1] / 2 * math.log(2 * math.pi)
        )
        return log_prob.mean()

    def sample_batch(self, z, batch):
        """
        sample from the learned distribution

        Parameters:
        num_pts (int): Number of samples to generate for each given condition
        condition (tensor): Conditions

        Returns:
        tensor[len(condition), num_pts, dims]: Samples
        """
        c = batch
        x, _ = self.forward(z, c, rev=True)
        return x
    
    def from_patches(self):
        raise NotImplementedError

    def to_patches(self):
        raise NotImplementedError


class CINN(BaseModel):
    def __init__(self, patch_shape, in_channels=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.patch_shape = patch_shape
        self.num_patches = [s // p for s, p in zip(self.shape, self.patch_shape)]
        self.in_channels = in_channels

        for i, (s, p) in enumerate(zip(self.shape, self.patch_shape)):
            assert (
                s % p == 0
            ), f"Input size ({s}) should be divisible by patch size ({p}) in axis {i}."

        self.net = None

    def forward(self, x, c, rev=False, jac=True):
        x = self.to_patches(x)
        z, log_jac = super().forward(x, c, rev=rev, jac=jac)
        z = self.from_patches(z)
        return z, log_jac

    def sample_batch(self, batch):
        """
        sample from the learned distribution

        Parameters:
        num_pts (int): Number of samples to generate for each given condition
        condition (tensor): Conditions

        Returns:
        tensor[len(condition), num_pts, dims]: Samples
        """
        z = torch.normal(
            0,
            1,
            size=(batch.shape[0], self.in_channels, *self.shape),
            device=batch.device,
            dtype=batch.dtype,
        )
        x = super().sample_batch(z, batch)
        return x.reshape(z.shape[0], self.in_channels, *self.shape)

    def build_net(self):
        raise NotImplementedError

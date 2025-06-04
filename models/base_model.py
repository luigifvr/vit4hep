import math
import torch
import torch.nn as nn
from torchdiffeq import odeint

from models.trajectories import linear_trajectory


class BaseModel(nn.Module):
    def __init__(self, shape):
        super().__init__()

        self.shape = shape

    def forward(self, x, c, rev=False, jac=True):
        z, log_jac = self.net.forward(x, c, rev=rev, jac=jac)
        return z, log_jac

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

    def _batch_loss(
        self,
    ):
        raise NotImplementedError

    def log_prob(self):
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

    def forward(self, x, c, rev=False, jac=True):
        x = self.to_patches(x)
        z, log_jac = super().forward(x, c, rev=rev, jac=jac)
        z = self.from_patches(z)
        return z, log_jac

    @torch.inference_mode()
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

    def _batch_loss(self, x):
        x = x[0].to(self.device, self.dtype)
        c = x[1].to(self.device, self.dtype)
        return -self.log_prob(x, c)  # neg log prob

    def build_net(self):
        raise NotImplementedError


class CFM(BaseModel):
    """
    Base class for a Conditional Flow Matching model
    """

    def __init__(
        self,
        patch_shape,
        in_channels=1,
        time_distribution="uniform",
        trajectory="linear",
        odeint_kwargs=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.time_distribution = self.get_time_distribution(time_distribution)
        self.trajectory = self.get_trajectory(trajectory)
        self.odeint_kwargs = odeint_kwargs

        self.patch_shape = patch_shape
        self.num_patches = [s // p for s, p in zip(self.shape, self.patch_shape)]
        self.in_channels = in_channels

        for i, (s, p) in enumerate(zip(self.shape, self.patch_shape)):
            assert (
                s % p == 0
            ), f"Input size ({s}) should be divisible by patch size ({p}) in axis {i}."

        self.net = None

    def get_trajectory(self, trajectory):
        if trajectory == "linear":
            return linear_trajectory
        else:
            raise ValueError

    def get_time_distribution(self, time_distribution):
        if time_distribution == "uniform":
            distr = torch.distributions.uniform.Uniform(low=0.0, high=1.0)
            return distr
        else:
            raise ValueError

    def forward(self, x, t, c):
        x = self.to_patches(x)
        z = self.net(x, t, c)
        z = self.from_patches(z)
        return z

    def _batch_loss(self, x):
        # get input and conditions
        x, c = x[0], x[1]
        x = x.to(self.device, self.dtype)
        c = c.to(self.device, self.dtype)

        t = self.time_distribution.sample([x.shape[0]] + [1] * (x.dim() - 1))
        t = t.to(self.device, self.dtype)

        x_0 = torch.randn_like(x)

        x_t, x_t_dot = self.trajectory(x_0, x, t)
        velocity = self.forward(x_t, t.view(-1, 1), c)

        loss = (velocity - x_t_dot) ** 2
        return loss.mean()

    @torch.inference_mode()
    def sample_batch(self, batch):
        """
        Generate n_samples new samples.
        Start from Gaussian random noise and solve the reverse ODE to obtain samples
        """
        dtype = batch.dtype
        device = batch.device

        x_T = torch.randn((batch.shape[0], self.in_channels, *self.shape), dtype=dtype, device=device)

        def f(t, x_t):
            t_torch = t.repeat((x_t.shape[0], 1)).to(self.device)
            return self.forward(x_t, t_torch, batch)

        solver = odeint  # also sdeint is possible

        sample = solver(
            f,
            x_T,
            torch.tensor([0.0, 1.0], dtype=dtype, device=device),  # (t_min, t_max)
            **self.odeint_kwargs,
        )[-1]

        return sample

    def build_net(self):
        raise NotImplementedError

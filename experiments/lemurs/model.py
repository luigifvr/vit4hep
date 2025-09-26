from einops import rearrange
import torch
from torchdiffeq import odeint

from models.base_model import CFM


class LEMURSCFM(CFM):
    def __init__(
        self,
        net,
        patch_shape,
        in_channels=1,
        time_distribution="uniform",
        trajectory="linear",
        odeint_kwargs=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            None,
            time_distribution,
            trajectory,
            odeint_kwargs,
            *args,
            **kwargs,
        )

        self.patch_shape = patch_shape
        self.num_patches = [s // p for s, p in zip(self.shape, self.patch_shape)]
        self.in_channels = in_channels

        for i, (s, p) in enumerate(zip(self.shape, self.patch_shape)):
            assert (
                s % p == 0
            ), f"Input size ({s}) should be divisible by patch size ({p}) in axis {i}."

        self.net = net

    def from_patches(self, x):
        x = rearrange(
            x,
            "b (l a r) (p1 p2 p3 c) -> b c (l p1) (a p2) (r p3)",
            **dict(
                zip(
                    ("l", "a", "r", "p1", "p2", "p3"),
                    self.num_patches + self.patch_shape,
                )
            ),
        )
        return x

    def to_patches(self, x):
        x = rearrange(
            x,
            "b c (l p1) (a p2) (r p3) -> b (l a r) (p1 p2 p3 c)",
            **dict(zip(("p1", "p2", "p3"), self.patch_shape)),
        )
        return x

    def _batch_loss(self, x):
        x[0] = x[0].permute(0, 3, 2, 1)  # 3: layers -> 1: radius
        x[0] = x[0].unsqueeze(1)  # add channel dimension
        return super()._batch_loss(x)

    def forward(self, x, t, c):
        x = self.to_patches(x)
        z = self.net(x, t, c)
        z = self.from_patches(z)
        return z

    @torch.inference_mode()
    def sample_batch(self, batch):
        """
        Generate n_samples new samples.
        Start from Gaussian random noise and solve the reverse ODE to obtain samples
        """
        dtype = batch.dtype
        device = batch.device

        x_T = torch.randn(
            (batch.shape[0], self.in_channels, *self.shape), dtype=dtype, device=device
        )

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

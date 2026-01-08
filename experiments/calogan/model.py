import torch
from einops import rearrange
from torchdiffeq import odeint

from models.base_model import CFM


class CaloGANCFM(CFM):
    def __init__(
        self,
        net,
        list_shape,
        list_edges,
        list_patch_shape,
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

        self.list_shape = list(list_shape)
        self.list_edges = list(list_edges)
        self.list_patch_shape = list(list_patch_shape)
        self.in_channels = in_channels

        self.num_patches_per_dim = []
        self.num_patches_per_layer = []
        for shape, patch_shape in zip(self.list_shape, self.list_patch_shape, strict=True):
            num_patches_dim = (
                (shape[0] // patch_shape[0]),
                (shape[1] // patch_shape[1]),
                (shape[2] // patch_shape[2]),
            )
            num_patches = num_patches_dim[0] * num_patches_dim[1] * num_patches_dim[2]
            self.num_patches_per_dim.append(num_patches_dim)
            self.num_patches_per_layer.append(num_patches)

        assert len(list_shape) == len(list_patch_shape), (
            "list_shape and list_patch_shape must have the same length"
        )
        for i, (s, p) in enumerate(zip(self.list_shape, self.list_patch_shape, strict=True)):
            for L, m in zip(s, p, strict=True):
                assert L % m == 0, (
                    f"Input size ({L}) should be divisible by patch size ({m}) in axis {i}."
                )

        self.net = net
        self.net.num_patches = self.num_patches_per_dim

    def from_patches(self, x):
        x_split = list(torch.split(x, self.num_patches_per_layer, dim=1))

        for k, patch_shape in enumerate(self.list_patch_shape):
            x_split[k] = rearrange(
                x_split[k],
                "b (l a r) (p1 p2 p3 c) -> b c (l p1) (a p2) (r p3)",
                **dict(zip(("l", "a", "r"), self.num_patches_per_dim[k], strict=True)),
                **dict(zip(("p1", "p2", "p3"), patch_shape, strict=True)),
                c=self.in_channels,
            )

            x_split[k] = x_split[k].flatten(start_dim=2)

        x_reconstructed = torch.cat(x_split, dim=2)
        return x_reconstructed

    def to_patches(self, x):
        x_split = list(torch.split(x, self.list_edges, dim=2))
        for k, shape in enumerate(self.list_shape):
            x_split[k] = x_split[k].reshape(-1, self.in_channels, *shape)
            x_split[k] = rearrange(
                x_split[k],
                "b c (l p1) (a p2) (r p3) -> b (l a r) (p1 p2 p3 c)",
                **dict(zip(("p1", "p2", "p3"), self.list_patch_shape[k], strict=True)),
            )
        x = torch.cat(x_split, dim=1)
        return x

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

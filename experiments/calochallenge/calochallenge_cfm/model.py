import torch
from einops import rearrange
from torchdiffeq import odeint

from models.base_model import CFM


class CaloChallengeCFM(CFM):
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
        self.num_patches = [s // p for s, p in zip(self.shape, self.patch_shape, strict=False)]
        self.in_channels = in_channels

        for i, (s, p) in enumerate(zip(self.shape, self.patch_shape, strict=False)):
            assert s % p == 0, (
                f"Input size ({s}) should be divisible by patch size ({p}) in axis {i}."
            )

        self.net = net

    def from_patches(self, x):
        x = rearrange(
            x,
            "b (l a r) (p1 p2 p3 c) -> b c (l p1) (a p2) (r p3)",
            **dict(
                zip(
                    ("l", "a", "r", "p1", "p2", "p3"),
                    self.num_patches + self.patch_shape,
                    strict=False,
                )
            ),
        )
        return x

    def to_patches(self, x):
        x = rearrange(
            x,
            "b c (l p1) (a p2) (r p3) -> b (l a r) (p1 p2 p3 c)",
            **dict(zip(("p1", "p2", "p3"), self.patch_shape, strict=False)),
        )
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


class CaloChallengeCFM_DS1(CaloChallengeCFM):
    def __init__(
        self,
        net,
        list_shape,
        list_edges,
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
            patch_shape,
            in_channels,
            time_distribution,
            trajectory,
            odeint_kwargs,
            *args,
            **kwargs,
        )

        self.list_shape = list(list_shape)
        self.list_edges = list(list_edges)

        self.num_patches_per_dim = []
        self.num_patches_per_layer = []
        for shape in self.list_shape:
            num_patches_dim = (
                (shape[0] // self.patch_shape[0]),
                (shape[1] // self.patch_shape[1]),
                (shape[2] // self.patch_shape[2]),
            )
            num_patches = num_patches_dim[0] * num_patches_dim[1] * num_patches_dim[2]
            self.num_patches_per_dim.append(num_patches_dim)
            self.num_patches_per_layer.append(num_patches)

        for i, s in enumerate(self.list_shape):
            for L, m in zip(s, patch_shape, strict=False):
                assert L % m == 0, (
                    f"Input size ({L}) should be divisible by patch size ({m}) in axis {i}."
                )

        self.net = net
        self.net.num_patches = self.num_patches_per_dim

    def from_patches(self, x):
        x_split = list(torch.split(x, self.num_patches_per_layer, dim=1))

        for k in range(len(self.num_patches_per_layer)):
            x_split[k] = rearrange(
                x_split[k],
                "b (l a r) (p1 p2 p3 c) -> b c (l p1) (a p2) (r p3)",
                **dict(zip(("l", "a", "r"), self.num_patches_per_dim[k], strict=False)),
                **dict(zip(("p1", "p2", "p3"), self.patch_shape, strict=False)),
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
                **dict(zip(("p1", "p2", "p3"), self.patch_shape, strict=False)),
            )
        x = torch.cat(x_split, dim=1)
        return x

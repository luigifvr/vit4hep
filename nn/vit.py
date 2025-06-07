"""Modified from github.com/facebookresearch/DiT/blob/main/models.py"""

import math
import torch
import torch.nn as nn

from einops import rearrange
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Mlp
from xformers.ops import memory_efficient_attention


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ViT(nn.Module):
    """
    Vision transformer-based diffusion network.
    """

    def __init__(self, param):

        super(ViT, self).__init__()

        defaults = {
            "dim": 3,
            "condition_dim": 46,
            "hidden_dim": 180,
            "out_channels": 1,
            "depth": 2,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "attn_drop": 0.0,
            "proj_drop": 0.0,
            "pos_embedding_coords": "cartesian",
            "temperature": 10000,
            "learn_pos_embed": True,
            "causal_attn": False,
            "checkpoint_grads": False,
            "patch_dim": 12,
            "num_patches": [15, 4, 9],
            "use_torch_sdpa": True,
        }

        for k, p in defaults.items():
            setattr(self, k, param[k] if k in param else p)

        # initialize x,t,c embeddings
        self.x_embedder = nn.Linear(self.patch_dim, self.hidden_dim)
        self.c_embedder = nn.Sequential(
            nn.Linear(self.condition_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.t_embedder = TimestepEmbedder(self.hidden_dim)

        # initialize position embeddings
        if self.learn_pos_embed:
            self.pos_embed_freqs = nn.Parameter(torch.randn(self.hidden_dim // 2))
            l, a, r = self.num_patches
            self.register_buffer("lgrid", torch.arange(l) / l)
            self.register_buffer("agrid", torch.arange(a) / a)
            self.register_buffer("rgrid", torch.arange(r) / r)
        else:
            self.register_buffer(
                "pos_embed",
                (
                    get_sincos_pos_embed(
                        self.pos_embedding_coords,
                        self.num_patches,
                        self.hidden_dim,
                        self.dim,
                        self.temperature,
                    )
                ),
            )

        # compute layer-causal attention mask
        if self.causal_attn:
            l, a, r = self.num_patches
            assert (
                self.dim == 3
            ), "A layer-causal attention mask should only be used in 3d"
            patch_idcs = torch.arange(l * a * r)
            self.attn_mask = nn.Parameter(
                patch_idcs[:, None] // (a * r)
                >= patch_idcs[None, :] // (a * r),  # tril (causal)
                requires_grad=False,
            )

        # initialize transformer stack
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    self.hidden_dim,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    attn_drop=self.attn_drop,
                    proj_drop=self.proj_drop,
                    attn_mask=self.attn_mask if self.causal_attn else None,
                    use_torch_sdpa=self.use_torch_sdpa,
                )
                for _ in range(self.depth)
            ]
        )

        # initialize output layer
        # TODO: final conv for ViT INN ?
        self.final_layer = FinalLayer(
            self.hidden_dim, self.patch_dim, self.out_channels, x_out=1
        )

        # custom weight initialization
        self.initialize_weights()

    def learnable_pos_embedding(self):  # TODO
        wz, wy, wx = (self.pos_embed_freqs * 2 * math.pi).chunk(3)
        z, y, x = torch.meshgrid(self.lgrid, self.agrid, self.rgrid, indexing="ij")
        z = z.flatten()[:, None] * wz[None, :]
        y = y.flatten()[:, None] * wy[None, :]
        x = x.flatten()[:, None] * wx[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim=1)
        return pe

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Forward pass of DiT.
        x: (B, C, *axis_sizes) tensor of spatial inputs
        t: (B,) tensor of diffusion timesteps
        c: (B, K) tensor of conditions
        """
        if self.learn_pos_embed:
            x = self.x_embedder(x) + self.learnable_pos_embedding()
        else:
            x = (
                self.x_embedder(x) + self.pos_embed
            )  # (B, T, D), where T = (L*A*R)/prod(patch_size)

        t = self.t_embedder(t)  # (B, D)
        c = self.c_embedder(c)  # (B, D)
        c = t + c
        for block in self.blocks:
            if self.checkpoint_grads:
                x = checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)  # (B, T, D)
        x = self.final_layer(x, c)  # (B, T, prod(patch_shape) * out_channels)
        return x


class ViT1D(ViT):
    """
    Vision transformer-based diffusion network.
    """

    def __init__(self, param):

        super().__init__(param)
        defaults = {
            "prod_num_patches": 15 * 4 * 9,  # TODO num_patches not defined
            "x_out": None,
        }

        for k, p in defaults.items():
            setattr(self, k, param[k] if k in param else p)

        # initialize position embeddings
        if self.learn_pos_embed:
            self.pos_embed_freqs = nn.Parameter(torch.randn(self.hidden_dim // 2))
            self.register_buffer(
                "grid", torch.arange(self.prod_num_patches) / self.prod_num_patches
            )
        else:
            self.register_buffer(
                "pos_embed",
                (
                    get_sincos_pos_embed(
                        self.pos_embedding_coords,
                        self.num_patches,
                        self.hidden_dim,
                        self.dim,
                        self.temperature,
                    )
                ),
            )

        # initialize transformer stack
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    self.hidden_dim,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    attn_drop=self.attn_drop,
                    proj_drop=self.proj_drop,
                    attn_mask=self.attn_mask if self.causal_attn else None,
                    use_torch_sdpa=self.use_torch_sdpa,
                )
                for _ in range(self.depth)
            ]
        )

        # initialize output layer
        # TODO: final conv for ViT INN ?
        self.final_layer = FinalLayer(
            self.hidden_dim, self.patch_dim, self.out_channels, self.x_out
        )

        # custom weight initialization
        self.initialize_weights()

    def learnable_pos_embedding(self):  # TODO
        wgrid = self.pos_embed_freqs * 2 * math.pi
        pos = self.grid[:, None] * wgrid[None, :]
        pe = torch.cat((pos.sin(), pos.cos()), dim=1)
        return pe

    def forward(self, x, c):
        """
        Forward pass of DiT.
        x: (B, C, *axis_sizes) tensor of spatial inputs
        c: (B, K) tensor of conditions
        """
        if self.learn_pos_embed:
            x = self.x_embedder(x) + self.learnable_pos_embedding()
        else:
            x = (
                self.x_embedder(x) + self.pos_embed
            )  # (B, T, D), where T = (L*A*R)/prod(patch_size)

        c = self.c_embedder(c)  # (B, D)
        for block in self.blocks:
            if self.checkpoint_grads:
                x = checkpoint(block, x, c, use_reentrant=False)
            else:
                x = block(x, c)  # (B, T, D)
        x = self.final_layer(x, c)  # (B, T, prod(patch_shape) * out_channels)
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_dim, patch_dim, out_channels=1, x_out=1):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_channels * x_out * patch_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t.float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Attention(nn.Module):

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        attn_mask: torch.Tensor = None,
        norm_layer: nn.Module = nn.LayerNorm,
        use_torch_sdpa: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_mask = attn_mask
        self.use_torch_sdpa = use_torch_sdpa

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_torch_sdpa:
            x = nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=self.attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            x = memory_efficient_attention(
                q.contiguous(),
                k.contiguous(),
                v.contiguous(),
                p=self.attn_drop.p if self.training else 0.0,
            )
            x = x.transpose(1, 2)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def get_sincos_pos_embed(
    pos_embedding_coords, num_patches, hidden_dim, dim, temperature=10000
):
    if pos_embedding_coords == "cylindrical" and dim == 3:
        pe = get_3d_cylindrical_sincos_pos_embed(num_patches, hidden_dim, temperature)
    elif pos_embedding_coords == "cartesian" and dim == 3:
        pe = get_3d_cartesian_sincos_pos_embed(num_patches, hidden_dim, temperature)
    elif pos_embedding_coords == "cylindrical" and dim == 1:
        pe = get_1d_cylindrical_sincos_pos_embed(num_patches, hidden_dim, temperature)
    else:
        raise ValueError
    return pe


def get_1d_cylindrical_sincos_pos_embed(num_patches, dim, temperature=10000):
    """
    Embeds patch positions based directly on input indices, which are assumed
    to be depth, angle, radius.
    """
    x = torch.arange(num_patches) / num_patches

    fourier_dim = dim // 2
    omega = torch.arange(fourier_dim) / (fourier_dim - 1)
    omega = 1.0 / (temperature**omega)
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat((x.sin(), x.cos()), dim=1)
    # padding can be implemented here

    return pe


def get_3d_cylindrical_sincos_pos_embed(num_patches, dim, temperature=10000):
    """
    Embeds patch positions based directly on input indices, which are assumed
    to be depth, angle, radius.
    """
    L, A, R = num_patches
    z, y, x = torch.meshgrid(
        torch.arange(L) / L, torch.arange(A) / A, torch.arange(R) / R, indexing="ij"
    )

    fourier_dim = dim // 6
    omega = torch.arange(fourier_dim) / (fourier_dim - 1)
    omega = 1.0 / (temperature**omega)
    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim=1)
    # padding can be implemented here

    return pe


def get_3d_cartesian_sincos_pos_embed(num_patches, dim, temperature=10000):
    """
    Embeds patch positions after converting input indices from polar to cartesian
    coordinates. i.e. depth, angle, radius -> depth, height, width
    """
    L, A, R = num_patches
    z, alpha, r = torch.meshgrid(
        torch.arange(L) / L,
        torch.arange(A) * (2 * math.pi / A),
        torch.arange(R) / R,
        indexing="ij",
    )
    x = r * alpha.cos()
    y = r * alpha.sin()

    fourier_dim = dim // 6
    omega = torch.arange(fourier_dim) / (fourier_dim - 1)
    omega = 1.0 / (temperature**omega)
    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim=1)
    # padding can be implemented here

    return pe

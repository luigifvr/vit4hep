import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from torchdiffeq import odeint


class ARtransformer(nn.Module):
    def __init__(self, params):
        super().__init__()
        # Read in the network specifications from the params
        self.params = params
        self.dim_embedding = self.params["dim_embedding"]
        self.dims_in = self.params["shape"][0]
        self.dims_c = self.params["n_con"]
        self.bayesian = False
        self.layer_cond = self.params.get("layer_cond", False)

        self.c_embed = self.params.get("c_embed", None)
        self.x_embed = self.params.get("x_embed", None)

        self.encode_t_dim = self.params.get("encode_t_dim", 64)
        self.encode_t_scale = self.params.get("encode_t_scale", 30)

        self.transformer = nn.Transformer(
            d_model=self.dim_embedding,
            nhead=params["n_head"],
            num_encoder_layers=params["n_encoder_layers"],
            num_decoder_layers=params["n_decoder_layers"],
            dim_feedforward=params["dim_feedforward"],
            dropout=params.get("dropout_transformer", 0.0),
            # activation=params.get("activation", "relu"),
            batch_first=True,
        )
        if self.x_embed:
            self.x_embed = nn.Sequential(
                nn.Linear(1, self.dim_embedding),
                nn.Linear(self.dim_embedding, self.dim_embedding),
            )
        if self.c_embed:
            self.c_embed = nn.Sequential(
                nn.Linear(1, self.dim_embedding),
                nn.ReLU(),
                nn.Linear(self.dim_embedding, self.dim_embedding),
            )
        self.t_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=self.encode_t_dim, scale=self.encode_t_scale),
            nn.Linear(self.encode_t_dim, self.encode_t_dim),
        )
        self.subnet = self.build_subnet()
        self.positional_encoding = PositionalEncoding(
            d_model=self.dim_embedding,
            max_len=max(self.dims_in, self.dims_c) + 1,
            dropout=0.0,
        )

    def compute_embedding(
        self, p: torch.Tensor, dim: int, embedding_net: nn.Module | None
    ) -> torch.Tensor:
        """
        Appends the one-hot encoded position to the momenta p. Then this is either zero-padded
        or an embedding net is used to compute the embedding of the correct dimension.
        """
        one_hot = torch.eye(dim, device=p.device, dtype=p.dtype)[None, : p.shape[1], :].expand(
            p.shape[0], -1, -1
        )
        if embedding_net is None:
            n_rest = self.dim_embedding - dim - p.shape[-1]
            assert n_rest >= 0
            zeros = torch.zeros((*p.shape[:2], n_rest), device=p.device, dtype=p.dtype)
            return torch.cat((p, one_hot, zeros), dim=2)
        else:
            return self.positional_encoding(embedding_net(p))

    def build_subnet(self):
        self.intermediate_dim = self.params.get("intermediate_dim", 512)
        self.dropout = self.params.get("dropout", 0.0)
        self.activation = self.params.get("activation", "SiLU")
        self.layers_per_block = self.params.get("layers_per_block", 8)
        self.normalization = self.params.get("normalization", None)

        cond_dim = self.encode_t_dim + self.dim_embedding
        if self.layer_cond:
            cond_dim += self.dims_in
        linear = nn.Linear(1 + cond_dim, self.intermediate_dim)
        layers = [linear, getattr(nn, self.activation)()]

        for _ in range(1, self.layers_per_block - 1):
            linear = nn.Linear(self.intermediate_dim, self.intermediate_dim)
            layers.append(linear)
            if self.normalization is not None:
                layers.append(getattr(nn, self.normalization)(self.intermediate_dim))
            if self.dropout is not None:
                layers.append(nn.Dropout(p=self.dropout))
            layers.append(getattr(nn, self.activation)())

        linear = nn.Linear(self.intermediate_dim, 1)
        layers.append(linear)

        return nn.Sequential(*layers)

    def sample_dimension(self, c: torch.Tensor):
        batch_size = c.size(0)
        dtype = c.dtype
        device = c.device

        net = self.subnet
        x_0 = torch.randn((batch_size, 1), device=device, dtype=dtype)

        # NN wrapper to pass into ODE solver
        def net_wrapper(t, x_t):
            t_torch = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)
            t_torch = self.t_embed(t_torch)
            v = net(torch.cat([x_t, t_torch.reshape(batch_size, -1), c.squeeze()], dim=-1))
            return v

        # Solve ODE from t=1 to t=0
        with torch.inference_mode():
            x_t = odeint(
                net_wrapper,
                x_0,
                torch.tensor([0, 1], dtype=dtype, device=device),
                **self.params.get("solver_kwargs", {}),
            )
        # Extract generated samples and mask out masses if not needed
        x_1 = x_t[-1]

        return x_1.unsqueeze(1)

    def forward(self, c, x_t=None, t=None, x=None, rev=False):
        if not rev:
            xp = nn.functional.pad(x[:, :-1], (0, 0, 1, 0))
            embedding = self.transformer(
                src=self.compute_embedding(c, dim=self.dims_c, embedding_net=self.c_embed),
                tgt=self.compute_embedding(xp, dim=self.dims_in + 1, embedding_net=self.x_embed),
                tgt_mask=torch.ones(
                    (xp.shape[1], xp.shape[1]), device=x.device, dtype=torch.bool
                ).triu(diagonal=1),
            )

            if self.layer_cond:
                layer_one_hot = repeat(
                    torch.eye(self.dims_in, device=x.device), "... -> b ...", b=len(c)
                )
                embedding = torch.cat([embedding, layer_one_hot], dim=2)

            t = self.t_embed(t)
            pred = self.subnet(torch.cat([x_t, t, embedding], dim=-1))

        else:
            x = torch.zeros((c.shape[0], 1, 1), device=c.device, dtype=c.dtype)
            c_embed = self.compute_embedding(c, dim=self.dims_c, embedding_net=self.c_embed)
            for i in range(self.dims_in):
                embedding = self.transformer(
                    src=c_embed,
                    tgt=self.compute_embedding(x, dim=self.dims_in + 1, embedding_net=self.x_embed),
                    tgt_mask=torch.ones(
                        (x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool
                    ).triu(diagonal=1),
                )
                if self.layer_cond:
                    layer_one_hot = repeat(
                        F.one_hot(torch.tensor(i, device=x.device), self.dims_in),
                        "d -> b 1 d",
                        b=len(c),
                    )
                    embedding = torch.cat([embedding[:, -1:, :], layer_one_hot], dim=2)
                x_new = self.sample_dimension(embedding[:, -1:, :])
                x = torch.cat((x, x_new), dim=1)

            pred = x[:, 1:]
            pred = pred.squeeze()

        return pred


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)

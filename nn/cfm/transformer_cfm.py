"""
Transformer-based MLP for CFM modified from CaloDREAM arXiv:2405.09629.
Used to generate energy ratios with a CFM generative network
"""

import torch
import torch.nn as nn
import math


class ParallelTransformer(nn.Module):
    """
    Predict velocity field for an entire vector in a single forward pass.
    The embedding is learned by a transformer network.
    """

    def __init__(self, param):
        super().__init__()
        # Read in the network specifications from the params
        defaults = {
            "dims_in": 46,
            "dims_c": 1,
            "dim_embedding": 180,
            "nhead": 4,
            "num_encoder_layers": 2,
            "num_decoder_layers": 4,
            "dim_feedforward": 256,
            "dropout": 0.0,
            "activation": "relu",
            "embeds": False,
            "encode_t_scale": 30,
            "encode_t_dim": 64,
        }

        for k, p in defaults.items():
            setattr(self, k, param[k] if k in param else p)

        self.time_embed = nn.Sequential(
            GaussianFourierProjection(
                embed_dim=self.encode_t_dim, scale=self.encode_t_scale
            ),
            nn.Linear(self.encode_t_dim, self.encode_t_dim),
        )
        if self.embeds:
            self.d_model = 2 * self.dim_embedding
            self.x_embed = nn.Linear(1, self.dim_embedding)
            self.c_embed = nn.Linear(1, 2 * self.dim_embedding)
            self.pos_embed_x = nn.Embedding(self.dims_in, self.dim_embedding)
            self.pos_embed_c = nn.Embedding(self.dims_c, 2 * self.dim_embedding)
            self.layer = nn.Linear(3 * self.dim_embedding, self.dim_feedforward)
        else:
            self.d_model = self.dim_embedding
            self.layer = nn.Linear(
                self.dim_embedding + self.encode_t_dim, self.dim_feedforward
            )

        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True,
        )

        self.layers = nn.Sequential(
            self.layer,
            nn.SiLU(),
            nn.Linear(self.dim_feedforward, 1),
        )

    def compute_embedding(
        self, p: torch.Tensor, n_components: int, t: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the embedding for the input vector p. Either use an embedding network
        or a vector with the positional encoding, the features, and zero padding.
        """
        if self.embeds:
            if t is not None:
                p = self.x_embed(p.unsqueeze(-1))
                p = p + self.pos_embed_x(torch.arange(n_components, device=p.device))
                t = self.time_embed(t).unsqueeze(1)
                p = torch.cat([t.repeat(1, p.size(1), 1), p], dim=-1)
                return p
            else:
                p = self.c_embed(p.unsqueeze(-1))
                p = p + self.pos_embed_c(torch.arange(n_components, device=p.device))
                return p
        else:
            p = p.unsqueeze(-1)
            one_hot = torch.eye(n_components, device=p.device, dtype=p.dtype)[
                None, : p.shape[1], :
            ].expand(p.shape[0], -1, -1)
            n_rest = self.dim_embedding - n_components - p.shape[-1]
            assert n_rest >= 0
            zeros = torch.zeros((*p.shape[:2], n_rest), device=p.device, dtype=p.dtype)
            return torch.cat((p, one_hot, zeros), dim=-1)

    def forward(self, x, t, condition=None):
        if condition is None:
            embedding = self.transformer.decoder(
                self.compute_embedding(x, n_components=self.dims_in, t=t),
                torch.zeros(
                    (x.size(0), x.size(1), 2 * self.dim_embedding),
                    device=x.device,
                    dtype=x.dtype,
                ),
            )
        else:
            embedding = self.transformer(
                src=self.compute_embedding(condition, n_components=self.dims_c),
                tgt=self.compute_embedding(x, n_components=self.dims_in, t=t),
            )

        t = self.time_embed(t)
        t = t.unsqueeze(1).expand(-1, embedding.size(1), -1)

        v_pred = self.layers(torch.cat([t, embedding], dim=-1))
        return v_pred.squeeze(-1)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class SinCos_embedding(nn.Module):

    def __init__(self, n_frequencies: int, sigmoid=True):
        super().__init__()
        self.arg = nn.Parameter(
            2 * math.pi * 2 ** torch.arange(n_frequencies), requires_grad=False
        )
        self.sigmoid = sigmoid

    def forward(self, x):
        if self.sigmoid:
            x_pp = nn.functional.sigmoid(x)
        else:
            x_pp = x
        frequencies = (x_pp.unsqueeze(-1) * self.arg).reshape(
            x_pp.size(0), x_pp.size(1), -1
        )
        return torch.cat([torch.sin(frequencies), torch.cos(frequencies)], dim=-1)


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)

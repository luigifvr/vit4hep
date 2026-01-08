import math

import torch
from einops import rearrange
from FrEIA.framework import ConditionNode, GraphINN, InputNode, Node, OutputNode
from torchdiffeq import odeint

from experiments.calochallenge.calochallenge_cinn.freia_utils import (
    get_block_kwargs,
    get_coupling_block,
    get_permutation_block,
    get_vit_block_kwargs,
)
from models.base_model import CINN


class CaloChallengeCINN(CINN):
    def __init__(
        self,
        patch_shape,
        in_channels,
        coupling_block,
        nblocks,
        is_spatial,
        cinn_kwargs,
        vit_kwargs,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.patch_shape = patch_shape
        self.num_patches = [s // p for s, p in zip(self.shape, self.patch_shape, strict=True)]
        self.in_channels = in_channels

        for i, (s, p) in enumerate(zip(self.shape, self.patch_shape, strict=True)):
            assert s % p == 0, (
                f"Input size ({s}) should be divisible by patch size ({p}) in axis {i}."
            )

        self.nblocks = nblocks
        self.CouplingBlock = get_coupling_block(coupling_block)
        self.PermuteBlock = get_permutation_block(nblocks, is_spatial)
        self.block_kwargs = get_vit_block_kwargs(
            nblocks,
            is_spatial,
            self.shape,
            self.patch_shape,
            cinn_kwargs,
            vit_kwargs,
        )

        assert len(self.block_kwargs) == len(self.PermuteBlock)
        assert len(self.block_kwargs) == self.nblocks

        self.net = self.build_net()

    def from_patches(self, x, dim=3):
        if dim == 3:
            x = rearrange(
                x,
                "b (l a r) (p1 p2 p3 c) -> b c (l p1) (a p2) (r p3)",
                **dict(
                    zip(
                        ("l", "a", "r", "p1", "p2", "p3"),
                        self.num_patches + self.patch_shape,
                        strict=True,
                    )
                ),
            )
        elif dim == 2:
            x = rearrange(
                x,
                "b (a r) (p1 p2 c) -> b c (a p1) (r p2)",
                **dict(
                    zip(("a", "r", "p1", "p2"), self.num_patches + self.patch_shape, strict=True)
                ),
            )
        else:
            raise ValueError(self.dim)
        return x

    def to_patches(self, x, dim=3):
        if dim == 3:
            x = rearrange(
                x,
                "b c (l p1) (a p2) (r p3) -> b (l a r) (p1 p2 p3 c)",
                **dict(zip(("p1", "p2", "p3"), self.patch_shape, strict=True)),
            )
        elif dim == 2:
            x = rearrange(
                x,
                "b c (a p1) (r p2) -> b (a r) (p1 p2 c)",
                **dict(zip(("p1", "p2"), self.patch_shape, strict=True)),
            )
        else:
            raise ValueError(dim)
        return x

    def build_net(self):
        self.in_dim = [math.prod(self.num_patches), math.prod(self.patch_shape)]

        nodes = [InputNode(*self.in_dim, name="Input")]
        cond_node = ConditionNode(1, name="cond")
        for i in range(self.nblocks):
            nodes.append(
                Node(
                    [nodes[-1].out0],
                    self.CouplingBlock,
                    self.block_kwargs[i],
                    conditions=cond_node,
                    name=f"block_{i}",
                )
            )
            nodes.append(
                Node(
                    [nodes[-1].out0],
                    self.PermuteBlock[i],
                    {},
                    name=f"permute_{i}",
                )
            )

        nodes.append(OutputNode([nodes[-1].out0], name="out"))
        nodes.append(cond_node)

        return GraphINN(nodes)

    def forward(self, x, c, rev=False, jac=True):
        x = self.to_patches(x)
        z, log_jac = super().forward(x, c, rev=rev, jac=jac)
        z = self.from_patches(z)
        return z, log_jac


class CaloChallengeEnergyCINN(CINN):
    def __init__(
        self,
        shape,
        coupling_block,
        nblocks,
        cinn_kwargs,
        subnet_kwargs,
    ):
        super().__init__(shape)

        self.nblocks = nblocks
        self.CouplingBlock = get_coupling_block(coupling_block)
        self.PermuteBlock = get_permutation_block(nblocks, is_spatial=None)
        self.block_kwargs = get_block_kwargs(nblocks, cinn_kwargs, subnet_kwargs)

        assert len(self.block_kwargs) == len(self.PermuteBlock)
        assert len(self.block_kwargs) == self.nblocks

        self.net = self.build_net()

    def build_net(self):
        nodes = [InputNode(*self.shape, name="Input")]
        cond_node = ConditionNode(1, name="cond")
        for i in range(self.nblocks):
            nodes.append(
                Node(
                    [nodes[-1].out0],
                    self.CouplingBlock,
                    self.block_kwargs[i],
                    conditions=cond_node,
                    name=f"block_{i}",
                )
            )
            nodes.append(
                Node(
                    [nodes[-1].out0],
                    self.PermuteBlock[i],
                    {},
                    name=f"permute_{i}",
                )
            )

        nodes.append(OutputNode([nodes[-1].out0], name="out"))
        nodes.append(cond_node)

        return GraphINN(nodes)

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
            size=(batch.shape[0], *self.shape),
            device=batch.device,
            dtype=batch.dtype,
        )
        x = super().sample_batch(z, batch)

        return x.reshape(z.shape[0], *self.shape)

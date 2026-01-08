import math

from FrEIA.modules import PermuteRandom

from nn.inn.permute import PermuteRandomS1
from nn.inn.rqs_nflows import (
    CaloRationalQuadraticSplineBlock,
    OneSidedCaloRationalQuadraticSplineBlock,
    SimpleRationalQuadraticSplineBlock,
)
from nn.inn.rqs_v2 import RationalQuadraticSpline
from nn.inn.subnets import SubnetMLP, SubnetViT


def get_coupling_block(coupling_block):
    """Returns the class and keyword arguments for different coupling block types"""
    if coupling_block == "CaloRQSplineNFlows":
        CouplingBlock = CaloRationalQuadraticSplineBlock
    elif coupling_block == "OneSidedCaloRQSplineNFlows":
        CouplingBlock = OneSidedCaloRationalQuadraticSplineBlock
    elif coupling_block == "CaloRQSplineFrEIA":
        CouplingBlock = RationalQuadraticSpline
    elif coupling_block == "RQSplineNFlows":
        CouplingBlock = SimpleRationalQuadraticSplineBlock
    else:
        raise ValueError(f"Unknown Coupling block type {coupling_block}")

    return CouplingBlock


def get_permutation_block(nblocks, is_spatial=None):
    """Returns the class and keyword arguments for different coupling block types"""
    PermuteBlocks = []
    for n in range(nblocks):
        if is_spatial is not None:
            if is_spatial[n]:
                PermuteBlocks.append(PermuteRandomS1)
            else:
                PermuteBlocks.append(PermuteRandom)
        else:
            PermuteBlocks.append(PermuteRandom)

    return PermuteBlocks


def get_vit_block_kwargs(nblocks, is_spatial, shape, patch_shape, cinn_kwargs, vit_kwargs):
    """Returns the class and keyword arguments for different coupling block types"""

    list_block_kwargs = []
    spatial_patch_dim = int(math.prod(patch_shape) / 2)
    patch_dim = int(math.prod(patch_shape))
    num_patches = [s // p for s, p in zip(shape, patch_shape, strict=False)]
    prod_num_patches = int(math.prod(num_patches))
    for spatial_split in is_spatial:
        block_kwargs = {}
        if spatial_split:

            def func(x_in, x_out):
                subnet = SubnetViT(
                    x_out=x_out,
                    patch_dim=spatial_patch_dim,
                    prod_num_patches=prod_num_patches,
                    **vit_kwargs,
                )
                return subnet

            block_kwargs["subnet_constructor"] = func
            block_kwargs["spatial"] = True
            block_kwargs.update(cinn_kwargs)
        else:

            def func(x_in, x_out):
                subnet = SubnetViT(
                    x_out=x_out,
                    patch_dim=patch_dim,
                    num_patches=num_patches,
                    prod_num_patches=prod_num_patches,
                    **vit_kwargs,
                )
                return subnet

            block_kwargs["subnet_constructor"] = func
            block_kwargs["spatial"] = False
            block_kwargs.update(cinn_kwargs)
        list_block_kwargs.append(block_kwargs)
    return list_block_kwargs


def get_block_kwargs(nblocks, cinn_kwargs, subnet_kwargs):
    """Returns the class and keyword arguments for different coupling block types"""

    list_block_kwargs = []
    for _ in range(nblocks):
        block_kwargs = {}

        def func(x_in, x_out):
            subnet = SubnetMLP(
                x_in=x_in,
                x_out=x_out,
                subnet_kwargs=subnet_kwargs,
            )
            return subnet

        block_kwargs["subnet_constructor"] = func
        block_kwargs.update(cinn_kwargs)

        list_block_kwargs.append(block_kwargs)
    return list_block_kwargs

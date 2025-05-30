import torch
import torch.nn as nn
from nn.vit import ViT

class SubnetViT(nn.Module):
    " A subnet constructor for a ViT"
    def __init__(self, x_out=None, shape=[1, 45, 16, 9], patch_shape=[3, 4, 1], spatial=False, **kwargs):
        super().__init__()

        vit_kwargs = {
            "shape": shape,
            "patch_shape": patch_shape,
            "x_out": x_out,
        }
        vit_kwargs.update(kwargs)
        self.vit = ViT(vit_kwargs).to(torch.float32)

    def forward(self, x, c):
        t = torch.ones(len(x), 1, dtype=torch.float, device=x.device) #TODO
        vit_output = self.vit(x.to(torch.float32), t, c[0].to(torch.float32))
        return vit_output
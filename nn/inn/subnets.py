import torch
import torch.nn as nn
from nn.vit import ViT1D


class SubnetViT(nn.Module):
    "A subnet constructor for a ViT"

    def __init__(self, x_out=None, patch_dim=24, num_patches=[15, 2, 9], prod_num_patches=15 * 2 * 9, **kwargs):
        super().__init__()

        vit_kwargs = {
            "patch_dim": patch_dim,
            "num_patches": num_patches,
            "prod_num_patches": prod_num_patches,
            "x_out": x_out,
        }
        vit_kwargs.update(kwargs)
        self.vit = ViT1D(vit_kwargs).to(torch.float32)

    def forward(self, x, c):
        vit_output = self.vit(x.to(torch.float32), c[0].to(torch.float32))
        return vit_output


class SubnetMLP(nn.Module):
    "A subnet constructor for a MLP"

    def __init__(
        self,
        x_in=None,
        x_out=None,
        subnet_kwargs=None,
    ):
        super().__init__()

        defaults = {
            "n_layers": 2,
            "hidden_channels": [128, 128],
            "dropout": 0.0,
        }

        self.x_in = x_in
        self.x_out = x_out
        for k, p in defaults.items():
            setattr(self, k, subnet_kwargs[k] if k in subnet_kwargs else p)

        self.layers = []
        self.layers.append(nn.Linear(self.x_in, self.hidden_channels[0]))
        self.layers.append(nn.ReLU())
        for n in range(self.n_layers - 1):
            self.layers.append(
                nn.Linear(self.hidden_channels[n], self.hidden_channels[n + 1])
            )
            if self.dropout > 0:
                self.layers.append(nn.Dropout(p=self.dropout))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(self.hidden_channels[n], self.x_out))

        self.mlp = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.mlp(x)

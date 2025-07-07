import os
from omegaconf import OmegaConf, open_dict
import torch
import torch.nn as nn
from torch_ema import ExponentialMovingAverage

from experiments.logger import LOGGER
from experiments.calochallenge.experiment import CaloChallenge
from nn.vit import FinalLayer, get_sincos_pos_embed


class CaloChallengeFT(CaloChallenge):
    """
    A class for fine tuning a neural network on a different CaloChallenge dataset
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        backbone_cfg = os.path.join(self.cfg.finetuning.backbone_cfg)
        self.backbone_cfg = OmegaConf.load(backbone_cfg)

        self.model_num_patches = self.cfg.model.net.param.num_patches
        self.model_patch_dim = self.cfg.model.net.param.patch_dim
        with open_dict(self.cfg):
            self.cfg.model.net = self.backbone_cfg.model.net
            self.cfg.ema = self.backbone_cfg.ema

    def init_model(self):
        super().init_model()

        if self.warm_start:
            return

        # load pretrained network
        model_path = os.path.join(
            self.backbone_cfg.run_dir,
            "models",
            f"model_run{self.backbone_cfg.run_idx}.pt",
        )
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)[
                "model"
            ]
        except FileNotFoundError:
            raise ValueError(f"Cannot load model from {model_path}")
        LOGGER.info(f"Loading pretrained model from {model_path}")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device, dtype=self.dtype)

        if self.cfg.finetuning.map_embedding:
            self.embedding = self.model.net.x_embedder
            self.embedding_mapper = nn.Linear(
                self.model_patch_dim, self.backbone_cfg.model.net.param.patch_dim
            ).to(self.device, dtype=self.dtype)
            LOGGER.info(
                (
                    f"Mapping embedding from {self.model_patch_dim} "
                    f"to {self.backbone_cfg.model.net.param.patch_dim}"
                )
            )
            self.model.net.x_embedder = nn.Sequential(
                self.embedding_mapper, self.embedding
            ).to(self.device, dtype=self.dtype)

        # define the positional embedding
        self.model.net.pos_embed = get_sincos_pos_embed(
            self.cfg.model.net.param.pos_embedding_coords,
            self.model_num_patches,
            self.cfg.model.net.param.hidden_dim,
            self.cfg.model.net.param.dim,
        ).to(self.device, dtype=self.dtype)

        # reinitialize final layer
        self.model.net.final_layer = FinalLayer(
            self.cfg.model.net.param.hidden_dim,
            self.model_patch_dim,
            self.cfg.model.net.param.out_channels,
        ).to(self.device, dtype=self.dtype)

        if self.cfg.ema:
            LOGGER.info(f"Re-initializing EMA")
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=self.cfg.training.ema_decay
            ).to(self.device)

    def _init_optimizer(self):
        # collect parameter lists
        params_embedder = (
            list(self.embedding.parameters())
            if self.cfg.finetuning.map_embedding
            else list(self.model.net.x_embedder.parameters())
        )

        params_backbone = (
            params_embedder
            + list(self.model.net.c_embedder.parameters())
            + list(self.model.net.t_embedder.parameters())
            + list(self.model.net.blocks.parameters())
        )
        if self.cfg.finetuning.map_embedding:
            params_embedder = self.embedding_mapper.parameters()
        params_head = self.model.net.final_layer.parameters()

        # assign parameter-specific learning rates
        param_groups = [
            {"params": params_backbone, "lr": self.cfg.finetuning.backbone_lr},
            {"params": params_head, "lr": self.cfg.finetuning.head_lr},
            (
                {"params": params_embedder, "lr": self.cfg.finetuning.embedder_lr}
                if self.cfg.finetuning.map_embedding
                else {}
            ),
        ]

        super()._init_optimizer(param_groups=param_groups)

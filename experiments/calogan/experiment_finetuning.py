import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf, open_dict
from torch_ema import ExponentialMovingAverage

from experiments.calogan.experiment import CaloGAN
from experiments.logger import LOGGER
from experiments.misc import remove_module_from_state_dict
from nn.vit import FinalLayer, get_sincos_pos_embed


class CaloGANFTCFM(CaloGAN):
    """
    A class for fine tuning a neural network on a different CaloChallenge dataset
    """

    def __init__(self, cfg, rank=0, world_size=1):
        super().__init__(cfg, rank, world_size)

        backbone_cfg = os.path.join(self.cfg.finetuning.backbone_cfg)
        self.backbone_cfg = OmegaConf.load(backbone_cfg)

        self.model_num_patches = self.cfg.model.net.param.num_patches
        self.model_patch_dim = self.cfg.model.net.param.patch_dim  # ft patch dim
        self.model_condition_dim = self.cfg.model.net.param.condition_dim
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
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)["model"]
        except FileNotFoundError as err:
            raise ValueError(f"Cannot load model from {model_path}") from err
        LOGGER.info(f"Loading pretrained model from {model_path}")
        state_dict = remove_module_from_state_dict(state_dict)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device, dtype=self.dtype)

        # add embedding layers
        self.add_embedding_layers()

        if self.cfg.ema:
            LOGGER.info("Re-initializing EMA")
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=self.cfg.training.ema_decay
            ).to(self.device)

        with open_dict(self.cfg):
            self.cfg.model.net.param.num_patches = self.model_num_patches
            self.cfg.model.net.param.patch_dim = self.model_patch_dim
            self.cfg.model.net.param.condition_dim = self.model_condition_dim

    def add_embedding_layers(self):
        """
        Add embedding layers to the model.
        This is necessary for fine-tuning on a different dataset.
        """
        if self.cfg.finetuning.map_x_embedding:
            self.embedding = self.model.net.x_embedder
            self.embedding_mapper = nn.Linear(
                self.model_patch_dim, self.backbone_cfg.model.net.param.patch_dim
            ).to(self.device, dtype=self.dtype)
            LOGGER.info(
                f"Mapping embedding from {self.model_patch_dim} "
                f"to {self.backbone_cfg.model.net.param.patch_dim}"
            )
            self.model.net.x_embedder = nn.Sequential(
                self.embedding_mapper, nn.SiLU(), self.embedding
            ).to(self.device, dtype=self.dtype)
        else:
            if self.cfg.finetuning.reinitialize_x_embedding:
                self.model.net.x_embedder = nn.Linear(
                    self.model_patch_dim,
                    self.cfg.model.net.param.hidden_dim,
                ).to(self.device, dtype=self.dtype)
            if self.cfg.finetuning.interpolate:
                x_embed_weights = self.model.net.x_embedder.weight
                print(x_embed_weights.shape)
                x_embed_weights = nn.functional.interpolate(
                    x_embed_weights.unsqueeze(1),
                    size=self.model_patch_dim,
                    mode="linear",
                ).squeeze(1)
                self.model.net.x_embedder.weight.data = x_embed_weights.data

        if self.cfg.finetuning.map_c_embedding:
            self.c_embedding = self.model.net.c_embedder
            self.c_embedding_mapper = nn.Linear(
                self.model_condition_dim,
                self.backbone_cfg.model.net.param.condition_dim,
            ).to(self.device, dtype=self.dtype)
            LOGGER.info(
                f"Mapping condition embedding from {self.model_condition_dim} "
                f"to {self.backbone_cfg.model.net.param.condition_dim}"
            )
            self.model.net.c_embedder = nn.Sequential(
                self.c_embedding_mapper, nn.SiLU(), self.c_embedding
            ).to(self.device, dtype=self.dtype)
        else:
            if self.cfg.finetuning.reinitialize_c_embedding:
                self.model.net.c_embedder = nn.Sequential(
                    nn.Linear(
                        self.model_condition_dim,
                        self.cfg.model.net.param.hidden_dim,
                    ),
                    nn.SiLU(),
                    nn.Linear(
                        self.cfg.model.net.param.hidden_dim,
                        self.cfg.model.net.param.hidden_dim,
                    ),
                ).to(self.device, dtype=self.dtype)
            if self.cfg.finetuning.interpolate:
                c_embed_weights = self.model.net.c_embedder[0].weight
                print(c_embed_weights.shape)
                c_embed_weights = nn.functional.interpolate(
                    c_embed_weights.unsqueeze(1),
                    size=self.model_condition_dim,
                    mode="linear",
                ).squeeze(1)
                print(c_embed_weights.shape)
                self.model.net.c_embedder[0].weight.data = c_embed_weights.data

        # define the positional embedding
        if self.model.net.learn_pos_embed:
            if self.cfg.finetuning.reinitialize_pos_embedding:
                L, a, r = self.model_num_patches
                self.model.net.lgrid = torch.arange(L, device=self.device, dtype=self.dtype) / L
                self.model.net.agrid = torch.arange(a, device=self.device, dtype=self.dtype) / a
                self.model.net.rgrid = torch.arange(r, device=self.device, dtype=self.dtype) / r
            else:
                pass
        else:
            self.model.net.pos_embed = get_sincos_pos_embed(
                self.cfg.model.net.param.pos_embedding_coords,
                self.model_num_patches,
                self.cfg.model.net.param.hidden_dim,
                self.cfg.model.net.param.dim,
            ).to(self.device, dtype=self.dtype)

        # reinitialize final layer
        if self.cfg.finetuning.reinitialize_final_layer:
            self.model.net.final_layer = FinalLayer(
                self.cfg.model.net.param.hidden_dim,
                self.model_patch_dim,
                self.cfg.model.net.param.out_channels,
            ).to(self.device, dtype=self.dtype)

    def _init_optimizer(self):
        # collect parameter lists
        if self.world_size > 1:
            params_embedder = (
                list(self.model.net.module.x_embedder.parameters())
                + list(self.model.net.module.c_embedder.parameters())
                + [self.model.net.module.pos_embed_freqs]
                if self.model.net.module.learn_pos_embed
                else []
            )

            params_backbone = list(self.model.net.module.t_embedder.parameters()) + list(
                self.model.net.module.blocks.parameters()
            )

            params_head = self.model.net.module.final_layer.parameters()
        else:
            params_embedder = (
                list(self.model.net.x_embedder.parameters())
                + list(self.model.net.c_embedder.parameters())
                + [self.model.net.pos_embed_freqs]
                if self.model.net.learn_pos_embed
                else []
            )

            params_backbone = list(self.model.net.t_embedder.parameters()) + list(
                self.model.net.blocks.parameters()
            )

            params_head = self.model.net.final_layer.parameters()

        # assign parameter-specific learning rates
        param_groups = [
            {"params": params_backbone, "lr": self.cfg.finetuning.backbone_lr},
            {"params": params_head, "lr": self.cfg.finetuning.head_lr},
            {"params": params_embedder, "lr": self.cfg.finetuning.embedder_lr},
        ]

        super()._init_optimizer(param_groups=param_groups)

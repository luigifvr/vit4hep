import os
from omegaconf import OmegaConf, open_dict

from experiments.logger import LOGGER
from experiments.calochallenge.experiment import CaloChallenge
from nn.vit import FinalLayer


class CaloChallengeFT(CaloChallenge):
    """
    A class for fine tuning a neural network on a different CaloChallenge dataset
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        backbone_cfg = os.path.join(self.cfg.finetuning.backbone_cfg)
        self.backbone_cfg = OmegaConf.load(backbone_cfg)

        with open_dict(self.cfg):
            self.cfg.model = self.backbone_cfg.model
            self.cfg.ema = self.backbone_cfg.ema

    def init_model(self):
        super().init_model()

        if self.warm_start:
            return

        # load pretrained network
        model_path = os.path.join(
            self.backbone_cfg.run_dir,
            "models",
            f"model_run{self.warmstart_cfg.run_idx}.pt",
        )
        try:
            state_dict = torch.load(model_path, map_location="cpu")["model"]
        except FileNotFoundError:
            raise ValueError(f"Cannot load model from {model_path}")
        LOGGER.info(f"Loading pretrained model from {model_path}")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device, dtype=self.dtype)

        # reinitialize final layer
        self.model.net.final_layer = FinalLayer(
            self.cfg.model.net.hidden_dim,
            self.cfg.model.net.patch_dim,
            self.cfg.model.net.out_channels,
        ).to(self.device)

        if self.cfg.ema:
            LOGGER.info(f"Re-initializing EMA")
            self.ema = ExponentialMovingAverage(
                self.model.parameters(), decay=self.cfg.training.ema_decay
            ).to(self.device)

    def _init_optimizer(self):
        # collect parameter lists
        params_backbone = (
            list(self.model.net.x_embedder.parameters())
            + list(self.model.net.c_embedder.parameters())
            + list(self.model.net.t_embedder.parameters())
            + list(self.model.net.pos_embed_freqs.parameters())
        )
        +list(self.model.net.blocks.parameters())

        params_head = self.model.net.final_layer.parameters()

        # assign parameter-specific learning rates
        param_groups = [
            {"params": params_backbone, "lr": self.cfg.finetuning.lr_backbone},
            {"params": params_head, "lr": self.cfg.finetuning.lr_head},
        ]

        super()._init_optimizer(param_groups=param_groups)

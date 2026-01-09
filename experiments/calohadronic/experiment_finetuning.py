import os
import time

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

from experiments.calohadronic.datasets import CaloHadCollator, CaloHadDataset
from experiments.calohadronic.experiment import CaloHadronic
from experiments.logger import LOGGER
from experiments.misc import remove_module_from_state_dict
from nn.vit import FinalLayer, get_sincos_pos_embed


class CaloHadronicFT(CaloHadronic):
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
            with open_dict(self.cfg):
                self.cfg.model.net.param.num_patches = self.model_num_patches
                self.cfg.model.net.param.patch_dim = self.model_patch_dim
                self.cfg.model.net.param.condition_dim = self.model_condition_dim
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
                c_embed_weights = nn.functional.interpolate(
                    c_embed_weights.unsqueeze(1),
                    size=self.model_condition_dim,
                    mode="linear",
                ).squeeze(1)
                self.model.net.c_embedder[0].weight.data = c_embed_weights.data

        # define the positional embedding
        self.model.net.num_patches = self.model_num_patches
        if self.model.net.learn_pos_embed:
            if self.cfg.finetuning.reinitialize_pos_embedding:
                pos_z, pos_y, pos_x = self.model.net.create_meshgrid()
                self.model.net.pos_z = pos_z.to(self.device, dtype=self.dtype)
                self.model.net.pos_y = pos_y.to(self.device, dtype=self.dtype)
                self.model.net.pos_x = pos_x.to(self.device, dtype=self.dtype)
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

    @torch.inference_mode()
    def sample_n(self):
        self.model.eval()

        t_0 = time.time()

        Einc = torch.tensor(
            np.random.uniform(10, 90, size=self.cfg.n_samples),
            dtype=self.dtype,
            device=self.device,
        ).unsqueeze(1)

        samples = {"energy": Einc}
        samples["extra_dims"] = torch.empty(
            self.cfg.model.shape[0], dtype=self.dtype, device=self.device
        )
        for fn in self.transforms:
            if hasattr(fn, "cond_transform"):
                samples = fn(samples)

        transformed_cond = torch.cat(
            (samples["energy"],),
            dim=-1,
        ).to(self.device)
        batchsize_sample = self.cfg.training.batchsize_sample
        transformed_cond_loader = DataLoader(
            dataset=transformed_cond, batch_size=batchsize_sample, shuffle=False
        )

        # sample u_i's if self is a shape model
        if self.cfg.model_type == "shape":
            if self.cfg.sample_us:
                u_samples = self.sample_us(transformed_cond_loader)
                transformed_cond = torch.cat([u_samples, transformed_cond], dim=1)

                # Add LEMURS conditions
                theta = self.cfg.gen_theta
                phi = self.cfg.gen_phi
                label = torch.tensor(self.cfg.gen_label, dtype=self.dtype).to(self.device)
                theta_tensor = torch.full(
                    (transformed_cond.shape[0], 1),
                    theta,
                    dtype=self.dtype,
                    device=self.device,
                )
                phi_tensor = torch.full(
                    (transformed_cond.shape[0], 1),
                    phi,
                    dtype=self.dtype,
                    device=self.device,
                )
                label_tensor = label.unsqueeze(0).repeat(transformed_cond.shape[0], 1)
                transformed_cond = torch.cat(
                    [transformed_cond, theta_tensor, phi_tensor, label_tensor], dim=1
                )
                transformed_cond_loader = DataLoader(
                    dataset=transformed_cond,
                    batch_size=batchsize_sample,
                    shuffle=False,
                )

                sample = torch.vstack(
                    [
                        self.model.sample_batch(c.to(self.device)).cpu()
                        for c in transformed_cond_loader
                    ]
                )
                conditions = transformed_cond
            else:
                # optionally use truth us
                transformed_cond = CaloHadDataset(
                    self.hdf5_dict_test,
                    dtype=self.dtype,
                    max_files_per_worker=self.max_files_per_worker,
                )
                test_collator = CaloHadCollator(
                    hdf5_train_dict=self.hdf5_dict_test,
                    transforms=self.transforms,
                    return_us=False,
                    dtype=self.dtype,
                    rank=self.rank,
                )
                # concatenate with Einc
                transformed_cond_loader = DataLoader(
                    dataset=transformed_cond,
                    batch_size=batchsize_sample,
                    shuffle=False,
                    collate_fn=test_collator,
                )

                conditions = torch.vstack([c[1] for c in transformed_cond_loader])
                sample = torch.vstack(
                    [
                        self.model.sample_batch(c[1].to(self.device)).cpu()
                        for c in transformed_cond_loader
                    ]
                )
        else:
            sample = torch.vstack(
                [self.model.sample_batch(c.to(self.device)).cpu() for c in transformed_cond_loader]
            )
            conditions = transformed_cond

        t_1 = time.time()
        sampling_time = t_1 - t_0
        LOGGER.info(f"sample_n: Finished generating {len(sample)} samples after {sampling_time} s.")
        return sample.detach().cpu(), conditions.detach().cpu()

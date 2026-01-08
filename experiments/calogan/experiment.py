# standard python libraries
import os
import time
import warnings

import h5py
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import experiments.calogan.transforms as transforms
from experiments.base_experiment import BaseExperiment
from experiments.calo_utils.us_evaluation.classifier import eval_ui_dists
from experiments.calo_utils.us_evaluation.plots import plot_ui_dists
from experiments.calogan.datasets import CaloGANDataset
from experiments.calogan.evaluate import eval_calogan_lowlevel

# Other functions of project
from experiments.logger import LOGGER


class CaloGAN(BaseExperiment):
    """
    Train a generative model on the CaloGAN datasets.
    Structure:

    init_data()          : Read in data parameters and prepare the datasets
    init_physics()       : Read in physics parameters (pass)
    _init_dataloader()   : Create the dataloaders for training and validation
    _init_loss()         : Define loss function overwritten in model class
    _init_metrics()      : Metrics to be tracjked during training (pass)
    _batch_loss()        : Calls the model's batch_loss function
    generate_Einc_ds1()  : Generate the incident energy distribution of CaloChallenge as in the training data
    sample_us()          : Sample energy ratios from the energy model
    sample_n()           : Generate n_samples from the trained model, either energy ratios or full normalized showers
    plot()               : First generate full shower, then make plots and evaluate
    save_sample()        : Save generated samples in the correct format
    load_energy_model()  : Load an external energy model if sample_us
    """

    def init_data(self):
        self.hdf5_train = self.cfg.data.training_file
        self.hdf5_test = self.cfg.data.test_file
        self.return_us = self.cfg.data.return_us
        self.transforms = []

        LOGGER.info("init_data: preparing model training")
        for name, kwargs in self.cfg.data.transforms.items():
            if "FromFile" in name:
                kwargs["model_dir"] = self.cfg.run_dir
            self.transforms.append(getattr(transforms, name)(**kwargs))
        LOGGER.info("init_data: list of preprocessing steps:")
        for _, transform in enumerate(self.transforms):
            LOGGER.info(f"{transform.__class__.__name__}")

        self.train_dataset = CaloGANDataset(
            self.hdf5_train,
            transform=self.transforms,
            return_us=self.return_us,
            device=self.device,
            dtype=self.dtype,
            rank=self.rank,
        )

        self.val_dataset = CaloGANDataset(
            self.hdf5_train,
            transform=self.transforms,
            return_us=self.return_us,
            device=self.device,
            dtype=self.dtype,
            rank=self.rank,
        )

        self.layer_boundaries = self.train_dataset.bin_edges

    def init_physics(self):
        pass

    def _init_dataloader(self):
        self.batch_size = (
            self.cfg.training.batchsize // self.world_size
            if self.world_size > 1
            else self.cfg.training.batchsize
        )

        if self.world_size > 1:
            self.train_dist_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
            self.val_dist_sampler = DistributedSampler(
                self.val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
            )
        else:
            self.train_dist_sampler = None
            self.val_dist_sampler = None

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_dist_sampler,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_dist_sampler,
            pin_memory=True,
        )

        LOGGER.info(
            f"init_dataloader: created training dataloader with {len(self.train_loader)} batches"
        )
        LOGGER.info(
            f"init_dataloader: created validation dataloader with {len(self.val_loader)} batches"
        )

    def _init_loss(self):
        pass

    def _init_metrics(self):
        pass

    def _batch_loss(self, data):
        return self.model._batch_loss(data)

    def evaluate(self):
        pass

    @torch.inference_mode()
    def sample_n(self):

        self.model.eval()

        t_0 = time.time()

        Einc = torch.rand((self.cfg.n_samples, 1)) * 99 + 1
        Einc = Einc.to(device=self.device, dtype=self.dtype)

        samples_dict = {}
        samples_dict["energy"] = Einc
        # transform Einc to basis used in training
        for fn in self.transforms:
            if hasattr(fn, "cond_transform"):
                samples_dict = fn(samples_dict)

        transformed_cond = samples_dict["energy"]
        batchsize_sample = self.cfg.training.batchsize_sample
        transformed_cond_loader = DataLoader(
            dataset=transformed_cond, batch_size=batchsize_sample, shuffle=False
        )

        # sample u_i's if self is a shape model
        if self.cfg.model_type == "shape":

            if self.cfg.sample_us:
                u_samples = self.sample_us(transformed_cond_loader)
                transformed_cond = torch.cat([transformed_cond, u_samples], dim=1)
            else:  # optionally use truth us
                transformed_cond = CaloGANDataset(
                    self.hdf5_test,
                    transform=self.transforms,
                    return_us=self.return_us,
                    device=self.device,
                ).energy.to(self.device)

            # concatenate with Einc
            transformed_cond_loader = DataLoader(
                dataset=transformed_cond, batch_size=batchsize_sample, shuffle=False
            )

        sample = torch.vstack(
            [self.model.sample_batch(c).cpu() for c in transformed_cond_loader]
        )

        t_1 = time.time()
        sampling_time = t_1 - t_0
        LOGGER.info(
            f"sample_n: Finished generating {len(sample)} samples "
            f"after {sampling_time} s."
        )
        return sample, transformed_cond.cpu()

    def sample_us(self, transformed_cond_loader):
        """Sample u_i's from the energy model"""
        # load energy model
        self.load_energy_model()

        # sample us
        t_0 = time.time()
        u_samples = torch.vstack(
            [self.energy_model.sample_batch(c) for c in transformed_cond_loader]
        )
        t_1 = time.time()
        LOGGER.info(
            f"sample_us: Finished generating {len(u_samples)} energy samples "
            f"after {t_1 - t_0} s."
        )

        u_samples_dict = {}
        u_samples_dict["extra_dims"] = u_samples
        for fn in self.energy_model_transforms[::-1]:
            if hasattr(fn, "u_transform"):
                fn.layer_keys = ["extra_dims"]
                u_samples_dict = fn(u_samples_dict, rev=True)
        for fn in self.transforms:
            if hasattr(fn, "u_transform"):
                fn.layer_keys = ["extra_dims"]
                u_samples_dict = fn(u_samples_dict)

        return u_samples_dict["extra_dims"].to(self.dtype)

    def plot(self):
        LOGGER.info("plot: generating samples")
        samples, conditions = self.sample_n()

        if self.cfg.model_type == "energy":
            reference = CaloGANDataset(
                self.hdf5_test,
                transform=self.transforms,  # TODO: Or, apply NormalizeEByLayer popped from model transforms
                return_us=self.return_us,
                device=self.device,
            )
            samples_dict = {}
            samples_dict["extra_dims"] = samples
            samples_dict["energy"] = conditions
            reference_dict = {}
            reference_dict["extra_dims"] = reference.layers
            reference_dict["energy"] = reference.energy
            # postprocess
            for fn in self.transforms[::-1]:
                if fn.__class__.__name__ == "NormalizeLayerEnergyGAN":
                    break  # this might break plotting
                fn.layer_keys = ["extra_dims"]
                samples_dict = fn(samples_dict, rev=True)
                reference_dict = fn(reference_dict, rev=True)

            samples = samples_dict["extra_dims"]
            reference = reference_dict["extra_dims"]
            # clip u_i's (except u_0) to [0,1]
            samples[:, 1:] = torch.clip(samples[:, 1:], min=0.0, max=1.0)
            reference[:, 1:] = torch.clip(reference[:, 1:], min=0.0, max=1.0)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_ui_dists(
                    samples.detach().cpu().numpy(),
                    reference.detach().cpu().numpy(),
                    cfg=self.cfg,
                )
                eval_ui_dists(
                    samples.detach().cpu().numpy(),
                    reference.detach().cpu().numpy(),
                    cfg=self.cfg,
                )
        else:
            bin_edges = self.cfg.data.bin_edges
            samples = samples.reshape(samples.shape[0], -1)

            samples_dict = {}
            samples_dict["energy"] = conditions[:, 0]
            samples_dict["extra_dims"] = conditions[:, 1:]
            samples_dict["layer_0"] = samples[:, bin_edges[0] : bin_edges[1]]
            samples_dict["layer_1"] = samples[:, bin_edges[1] : bin_edges[2]]
            samples_dict["layer_2"] = samples[:, bin_edges[2] : bin_edges[3]]
            # postprocess
            for fn in self.transforms[::-1]:
                samples_dict = fn(samples_dict, rev=True)

            samples = (
                torch.hstack(
                    (
                        samples_dict["layer_0"],
                        samples_dict["layer_1"],
                        samples_dict["layer_2"],
                    ),
                )
                .detach()
                .cpu()
                .numpy()
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                eval_calogan_lowlevel(samples, self.cfg)

    def save_sample(self, sample, energies, name=""):
        """Save sample in the correct format"""
        save_file = h5py.File(self.cfg.base_dir + f"samples{name}.hdf5", "w")
        save_file.create_dataset("incident_energies", data=energies, compression="gzip")
        save_file.create_dataset("showers", data=sample, compression="gzip")
        save_file.close()

    def load_energy_model(self):
        # initialize model
        energy_model_cfg = OmegaConf.load(self.cfg.energy_model + "config.yaml")
        # get transforms
        self.energy_model_transforms = []
        for name, kwargs in energy_model_cfg.data.transforms.items():
            if "FromFile" in name:
                kwargs["model_dir"] = energy_model_cfg.run_dir
            self.energy_model_transforms.append(getattr(transforms, name)(**kwargs))

        self.energy_model = instantiate(energy_model_cfg.model)
        num_parameters = sum(
            p.numel() for p in self.energy_model.parameters() if p.requires_grad
        )
        LOGGER.info(
            f"Instantiated energy model {type(self.energy_model.net).__name__} with {num_parameters} learnable parameters"
        )
        model_path = os.path.join(energy_model_cfg.run_dir, "models", "model_run0.pt")
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)[
                "model"
            ]
            LOGGER.info(f"Loading energy model from {model_path}")
            self.energy_model.load_state_dict(state_dict)
        except FileNotFoundError as err:
            raise ValueError(f"Cannot load model from {model_path}") from err

        self.energy_model.to(self.device, dtype=self.dtype)

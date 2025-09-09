# standard python libraries
import numpy as np
import torch
import os, time
import warnings
from torch.utils.data import DataLoader
import os
import h5py
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Other functions of project
from experiments.logger import LOGGER
from experiments.base_experiment import BaseExperiment
from experiments.calochallenge.datasets import CaloChallengeDataset
import experiments.calochallenge.transforms as transforms
from experiments.calochallenge.challenge_files import evaluate
from experiments.calochallenge.plots import plot_ui_dists


class CaloChallenge(BaseExperiment):
    """
    Train a generative model on the CaloChallenge datasets.
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
        self.particle_type = self.cfg.data.particle_type
        self.xml_filename = self.cfg.data.xml_filename
        self.train_val_frac = self.cfg.data.train_val_frac
        self.transforms = []

        LOGGER.info("init_data: preparing model training")
        for name, kwargs in self.cfg.data.transforms.items():
            if "FromFile" in name:
                kwargs["model_dir"] = self.cfg.run_dir
            self.transforms.append(getattr(transforms, name)(**kwargs))
        LOGGER.info("init_data: list of preprocessing steps:")
        for idx, transform in enumerate(self.transforms):
            LOGGER.info(f"{transform.__class__.__name__}")

        self.train_dataset = CaloChallengeDataset(
            self.hdf5_train,
            self.particle_type,
            self.xml_filename,
            train_val_frac=self.train_val_frac,
            transform=self.transforms,
            split="training",
            device=self.device,
            dtype=self.dtype,
            rank=self.rank,
        )

        self.val_dataset = CaloChallengeDataset(
            self.hdf5_train,
            self.particle_type,
            self.xml_filename,
            train_val_frac=self.train_val_frac,
            transform=self.transforms,
            split="validation",
            device=self.device,
            dtype=self.dtype,
            rank=self.rank,
        )

        self.layer_boundaries = self.train_dataset.layer_boundaries

    def init_physics(self):
        pass

    def _init_dataloader(self):
        self.batch_size = (
            self.cfg.training.batchsize // self.world_size
            if self.world_size > 1
            else self.cfg.training.batchsize
        )

        self.train_dist_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
        )
        self.val_dist_sampler = torch.utils.data.distributed.DistributedSampler(
            self.val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True
        )

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

    def generate_Einc_ds1(self, sample_multiplier=1000):
        """generate the incident energy distribution of CaloChallenge ds1
        sample_multiplier controls how many samples are generated: 10* sample_multiplier for low energies,
        and 5, 3, 2, 1 times sample multiplier for the highest energies

        """
        ret = np.logspace(8, 18, 11, base=2)
        ret = np.tile(ret, 10)
        ret = np.array(
            [
                *ret,
                *np.tile(2.0**19, 5),
                *np.tile(2.0**20, 3),
                *np.tile(2.0**21, 2),
                *np.tile(2.0**22, 1),
            ]
        )
        ret = np.tile(ret, sample_multiplier)
        np.random.shuffle(ret)
        return ret

    @torch.inference_mode()
    def sample_n(self):

        self.model.eval()

        t_0 = time.time()

        Einc = torch.tensor(
            (
                10 ** np.random.uniform(3, 6, size=self.cfg.n_samples)
                if self.cfg.eval_dataset in ["2", "3"]
                else self.generate_Einc_ds1()
            ),
            dtype=self.dtype,
            device=self.device,
        ).unsqueeze(1)

        # transform Einc to basis used in training
        dummy, transformed_cond = None, Einc
        for fn in self.transforms:
            if hasattr(fn, "cond_transform"):
                dummy, transformed_cond = fn(dummy, transformed_cond)

        batchsize_sample = self.cfg.training.batchsize_sample
        transformed_cond_loader = DataLoader(
            dataset=transformed_cond, batch_size=batchsize_sample, shuffle=False
        )

        # sample u_i's if self is a shape model
        if self.cfg.model_type == "shape":

            if self.cfg.sample_us:  # TODO
                u_samples = self.sample_us(transformed_cond_loader)
                transformed_cond = torch.cat([transformed_cond, u_samples], dim=1)
            else:  # optionally use truth us
                transformed_cond = CaloChallengeDataset(
                    self.hdf5_test,
                    self.particle_type,
                    self.xml_filename,
                    transform=self.transforms,
                    split="full",
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

        return sample.detach().cpu(), transformed_cond.detach().cpu()

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

        for fn in self.energy_model_transforms[::-1]:
            if hasattr(fn, "u_transform"):
                u_samples, _ = fn(u_samples, None, rev=True)
        for fn in self.transforms:
            if hasattr(fn, "u_transform"):
                u_samples, _ = fn(u_samples, None)

        return u_samples.to(self.dtype)

    def plot(self):
        LOGGER.info("plot: generating samples")
        samples, conditions = self.sample_n()

        if self.cfg.model_type == "energy":
            reference = CaloChallengeDataset(
                self.hdf5_test,
                self.particle_type,
                self.xml_filename,
                transform=self.transforms,  # TODO: Or, apply NormalizeEByLayer popped from model transforms
                device=self.device,
            ).layers

            # postprocess
            for fn in self.transforms[::-1]:
                if fn.__class__.__name__ == "NormalizeByElayer":
                    break  # this might break plotting
                samples, _ = fn(samples, conditions, rev=True)
                reference, _ = fn(reference, conditions, rev=True)

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
                evaluate.eval_ui_dists(
                    samples.detach().cpu().numpy(),
                    reference.detach().cpu().numpy(),
                    cfg=self.cfg,
                )
        else:
            # postprocess
            for fn in self.transforms[::-1]:
                samples, conditions = fn(samples, conditions, rev=True)

            samples = samples.numpy()
            conditions = conditions.numpy()

            self.save_sample(samples, conditions)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                evaluate.run_from_py(samples, conditions, self.cfg)

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
        model_path = os.path.join(energy_model_cfg.run_dir, "models", f"model_run0.pt")
        try:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)[
                "model"
            ]
            LOGGER.info(f"Loading energy model from {model_path}")
            self.energy_model.load_state_dict(state_dict)
        except FileNotFoundError:
            raise ValueError(f"Cannot load model from {model_path}")

        self.energy_model.to(self.device, dtype=self.dtype)

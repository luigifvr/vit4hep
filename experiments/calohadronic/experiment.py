# standard python libraries
import numpy as np
import torch
import os, time
import warnings
from torch.utils.data import DataLoader

from torch.utils.data.distributed import DistributedSampler
import os
import h5py
from hydra.utils import instantiate
from omegaconf import OmegaConf

# Other functions of project
from experiments.logger import LOGGER
from experiments.base_experiment import BaseExperiment
from experiments.calohadronic.datasets import CaloHadDataset, CaloHadCollator
import experiments.calohadronic.transforms as transforms
from experiments.calohadronic.utils import load_data
from experiments.calohadronic.evaluate import run_from_py
from experiments.calo_utils.us_evaluation.plots import plot_ui_dists
from experiments.calo_utils.us_evaluation.classifier import eval_ui_dists


class CaloHadronic(BaseExperiment):
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
        self.hdf5_dict_train = self.cfg.data.training_file_dict
        self.hdf5_dict_test = self.cfg.data.test_file_dict
        self.max_files_per_worker = self.cfg.data.max_files_per_worker
        self.return_us = self.cfg.data.return_us
        self.transforms = []

        LOGGER.info("init_data: preparing model training")
        for name, kwargs in self.cfg.data.transforms.items():
            if "FromFile" in name:
                if kwargs["model_dir"] is None:
                    kwargs["model_dir"] = self.cfg.run_dir
            self.transforms.append(getattr(transforms, name)(**kwargs))
        LOGGER.info("init_data: list of preprocessing steps:")
        for _, transform in enumerate(self.transforms):
            LOGGER.info(f"{transform.__class__.__name__}")

        # init standardization from ~20k samples
        file_0_path = next(iter(self.hdf5_dict_train.values()))[0]
        file_0 = h5py.File(file_0_path, "r")
        if self.transforms:
            dummy_data = load_data(file_0, local_index=None, dtype=self.dtype)
            for fn in self.transforms:
                dummy_data = fn(dummy_data, rank=self.rank)
        file_0.close()
        del dummy_data

        self.train_dataset = CaloHadDataset(
            self.hdf5_dict_train,
            dtype=self.dtype,
            max_files_per_worker=self.max_files_per_worker,
        )

        self.val_dataset = CaloHadDataset(
            self.hdf5_dict_test,
            dtype=self.dtype,
            max_files_per_worker=self.max_files_per_worker,
        )

    def init_physics(self):
        pass

    def _init_dataloader(self):
        # instantiate collate_fn
        collator = CaloHadCollator(
            hdf5_train_dict=self.hdf5_dict_train,
            transforms=self.transforms,
            return_us=self.return_us,
            dtype=self.dtype,
            rank=self.rank,
        )

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
            num_workers=8,
            persistent_workers=True,
            collate_fn=collator,
            prefetch_factor=4,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_dist_sampler,
            pin_memory=True,
            num_workers=8,
            persistent_workers=True,
            collate_fn=collator,
            prefetch_factor=4,
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
                # concatenate with Einc
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
                [
                    self.model.sample_batch(c.to(self.device)).cpu()
                    for c in transformed_cond_loader
                ]
            )
            conditions = transformed_cond

        t_1 = time.time()
        sampling_time = t_1 - t_0
        LOGGER.info(
            f"sample_n: Finished generating {len(sample)} samples "
            f"after {sampling_time} s."
        )

        return sample.detach().cpu(), conditions.detach().cpu()

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
        u_samples_dict["ecal"] = torch.ones(
            *self.cfg.model.list_shape[0], dtype=self.dtype, device=self.device
        )
        u_samples_dict["hcal"] = torch.ones(
            *self.cfg.model.list_shape[1], dtype=self.dtype, device=self.device
        )

        u_samples_dict["extra_dims"] = u_samples
        for fn in self.energy_model_transforms[::-1]:
            if hasattr(fn, "u_transform"):
                fn.keys = ["extra_dims"]
                u_samples_dict = fn(u_samples_dict, rev=True)
        for fn in self.transforms:
            if hasattr(fn, "u_transform"):
                u_samples_dict = fn(u_samples_dict)
        return u_samples_dict["extra_dims"].to(self.dtype)

    def plot(self):
        LOGGER.info("plot: generating samples")
        samples, conditions = self.sample_n()

        if self.cfg.model_type == "energy":
            reference = CaloHadDataset(
                self.hdf5_dict_test,
                dtype=self.dtype,
                max_files_per_worker=self.max_files_per_worker,
            )
            test_collator = CaloHadCollator(
                hdf5_train_dict=self.hdf5_dict_test,
                transforms=self.transforms,
                return_us=True,
                dtype=self.dtype,
                rank=self.rank,
            )
            reference_loader = DataLoader(
                dataset=reference,
                batch_size=self.cfg.training.batchsize_sample,
                shuffle=False,
                collate_fn=test_collator,
            )
            reference_energy_ratios = torch.vstack(
                [b[0] for b in reference_loader]
            ).cpu()
            reference_conditions = torch.vstack([b[1] for b in reference_loader]).cpu()

            samples_dict = {}
            samples_dict["extra_dims"] = samples
            samples_dict["energy"] = conditions[:, 0].unsqueeze(1)

            reference_dict = {}
            reference_dict["extra_dims"] = reference_energy_ratios
            reference_dict["energy"] = reference_conditions[:, 0].unsqueeze(1)
            # postprocess
            for fn in self.transforms[::-1]:
                if fn.__class__.__name__ == "CaloHadNormalizeByElayer":
                    break  # this might break plotting
                if hasattr(fn, "u_transform"):
                    fn.keys = ["extra_dims"]
                    samples_dict = fn(samples_dict, rev=True)
                    reference_dict = fn(reference_dict, rev=True)

            # clip u_i's (except u_0) to [0,1]
            samples = samples_dict["extra_dims"]
            reference = reference_dict["extra_dims"]
            samples[:, 1:] = torch.clip(samples[:, 1:], min=0.0, max=1.0)
            reference[:, 1:] = torch.clip(reference[:, 1:], min=0.0, max=1.0)

            samples, reference = (
                samples.detach().cpu().numpy(),
                reference.detach().cpu().numpy(),
            )
            self.save_sample(samples_dict, name=f"_{self.cfg.run_idx}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                plot_ui_dists(
                    samples,
                    reference,
                    cfg=self.cfg,
                )
                eval_ui_dists(
                    samples,
                    reference,
                    cfg=self.cfg,
                )
        else:
            samples = samples.squeeze(1)

            samples_dict = {}
            samples_dict["ecal"] = samples[:, : (10 * 15 * 15)].reshape(-1, 10, 15, 15)
            samples_dict["hcal"] = samples[:, -(48 * 30 * 30) :].reshape(-1, 48, 30, 30)

            n_layers = samples_dict["ecal"].shape[1] + samples_dict["hcal"].shape[1]
            samples_dict["extra_dims"] = conditions[:, :n_layers]
            samples_dict["energy"] = conditions[:, n_layers : (n_layers + 1)].unsqueeze(
                1
            )
            for key in samples_dict.keys():
                samples_dict[key] = samples_dict[key].clone()
            # postprocess
            for fn in self.transforms[::-1]:
                samples_dict = fn(samples_dict, rev=True)

            ecal = samples_dict["ecal"].numpy()
            hcal = samples_dict["hcal"].numpy()
            Einc = samples_dict["energy"].numpy()

            self.save_sample(samples_dict, name=f"_{self.cfg.run_idx}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run_from_py(ecal, hcal, Einc, self.cfg)

    def eval_sample(self, dirname=""):
        ecal, hcal, energies = self.load_sample(dirname=dirname)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_from_py(ecal, hcal, energies, self.cfg)

    def save_sample(self, samples_dict, name=""):
        """Save sample in the correct format"""
        save_file = h5py.File(self.cfg.run_dir + f"/samples{name}.hdf5", "w")
        for key in samples_dict.keys():
            save_file.create_dataset(key, data=samples_dict[key], compression="gzip")
        save_file.close()

    def load_sample(self, dirname=""):
        """Load sample from the correct format"""
        if dirname == "":
            dirname = self.cfg.run_dir + f"/samples_{self.cfg.run_idx}.hdf5"
        LOGGER.info(f"load_sample: loading samples from {dirname}")
        load_file = h5py.File(dirname, "r")
        energies = load_file["energy"][:]
        ecal = load_file["ecal"][:]
        hcal = load_file["hcal"][:]
        load_file.close()
        return ecal, hcal, energies

    def load_energy_model(self):
        # initialize model
        energy_model_cfg = OmegaConf.load(self.cfg.energy_model + "config.yaml")
        # get transforms
        self.energy_model_transforms = []
        for name, kwargs in energy_model_cfg.data.transforms.items():
            if "FromFile" in name:
                kwargs["model_dir"] = energy_model_cfg.run_dir
            self.energy_model_transforms.append(getattr(transforms, name)(**kwargs))
        # init standardization
        file_0_path = next(iter(self.hdf5_dict_train.values()))[0]
        file_0 = h5py.File(file_0_path, "r")
        if self.energy_model_transforms:
            dummy_data = load_data(file_0, local_index=None, dtype=self.dtype)
            for fn in self.energy_model_transforms:
                dummy_data = fn(dummy_data, rank=self.rank)
        file_0.close()
        del dummy_data

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

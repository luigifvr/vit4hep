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
from experiments.lemurs.datasets import LEMURSDataset, LEMURSCollator
import experiments.lemurs.transforms as transforms
from experiments.calochallenge.challenge_files import evaluate
from experiments.calochallenge.plots import plot_ui_dists


class LEMURS(BaseExperiment):
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
        self.num_classes = self.cfg.data.num_classes
        self.transforms = []

        LOGGER.info("init_data: preparing model training")
        for name, kwargs in self.cfg.data.transforms.items():
            if "FromFile" in name:
                kwargs["model_dir"] = self.cfg.run_dir
            self.transforms.append(getattr(transforms, name)(**kwargs))
        LOGGER.info("init_data: list of preprocessing steps:")
        for _, transform in enumerate(self.transforms):
            LOGGER.info(f"{transform.__class__.__name__}")

        self.train_dataset = LEMURSDataset(
            self.hdf5_dict_train,
            dtype=self.dtype,
            max_files_per_worker=self.cfg.data.max_files_per_worker,
        )

        self.val_dataset = LEMURSDataset(
            self.hdf5_dict_test,
            dtype=self.dtype,
            max_files_per_worker=self.cfg.data.max_files_per_worker,
        )

    def init_physics(self):
        pass

    def _init_dataloader(self):
        # instantiate collate_fn
        collator = LEMURSCollator(
            hdf5_train_dict=self.hdf5_dict_train,
            transforms=self.transforms,
            num_classes=self.num_classes,
            dtype=self.dtype,
            rank=self.rank,
        )

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
            num_workers=4,
            persistent_workers=True,
            collate_fn=collator,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            sampler=self.val_dist_sampler,
            pin_memory=True,
            num_workers=4,
            persistent_workers=True,
            collate_fn=collator,
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

    def sample_initial_conds(self):
        Einc = torch.tensor(
            (np.random.uniform(10**3, 10**6, size=self.cfg.n_samples)),
            dtype=self.dtype,
            device=self.device,
        ).unsqueeze(1)

        phi = torch.tensor(
            (np.random.uniform(0, 2 * np.pi, size=self.cfg.n_samples)),
            dtype=self.dtype,
            device=self.device,
        ).unsqueeze(1)

        cos_theta = torch.tensor(
            (
                np.random.uniform(
                    torch.cos(torch.tensor(0.87)),
                    torch.cos(torch.tensor(2.27)),
                    size=self.cfg.n_samples,
                )
            ),
            dtype=self.dtype,
            device=self.device,
        ).unsqueeze(1)
        theta = torch.acos(cos_theta)
        return Einc, phi, theta

    @torch.inference_mode()
    def sample_n(self):

        self.model.eval()

        t_0 = time.time()

        Einc, phi, theta = self.sample_initial_conds()
        gen_label_vector = (
            self.cfg.data.gen_label_vector
        )  # one-hot encoded vector, e.g. [0,1,0,0]
        labels = torch.tensor(
            np.tile(gen_label_vector, (self.cfg.n_samples, 1)),
            dtype=self.dtype,
            device=self.device,
        )  # shape (n_samples, num_classes)

        samples = {
            "incident_energy": Einc,
            "incident_phi": phi,
            "incident_theta": theta,
        }
        samples["showers"] = torch.empty(
            *self.cfg.model.shape, dtype=self.dtype, device=self.device
        )
        samples["extra_dims"] = torch.empty(
            self.cfg.model.shape[0], dtype=self.dtype, device=self.device
        )
        samples["label"] = labels
        for fn in self.transforms:
            if hasattr(fn, "cond_transform"):
                samples = fn(samples)

        transformed_cond = torch.cat(
            (
                samples["incident_energy"],
                samples["incident_theta"],
                samples["incident_phi"],
                samples["label"],
            ),
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
                transformed_cond = torch.cat([transformed_cond, u_samples], dim=1)
            else:  # optionally use truth us
                transformed_cond = LEMURSDataset(
                    self.hdf5_dict_test,
                    dtype=self.dtype,
                    max_files_per_worker=self.cfg.data.max_files_per_worker,
                )
                test_collator = LEMURSCollator(
                    hdf5_train_dict=self.hdf5_dict_test,
                    transforms=self.transforms,
                    num_classes=self.num_classes,
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

        sample = torch.vstack(
            [
                self.model.sample_batch(c[1].to(self.device)).cpu()
                for c in transformed_cond_loader
            ]
        )
        conditions = torch.vstack([c[1] for c in transformed_cond_loader])

        t_1 = time.time()
        sampling_time = t_1 - t_0
        LOGGER.info(
            f"sample_n: Finished generating {len(sample)} samples "
            f"after {sampling_time} s."
        )

        return sample.cpu(), conditions.cpu()

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
            reference = LEMURSDataset(
                self.hdf5_test,
                dtype=self.dtype,
                max_files_per_worker=self.cfg.data.max_files_per_worker,
            )
            test_collator = LEMURSCollator(
                hdf5_train_dict=self.hdf5_dict_test,
                transforms=self.transforms,
                num_classes=self.num_classes,
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

            samples_dict = {}
            samples_dict["extra_dims"] = samples
            reference_dict = {}
            reference_dict["extra_dims"] = reference_energy_ratios
            # postprocess
            for fn in self.transforms[::-1]:
                if fn.__class__.__name__ == "LEMURSNormalizeByElayer":
                    break  # this might break plotting
                samples_dict = fn(samples_dict, rev=True)
                reference_dict = fn(reference_dict, rev=True)

            # clip u_i's (except u_0) to [0,1]
            samples = samples_dict["extra_dims"]
            reference = reference_dict["extra_dims"]
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
            # reshape as in LEMURS
            samples = samples.squeeze(1)
            samples = samples.permute(0, 3, 2, 1)

            samples_dict = {}
            samples_dict["showers"] = samples
            samples_dict["extra_dims"] = conditions[:, : samples.shape[-1]]
            samples_dict["incident_energy"] = conditions[
                :, samples.shape[-1]
            ].unsqueeze(1)
            samples_dict["incident_theta"] = conditions[
                :, samples.shape[-1] + 1
            ].unsqueeze(1)
            samples_dict["incident_phi"] = conditions[
                :, samples.shape[-1] + 2
            ].unsqueeze(1)
            samples_dict["label"] = conditions[:, samples.shape[-1] + 3 :]

            # postprocess
            for fn in self.transforms[::-1]:
                samples_dict = fn(samples_dict, rev=True)

            samples = samples_dict["showers"].numpy()
            Einc = samples_dict["incident_energy"].numpy()
            theta = samples_dict["incident_theta"].numpy()
            phi = samples_dict["incident_phi"].numpy()

            self.save_sample(samples, conditions)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                evaluate.run_from_py(samples, Einc, theta, phi, self.cfg)

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

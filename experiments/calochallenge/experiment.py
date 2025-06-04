# standard python libraries
import numpy as np
import torch
import os, time
import warnings
from torch.utils.data import DataLoader
import os
import h5py
from hydra.utils import instantiate

# Other functions of project
from experiments.logger import LOGGER
from experiments.base_experiment import BaseExperiment
from experiments.calochallenge.datasets import CaloChallengeDataset
import experiments.calochallenge.transforms as transforms
from challenge_files import evaluate
from experiments.calochallenge.plots import plot_ui_dists


class CaloChallenge(BaseExperiment):
    """
    Base Class for Generative Models to inherit from.
    Children classes should overwrite the individual methods as needed.
    Every child class MUST overwrite the methods:

    def build_net(self): should register some NN architecture as self.net
    def batch_loss(self, x): takes a batch of samples as input and returns the loss
    def sample_n_parallel(self, n_samples): generates and returns n_samples new samples

    See tbd.py for an example of child class

    Structure:

    __init__(params)      : Read in parameters and register the important ones
    build_net()           : Create the NN and register it as self.net
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    prepare_training()    : Read in the appropriate parameters and prepare the model for training
                            Currently this is called from run_training(), so it should not be called on its own
    run_training()        : Run the actual training.
                            Necessary parameters are read in and the training is performed.
                            This calls on the methods train_one_epoch() and validate_one_epoch()
    train_one_epoch()     : Performs one epoch of model training.
                            This calls on the method batch_loss(x)
    validate_one_epoch()  : Performs one epoch of validation.
                            This calls on the method batch_loss(x)
    batch_loss(x)         : Takes one batch of samples as input and returns the loss.
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    sample_n(n_samples)   : Generates and returns n_samples new samples as a numpy array
                            HAS TO BE OVERWRITTEN IN CHILD CLASS
    sample_and_plot       : Generates n_samples and makes plots with them.
                            This is meant to be used during training if intermediate plots are wanted

    """

    def init_data(self):
        self.hdf5_train = self.cfg.data.training_file
        self.hdf5_test = self.cfg.data.test_file
        self.particle_type = self.cfg.data.particle_type
        self.xml_filename = self.cfg.data.xml_filename
        self.train_val_frac = self.cfg.data.train_val_frac
        self.transforms = []

        for name, kwargs in self.cfg.data.transforms.items():
            if name == "StandardizeFromFile":
                kwargs["model_dir"] = self.cfg.run_dir
            self.transforms.append(getattr(transforms, name)(**kwargs))

        LOGGER.info("init_data: preparing model training")
        LOGGER.info("init_data: list of preprocessing steps ")
        LOGGER.info(self.transforms)

        self.train_dataset = CaloChallengeDataset(
            self.hdf5_train,
            self.particle_type,
            self.xml_filename,
            train_val_frac=self.train_val_frac,
            transform=self.transforms,
            split="training",
            device=self.device,
            dtype=self.dtype,
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
        )

        self.layer_boundaries = self.train_dataset.layer_boundaries

    def init_physics(self):
        pass

    def _init_dataloader(self):
        self.batch_size = self.cfg.training.batchsize
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=True
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

    # @torch.inference_mode()
    # def latent_samples(self, epoch=None):
    #     """
    #         Plot latent space distribution.

    #         Parameters:
    #         epoch (int): current epoch
    #     """
    #     self.eval()
    #     with torch.no_grad():
    #         samples = torch.zeros(self.val_loader.dataset.layers.shape) #TODO ugly
    #         stop = 0
    #         for x, c in self.val_loader:
    #             start = stop
    #             stop += len(x)
    #             samples[start:stop] = self.forward(x,c)[0].cpu()
    #         samples = samples.reshape(-1, math.prod(self.shape)).numpy()
    #     plot_latent(samples, self.doc.basedir, epoch)

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
                # load energy model
                self.load_energy_model()

                # sample us
                u_samples = torch.vstack(
                    [self.energy_model.sample_batch(c) for c in transformed_cond_loader]
                )

                transformed_cond = torch.cat([transformed_cond, u_samples], dim=1)
            else:  # optionally use truth us
                transformed_cond = CaloChallengeDataset(
                    self.hdf5_test,
                    self.particle_type,
                    self.xml_filename,
                    transform=self.transforms,
                    device=self.device,
                ).energy

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

            samples = samples.detach().cpu().numpy()
            conditions = conditions.detach().cpu().numpy()

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
        self.energy_model = instantiate(self.cfg.energy_model)
        num_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        LOGGER.info(
            f"Instantiated energy model {type(self.energy_model.net).__name__} with {num_parameters} learnable parameters"
        )
        model_path = os.path.join(
            self.cfg.energy_model.run_dir, "models", f"model_run0.pt"
        )
        try:
            state_dict = torch.load(model_path, map_location="cpu")["model"]
            LOGGER.info(f"Loading energy model from {model_path}")
            self.energy_model.load_state_dict(state_dict)
        except FileNotFoundError:
            raise ValueError(f"Cannot load model from {model_path}")

        self.energy_model.to(self.device, dtype=self.dtype)

    # def load_other(self, model_dir):
    #     """ Load a different model (e.g. to sample u_i's)"""

    #     with open(os.path.join(model_dir, 'params.yaml')) as f:
    #         params = yaml.load(f, Loader=yaml.FullLoader)

    #     model_class = params['model']
    #     # choose model
    #     if model_class == 'TBD':
    #         Model = self.__class__
    #     if model_class == 'TransfusionAR':
    #         from Models import TransfusionAR
    #         Model = TransfusionAR
    #     elif model_class == 'AE':
    #         from Models import AE
    #         Model = AE

    #     # load model
    #     doc = Documenter(None, existing_run=model_dir, read_only=True)
    #     other = Model(params, self.device, doc)
    #     state_dicts = torch.load(
    #         os.path.join(model_dir, 'model.pt'), map_location=self.device
    #     )
    #     other.net.load_state_dict(state_dicts["net"])

    #     # use eval mode and freeze weights
    #     other.eval()
    #     for p in other.parameters():
    #         p.requires_grad = False

    #     return other

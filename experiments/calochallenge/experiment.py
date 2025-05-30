# standard python libraries
import numpy as np
import torch
import torch.nn as nn
import os, time
import warnings
from torch.utils.data import DataLoader
import os
import sys
import h5py

# Other functions of project
from experiments.logger import LOGGER
from experiments.base_experiment import BaseExperiment
from experiments.calochallenge.datasets import CaloChallengeDataset
import experiments.calochallenge.transforms as transforms
from challenge_files import evaluate

# from Util.util import *
# from datasets import *
# from documenter import Documenter
# from plotting_util import *
# from transforms import *
# from challenge_files import *
# from challenge_files import evaluate # avoid NameError: 'evaluate' is not defined
# import models
# from models import *


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

    # def __init__(self, cfg):
    #     """
    #     :param params: file with all relevant model parameters
    #     """
    #     super(BaseExperiment).__init__(cfg)
    # self.doc = doc
    # self.params = params
    # self.device = device
    # self.shape = self.params['shape']#get(self.params,'shape')
    # self.conditional = get(self.params,'conditional',False)
    # self.single_energy = get(self.params, 'single_energy', None) # Train on a single energy
    # self.eval_mode = get(self.params, 'eval_mode', 'all')

    # self.batch_size = self.params["batch_size"]
    # self.batch_size_sample = get(self.params, "batch_size_sample", 10_000)

    # self.net = self.build_net()
    # print(self.net)
    # param_count = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
    # print(f'init model: Model has {param_count} parameters')
    # self.params['parameter_count'] = param_count

    # self.epoch = get(self.params, "total_epochs", 0)
    # self.iterations = get(self.params,"iterations", 1)
    # self.regular_loss = []
    # self.kl_loss = []

    # self.runs = get(self.params, "runs", 0)
    # self.iterate_periodically = get(self.params, "iterate_periodically", False)
    # self.validate_every = get(self.params, "validate_every", 50)

    # # augment data
    # self.aug_transforms = get(self.params, "augment_batch", False)

    # # load autoencoder for latent modelling
    # #self.ae_dir = get(self.params, "autoencoder", None)
    # #if self.ae_dir is None:
    # self.transforms = get_transformations(
    #     params.get('transforms', None), doc=self.doc
    # )
    # self.latent = False
    # else:
    #    self.ae = self.load_other(self.ae_dir)# model_class='AE'
    #    self.transforms = self.ae.transforms
    #    self.latent = True
    def init_data(self):
        self.hdf5_train = self.cfg.data.training_file
        self.hdf5_test = self.cfg.data.test_file
        self.particle_type = self.cfg.data.particle_type
        self.xml_filename = self.cfg.data.xml_filename
        self.val_frac = self.cfg.data.val_frac
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
            val_frac=self.val_frac,
            transform=self.transforms,
            split="training",
            device=self.device,
            dtype=self.dtype,
        )

        self.val_dataset = CaloChallengeDataset(
            self.hdf5_train,
            self.particle_type,
            self.xml_filename,
            val_frac=self.val_frac,
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

    # def run_training(self):

    #     self.prepare_training()
    #     samples = []
    #     n_epochs = get(self.params, "n_epochs", 100)
    #     past_epochs = get(self.params, "total_epochs", 0)
    #     if past_epochs != 0:
    #         self.load(epoch=past_epochs)
    #         self.scheduler = set_scheduler(self.optimizer, self.params, self.n_trainbatches, last_epoch=self.params.get("total_epochs", -1)*self.n_trainbatches)
    #     print(f"train_model: Model has been trained for {past_epochs} epochs before.")
    #     print(f"train_model: Beginning training. n_epochs set to {n_epochs}", flush=True)

    #     self.latent_samples(epoch=0)
    #     t_0 = time.time()
    #     for e in range(n_epochs):
    #         t0 = time.time()

    #         self.epoch = past_epochs + e
    #         self.net.train()
    #         self.train_one_epoch()

    #         if (self.epoch + 1) % self.validate_every == 0:
    #             self.eval()
    #             self.validate_one_epoch()

    #         if self.sample_periodically:
    #             if (self.epoch + 1) % self.sample_every == 0:
    #                 self.eval()

    #                 # # if true then i * bayesian samples will be drawn, else just 1
    #                 # iterations = self.iterations if self.iterate_periodically else 1
    #                 # bay_samples = []
    #                 # for i in range(0, iterations):
    #                 #     sample, c = self.sample_n()
    #                 #     bay_samples.append(sample)
    #                 # samples = np.concatenate(bay_samples)
    #                 if get(self.params, "reconstruct", False):
    #                     samples, c = self.reconstruct_n()
    #                 else:
    #                     samples, c = self.sample_n()
    #                 self.plot_samples(samples=samples, conditions=c, name=self.epoch, energy=self.single_energy, mode=self.eval_mode)

    #         # save model periodically, useful when trying to understand how weights are learned over iterations
    #         if get(self.params,"save_periodically",False):
    #             if (self.epoch + 1) % get(self.params,"save_every",10) == 0 or self.epoch==0:
    #                 self.save(epoch=f"{self.epoch}")

    #         # estimate training time
    #         if e==0:
    #             t1 = time.time()
    #             dtEst= (t1-t0) * n_epochs
    #             print(f"Training time estimate: {dtEst/60:.2f} min = {dtEst/60**2:.2f} h", flush=True)
    #         sys.stdout.flush()
    #     t_1 = time.time()
    #     traintime = t_1 - t_0
    #     self.params['train_time'] = traintime
    #     print(
    #         f"train_model: Finished training {n_epochs} epochs after {traintime:.2f} s = {traintime / 60:.2f} min = {traintime / 60 ** 2:.2f} h.", flush=True)

    #     #save final model
    #     print("train_model: Saving final model: ", flush=True)
    #     self.save()
    #     # generate and plot samples at the end
    #     if get(self.params, "sample", True):
    #         print("generate_samples: Start generating samples", flush=True)
    #         if get(self.params, "reconstruct", False):
    #             samples, c = self.reconstruct_n()
    #         else:
    #             samples, c = self.sample_n()
    #         self.plot_samples(samples=samples, conditions=c, energy=self.single_energy)

    # def train_one_epoch(self):
    #     # create list to save train_loss
    #     train_losses = np.array([])
    #     grad_norms = np.array([])

    #     # iterate batch wise over input
    #     for batch_id, x in enumerate(self.train_loader):

    #         self.optimizer.zero_grad(set_to_none=True)

    #         # calculate batch loss
    #         loss = self.batch_loss(x)
    #         if np.isfinite(loss.item()): # and (abs(loss.item() - loss_m) / loss_s < 5 or len(self.train_losses_epoch) == 0):
    #             loss.backward()
    #             clip = self.params.get('clip_gradients_to', None)
    #             if clip:
    #                 nn.utils.clip_grad_norm_(self.net.parameters(), clip)

    #             grad_norm = (
    #                     torch.nn.utils.clip_grad_norm(
    #                         self.model.parameters(), float("inf")
    #                         ).cpu().item()
    #                     )

    #             self.optimizer.step()
    #             train_losses = np.append(train_losses, loss.item())
    #             grad_norms = np.append(grad_norms, grad_norm)
    #             # if self.log:
    #             #     self.logger.add_scalar("train_losses", train_losses[-1], self.epoch*self.n_trainbatches + batch_id)

    #             if self.use_scheduler:
    #                 self.scheduler.step()
    #                 # if self.log:
    #                 #     self.logger.add_scalar("learning_rate", self.scheduler.get_last_lr()[0],
    #                 #                            self.epoch * self.n_trainbatches + batch_id)

    #         else:
    #             print(f"train_model: Unstable loss. Skipped backprop for epoch {self.epoch}, batch_id {batch_id}")

    #     self.train_losses_epoch = np.append(self.train_losses_epoch, train_losses.mean())
    #     self.grad_norms_epoch = np.append(self.grad_norms_epoch, grad_norms.mean())
    #     self.train_losses = np.concatenate([self.train_losses, train_losses], axis=0)
    #     if self.log:
    #         self.logger.add_scalar("train_losses_epoch", self.train_losses_epoch[-1], self.epoch)
    #         self.logger.add_scalar("grad_norm", self.grad_norms_epoch[-1], self.epoch)
    #         if self.use_scheduler:
    #             self.logger.add_scalar("learning_rate_epoch", self.scheduler.get_last_lr()[0],
    #                                    self.epoch)

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

    # @torch.inference_mode()
    # def validate_one_epoch(self):

    #     val_losses = np.array([])
    #     # iterate batch wise over input
    #     for batch_id, x in enumerate(self.val_loader):

    #         # calculate batch loss
    #         loss = self.batch_loss(x)
    #         val_losses = np.append(val_losses, loss.item())
    #         # if self.log:
    #         #     self.logger.add_scalar("val_losses", val_losses[-1], self.epoch*self.n_trainbatches + batch_id)

    #     self.val_losses_epoch = np.append(self.val_losses_epoch, val_losses.mean())
    #     self.val_losses = np.concatenate([self.val_losses, val_losses], axis=0)
    #     if self.log:
    #         self.logger.add_scalar("val_losses_epoch", self.val_losses_epoch[-1], self.epoch)
    #     self.latent_samples(epoch=len(self.val_losses_epoch))

    # def batch_loss(self, x):
    #     pass

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
            # load energy model
            # energy_model = self.load_other(self.params['energy_model'])

            if self.cfg.sample_us:  # TODO
                # sample us
                u_samples = torch.vstack(
                    [energy_model.sample_batch(c) for c in transformed_cond_loader]
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

    # def sample_batch(self, batch):
    #     pass

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

            plot_ui_dists(
                samples.detach().cpu().numpy(),
                reference.detach().cpu().numpy(),
                documenter=doc,
            )
            evaluate.eval_ui_dists(
                samples.detach().cpu().numpy(),
                reference.detach().cpu().numpy(),
                documenter=doc,
                params=self.params,
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

    # def plot_saved_samples(self, name="", energy=None, doc=None):
    #     if doc is None: doc = self.doc
    #     mode = self.params.get('eval_mode', 'all')
    #     script_args = (
    #         f"-i {doc.basedir}/ "
    #         f"-r {self.params['eval_hdf5_file']} -m {mode} --cut {self.params['eval_cut']} "
    #         f"-d {self.params['eval_dataset']} --output_dir {doc.basedir}/final/ --save_mem"
    #     ) + (f" --energy {energy}" if energy is not None else '')
    #     evaluate.main(script_args.split())

    def save_sample(self, sample, energies, name=""):
        """Save sample in the correct format"""
        save_file = h5py.File(self.cfg.base_dir + f"samples{name}.hdf5", "w")
        save_file.create_dataset("incident_energies", data=energies)
        save_file.create_dataset("showers", data=sample)
        save_file.close()

    # def save(self, epoch=""):
    #     """ Save the model, and more if needed"""
    #     torch.save({"opt": self.optimizer.state_dict(),
    #                 "net": self.net.state_dict(),
    #                 "losses": self.train_losses_epoch,
    #                 "epoch": self.epoch,
    #                 "scheduler": self.scheduler.state_dict()}
    #                 , self.doc.get_file(f"model{epoch}.pt"))

    # def load(self, epoch=""):
    #     """ Load the model, and more if needed"""
    #     name = self.doc.get_file(f"model{epoch}.pt")
    #     state_dicts = torch.load(name, map_location=self.device)
    #     self.net.load_state_dict(state_dicts["net"])

    #     if "losses" in state_dicts:
    #         self.train_losses_epoch = state_dicts.get("losses", {})
    #     if "epoch" in state_dicts:
    #         self.epoch = state_dicts.get("epoch", 0)
    #     #if "opt" in state_dicts:
    #     #    self.optimizer.load_state_dict(state_dicts["opt"])
    #     #if "scheduler" in state_dicts:
    #     #    self.scheduler.load_state_dict(state_dicts["scheduler"])
    #     self.net.to(self.device)

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

import os
import torch
import numpy as np
import gc

from torch.utils.data import Dataset
from experiments.logger import LOGGER
from experiments.calochallenge.transforms import *
from experiments.calochallenge.utils import load_data, get_energy_and_sorted_layers


class CaloChallengeDataset(Dataset):
    """Dataset for CaloChallenge showers"""

    def __init__(
        self,
        hdf5_file,
        particle_type,
        xml_filename,
        val_frac=0.3,
        transform=None,
        split="full",
        device="cpu",
        dtype=torch.float32,
    ):
        """
        Arguments:
            hdf5_file: path to hdf5 file
            particle_type: photon, pion or electron
            xml_filename: path to XML filename
            transform: list of transformations
        """

        self.voxels, self.layer_boundaries = load_data(
            hdf5_file, particle_type, xml_filename
        )
        self.energy, self.layers = get_energy_and_sorted_layers(self.voxels)
        del self.voxels

        self.transform = transform
        self.device = device
        self.dtype = dtype

        self.energy = torch.tensor(self.energy, dtype=self.dtype)
        self.layers = torch.tensor(self.layers, dtype=self.dtype)

        # apply preprocessing and then move to GPU
        if self.transform:
            for fn in self.transform:
                self.layers, self.energy = fn(self.layers, self.energy)

        val_size = int(len(self.energy) * val_frac)
        trn_size = len(self.energy) - val_size
        # make train/val split
        if split == "training":
            self.layers = self.layers[:trn_size]
            self.energy = self.energy[:trn_size]
        elif split == "validation":
            self.layers = self.layers[-val_size:]
            self.energy = self.energy[-val_size:]

        self.layers = self.layers.to(device)
        self.energy = self.energy.to(device)

        self.min_bounds = self.layers.min()
        self.max_bounds = self.layers.max()

        LOGGER.info(f"datasets: loaded {split} data with shape {*self.layers.shape,}")
        LOGGER.info(
            f"datasets: boundaries of dataset are ({self.min_bounds}, {self.max_bounds})"
        )

    def __len__(self):
        return len(self.energy)

    def __getitem__(self, idx):
        return self.layers[idx], self.energy[idx]

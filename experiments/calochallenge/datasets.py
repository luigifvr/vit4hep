import torch

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
        train_val_frac=[0.7, 0.3],
        transform=None,
        split="full",
        device="cpu",
        dtype=torch.float32,
        rank=0,
    ):
        """
        Arguments:
            hdf5_file: path to hdf5 file
            particle_type: photon, pion or electron
            xml_filename: path to XML filename
            transform: list of transformations
        """
        assert split == "full" or train_val_frac[0] + train_val_frac[1] <= 1.0

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

        # apply preprocessing
        if self.transform:
            for fn in self.transform:
                self.layers, self.energy = fn(self.layers, self.energy, rank=rank)

        val_size = int(len(self.energy) * train_val_frac[1])
        trn_size = int(len(self.energy) * train_val_frac[0])
        # make train/val split
        if split == "training":
            self.layers = self.layers[:trn_size]
            self.energy = self.energy[:trn_size]
        elif split == "validation":
            self.layers = self.layers[-val_size:]
            self.energy = self.energy[-val_size:]
        elif split == "full":
            self.layers = self.layers[...]
            self.energy = self.energy[...]

        self.layers = self.layers.to(dtype)
        self.energy = self.energy.to(dtype)

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

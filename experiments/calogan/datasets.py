import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.calogan.utils import load_data
from experiments.logger import LOGGER


class CaloGANDataset(Dataset):
    """Dataset for CaloGAN (arXiv:1712.10321) showers"""

    def __init__(
        self,
        hdf5_file,
        transform=None,
        return_us=False,
        dtype=torch.float32,
        rank=0,
    ):
        """
        Arguments:
            hdf5_file: path to hdf5 file
            transform: list of transformations applied as preprocessing
            return_us: whether to return the extra layer energy conditions (used for the energy network)
            dtype: data type for the voxels and conditions
            rank: rank of the process
        """

        self.data_dict = load_data(hdf5_file)
        self.bin_edges = np.array([0, 288, 432, 504])

        for key in self.data_dict.keys():
            self.data_dict[key] = torch.tensor(self.data_dict[key]).flatten(start_dim=1)

        self.return_us = return_us
        self.transform = transform
        self.dtype = dtype

        # apply preprocessing
        if self.transform:
            for fn in self.transform:
                if fn.__class__.__name__ == "NormalizeLayerEnergyGAN":
                    fn.bin_edges = self.bin_edges
                self.data_dict = fn(self.data_dict, rank=rank)

        if self.return_us:
            self.layers = self.data_dict["extra_dims"]
            self.energy = self.data_dict["energy"]
        else:
            # store data as 2d array with flattened layers: (dataset size, # of voxels)
            # geometric information is recovered from the bin edges
            # and the model config file
            self.layers = torch.hstack(
                (
                    self.data_dict["layer_0"],
                    self.data_dict["layer_1"],
                    self.data_dict["layer_2"],
                ),
            )
            self.layers = self.layers.unsqueeze(1)
            self.energy = torch.hstack(
                (self.data_dict["energy"], self.data_dict["extra_dims"]),
            )

        self.layers = self.layers.to(dtype)
        self.energy = self.energy.to(dtype)

        self.min_bounds = self.layers.min()
        self.max_bounds = self.layers.max()

        LOGGER.info(f"datasets: loaded data with shape {(*self.layers.shape,)}")
        LOGGER.info(f"datasets: boundaries of dataset are ({self.min_bounds}, {self.max_bounds})")

    def __len__(self):
        return len(self.energy)

    def __getitem__(self, idx):
        return self.layers[idx], self.energy[idx]

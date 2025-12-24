import h5py
import torch
from collections import OrderedDict

from torch.utils.data import Dataset
from experiments.logger import LOGGER
from experiments.calohadronic.transforms import *
from experiments.calohadronic.utils import load_data


class CaloHadDataset(Dataset):
    """
    Dataset that loads data from multiple HDF5 files with LRU caching of file handles per worker.

    _build_index_map: Create a global map from a list of hdf5 input files
    _init_worker: Initialize a worker
    _get_file_handle: For each worker, manage the number of files open at the same time
    """

    def __init__(self, hdf5_files_dict, max_files_per_worker=4, dtype="float32"):
        self.max_open_files = max_files_per_worker
        self.open_files_cache = None
        self.worker_id = None
        self.dtype = dtype

        # create the index map in the main process
        self.index_map = self._build_index_map(hdf5_files_dict)
        self.dataset_size = len(self.index_map)
        LOGGER.info(f"Dataset indexed with {self.dataset_size} samples.")

    def _build_index_map(self, hdf5_files_dict):
        """Create a map from a global index to a (file_path, local_index) tuple."""
        index_map = []
        for label, file_list in hdf5_files_dict.items():
            for file_path in file_list:
                try:
                    with h5py.File(file_path, "r") as f:
                        num_samples = len(f["events"])
                        for local_idx in range(num_samples):
                            index_map.append((file_path, local_idx))
                except (IOError, KeyError) as e:
                    LOGGER.error(f"Could not read {file_path} for class {label}: {e}")
        return index_map

    def _init_worker(self):
        """Initializes state within a worker process."""
        self.open_files_cache = OrderedDict()
        worker_info = torch.utils.data.get_worker_info()
        self.worker_id = worker_info.id if worker_info is not None else -1

    def _get_file_handle(self, file_path):
        """Manages the LRU cache of file handles for a single worker."""
        if file_path in self.open_files_cache:
            self.open_files_cache.move_to_end(file_path)
            return self.open_files_cache[file_path]

        if len(self.open_files_cache) >= self.max_open_files:
            _, old_file = self.open_files_cache.popitem(last=False)
            old_file.close()

        file_handle = h5py.File(file_path, "r")
        self.open_files_cache[file_path] = file_handle
        return file_handle

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.worker_id is None:
            self._init_worker()

        file_path, local_idx = self.index_map[idx]
        file_handle = self._get_file_handle(file_path)

        raw_data = load_data(file_handle, local_idx, dtype=self.dtype)
        return raw_data


class CaloHadCollator:
    """
    Collator that applies transformations at the batch level.
    """

    def __init__(
        self,
        hdf5_train_dict,
        transforms,
        return_us=False,
        rank=0,
        dtype="float32",
    ):
        self.hdf5_train_dict = hdf5_train_dict
        self.transforms = transforms
        self.return_us = return_us
        self.dtype = dtype
        self.rank = rank
        self.worker_id = None  # set on first call

        # init standardization
        file_0_path = next(iter(hdf5_train_dict.values()))[0]
        file_0 = h5py.File(file_0_path, "r")
        if self.transforms:
            dummy_data = load_data(file_0, local_index=None, dtype=self.dtype)
            for fn in self.transforms:
                dummy_data = fn(dummy_data, rank=self.rank)
        file_0.close()
        del dummy_data

    def __call__(self, raw_batch):
        if self.worker_id is None:
            worker_info = torch.utils.data.get_worker_info()
            self.worker_id = worker_info.id if worker_info is not None else -1

        batch_dict = {
            key: torch.cat([item[key] for item in raw_batch], dim=0)
            for key in raw_batch[0]
        }

        if self.transforms:
            for fn in self.transforms:
                batch_dict = fn(batch_dict)

        if self.return_us:
            energy_ratios = batch_dict.pop("extra_dims")
            conds = batch_dict["energy"]
            return energy_ratios, conds
        else:
            ecal = batch_dict.pop("ecal")
            hcal = batch_dict.pop("hcal")
            shower = torch.cat((ecal, hcal), dim=2)
            conds = torch.cat(
                (
                    batch_dict["extra_dims"],
                    batch_dict["energy"],
                ),
                dim=-1,
            )
            # check if there are additional conditions, e.g. coming from LEMURS
            if batch_dict.get("additional_conds") is not None:
                conds = torch.cat((conds, batch_dict["additional_conds"]), dim=-1)
            return shower, conds

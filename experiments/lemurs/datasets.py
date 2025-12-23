import torch
import h5py
from collections import OrderedDict

from torch.utils.data import Dataset
from experiments.logger import LOGGER
from experiments.lemurs.transforms import *
from experiments.lemurs.utils import load_data


class LEMURSDataset(Dataset):
    """
    A "dumb" dataset that only loads raw, untransformed data for a given index.
    Designed to be used with a smart, batch-aware collate_fn.
    Includes timing for __getitem__.
    """

    def __init__(self, hdf5_files_dict, max_files_per_worker=4, dtype="float32"):
        self.max_open_files = max_files_per_worker
        self.open_files_cache = None
        self.worker_id = None
        self.dtype = dtype

        # create the index map in the main process
        self.label_to_idx = {label: i for i, label in enumerate(hdf5_files_dict.keys())}
        self.num_classes = len(self.label_to_idx)
        self.index_map = self._build_index_map(hdf5_files_dict)
        self.dataset_size = len(self.index_map)
        LOGGER.info(f"Dataset indexed with {self.dataset_size} samples.")

    def _build_index_map(self, hdf5_files_dict):
        """Create a map from a global index to a (file_path, local_index, class_idx) tuple."""
        index_map = []
        for label, file_list in hdf5_files_dict.items():
            class_idx = self.label_to_idx[label]
            for file_path in file_list:
                try:
                    with h5py.File(file_path, "r") as f:
                        num_samples = len(f["events"])
                        for local_idx in range(num_samples):
                            index_map.append((file_path, local_idx, class_idx))
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
            _old_path, old_file = self.open_files_cache.popitem(last=False)
            old_file.close()

        file_handle = h5py.File(file_path, "r")
        self.open_files_cache[file_path] = file_handle
        return file_handle

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        if self.worker_id is None:
            self._init_worker()

        file_path, local_idx, class_idx = self.index_map[idx]
        file_handle = self._get_file_handle(file_path)

        raw_data = load_data(file_handle, local_idx, dtype=self.dtype)
        raw_data["class_idx"] = class_idx
        return raw_data


class LEMURSCollator:
    """
    Collator that applies transformations at the batch level.
    """

    def __init__(
        self,
        hdf5_train_dict,
        transforms,
        num_classes,
        gen_label=None,
        return_us=False,
        rank=0,
        dtype="float32",
    ):
        self.hdf5_train_dict = hdf5_train_dict
        self.transforms = transforms
        self.num_classes = num_classes
        self.gen_label = gen_label
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
            if key != "class_idx"
        }
        class_indices = [item["class_idx"] for item in raw_batch]

        if self.gen_label is not None:
            labels = (
                torch.tensor(self.gen_label)
                .repeat(len(class_indices), 1)
                .to(self.dtype, non_blocking=True)
            )
        else:
            labels = torch.nn.functional.one_hot(
                torch.tensor(class_indices), self.num_classes
            ).float()
        batch_dict["label"] = labels

        if self.transforms:
            for fn in self.transforms:
                batch_dict = fn(batch_dict)

        if self.return_us:
            energy_ratios = batch_dict.pop("extra_dims")
            conds = torch.cat(
                (
                    batch_dict["incident_energy"],
                    batch_dict["incident_theta"],
                    batch_dict["incident_phi"],
                ),
                dim=-1,
            )
            return energy_ratios, conds
        else:
            shower = batch_dict.pop("showers")
            conds = torch.cat(
                (
                    batch_dict["extra_dims"],
                    batch_dict["incident_energy"],
                    batch_dict["incident_theta"],
                    batch_dict["incident_phi"],
                    batch_dict["label"],
                ),
                dim=-1,
            )
            return shower, conds

import torch


def load_data(hdf5_file, local_index=None, dtype="float32"):
    """
    Load data from a numpy .npz file containing ecal and hcal showers
    """
    slicer = local_index if local_index is not None else slice(None)
    event_data = hdf5_file["events"][slicer]
    data = {
        "energy": torch.from_numpy(event_data["energy"]).to(dtype),
        "ecal": torch.from_numpy(event_data["ecal"]).to(dtype),
        "hcal": torch.from_numpy(event_data["hcal"]).to(dtype),
    }

    # reshape if a single event is loaded
    if local_index is not None:
        for key, tensor in data.items():
            data[key] = tensor.unsqueeze(0)

    return data

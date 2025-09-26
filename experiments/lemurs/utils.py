import torch


def load_data(hdf5_file, local_index=None, dtype="float32"):
    """
    Load data from an HDF5 file containing a structured 'events' dataset.
    """
    slicer = local_index if local_index is not None else slice(None)

    event_data = hdf5_file["events"][slicer]
    data = {
        "incident_energy": torch.from_numpy(event_data["incident_energy"]).to(dtype),
        "incident_theta": torch.from_numpy(event_data["incident_theta"]).to(dtype),
        "incident_phi": torch.from_numpy(event_data["incident_phi"]).to(dtype),
        "showers": torch.from_numpy(event_data["showers"]).to(dtype),
    }

    # reshape if a single event is loaded
    if local_index is not None:
        for key, tensor in data.items():
            data[key] = tensor.unsqueeze(0)

    return data

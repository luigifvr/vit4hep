import numpy as np
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


def prepare_low_data_for_classifier(
    voxel_orig,
    E_inc_orig,
    angles_orig,
    hlf_class,
    label,
    cut=0.0,
    normed=False,
    single_energy=None,
):
    """takes hdf5_file, extracts Einc and voxel energies, appends label, returns array"""
    voxel = voxel_orig.copy()
    E_inc = E_inc_orig.copy()
    angles = angles_orig.copy()
    if normed:
        E_norm_rep = []
        E_norm = []
        for idx, layer_id in enumerate(hlf_class.GetElayers()):
            E_norm_rep.append(
                np.repeat(
                    hlf_class.GetElayers()[layer_id].reshape(-1, 1),
                    hlf_class.num_voxel[idx],
                    axis=1,
                )
            )
            E_norm.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
        E_norm_rep = np.concatenate(E_norm_rep, axis=1)
        E_norm = np.concatenate(E_norm, axis=1)
    if normed:
        voxel = voxel / (E_norm_rep + 1e-16)
        ret = np.concatenate(
            [
                np.log10(E_inc),
                voxel,
                np.log10(E_norm + 1e-8),
                label * np.ones_like(E_inc),
            ],
            axis=1,
        )
    else:
        voxel = voxel / E_inc
        ret = np.concatenate(
            [
                np.log10(E_inc),
                # angles,
                voxel,
                label * np.ones_like(E_inc),
            ],
            axis=1,
        )
    return ret


def prepare_high_data_for_classifier(
    voxel_orig, E_inc_orig, angles_orig, hlf_class, label, cut=0.0, single_energy=None
):
    """takes hdf5_file, extracts high-level features, appends label, returns array"""
    E_inc = E_inc_orig.copy()
    E_tot = hlf_class.GetEtot()
    angles = angles_orig.copy()
    E_layer = []
    for layer_id in hlf_class.GetElayers():
        E_layer.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
    EC_etas = []
    EC_phis = []
    Width_etas = []
    Width_phis = []
    for layer_id in hlf_class.layersBinnedInAlpha:
        EC_etas.append(hlf_class.GetECEtas()[layer_id].reshape(-1, 1))
        EC_phis.append(hlf_class.GetECPhis()[layer_id].reshape(-1, 1))
        Width_etas.append(hlf_class.GetWidthEtas()[layer_id].reshape(-1, 1))
        Width_phis.append(hlf_class.GetWidthPhis()[layer_id].reshape(-1, 1))
    E_layer = np.concatenate(E_layer, axis=1)
    EC_etas = np.concatenate(EC_etas, axis=1)
    EC_phis = np.concatenate(EC_phis, axis=1)
    Width_etas = np.concatenate(Width_etas, axis=1)
    Width_phis = np.concatenate(Width_phis, axis=1)
    ret = np.concatenate(
        [
            np.log10(E_inc),
            # angles,
            np.log10(E_layer + 1e-8),
            EC_etas / 1e2,
            EC_phis / 1e2,
            Width_etas / 1e2,
            Width_phis / 1e2,
            label * np.ones_like(E_inc),
        ],
        axis=1,
    )
    return ret

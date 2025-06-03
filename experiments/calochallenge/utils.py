import torch
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import yaml
import math
import numpy as np
import h5py

from challenge_files.XMLHandler import XMLHandler
import challenge_files.HighLevelFeatures as HLF
import experiments.calochallenge.transforms as transforms


def load_data(
    filename, particle_type, xml_filename, threshold=1e-5, single_energy=None
):
    """Loads the data for a datasets 1,2,3 from the calochallenge"""

    # Create a XML_handler to extract the layer boundaries. (Geometric setup is stored in the XML file)
    xml_handler = XMLHandler(particle_name=particle_type, filename=xml_filename)

    layer_boundaries = np.unique(xml_handler.GetBinEdges())

    # Prepare a container for the loaded data
    data = {}

    # Load and store the data. Make sure to slice according to the layers.
    # Also normalize to 100 GeV (The scale of the original data is MeV)
    data_file = h5py.File(filename, "r")
    # data["energy"] = data_file["incident_energies"][:]
    if single_energy is not None:
        energy_mask = data_file["incident_energies"][:] == single_energy
    else:
        energy_mask = np.full(len(data_file["incident_energies"]), True)

    data["energy"] = data_file["incident_energies"][:][energy_mask].reshape(-1, 1)
    for layer_index, (layer_start, layer_end) in enumerate(
        zip(layer_boundaries[:-1], layer_boundaries[1:])
    ):
        data[f"layer_{layer_index}"] = data_file["showers"][..., layer_start:layer_end][
            energy_mask.flatten()
        ]
    data_file.close()

    return data, layer_boundaries


def get_energy_and_sorted_layers(data):
    """returns the energy and the sorted layers from the data dict"""

    # Get the incident energies
    energy = data["energy"]

    # Get the number of layers layers from the keys of the data array
    number_of_layers = len(data) - 1

    # Create a container for the layers
    layers = []

    # Append the layers such that they are sorted.
    for layer_index in range(number_of_layers):
        layer = f"layer_{layer_index}"

        layers.append(data[layer])

    layers = np.concatenate(layers, axis=1)

    return energy, layers

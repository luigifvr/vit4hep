import h5py


def load_data(data_file):
    full_file = h5py.File(data_file, "r")
    layer_0 = full_file["layer_0"][:] / 1e3
    layer_1 = full_file["layer_1"][:] / 1e3
    layer_2 = full_file["layer_2"][:] / 1e3
    energy = full_file["energy"][:] / 1e0

    full_file.close()

    data = {
        "layer_0": layer_0,
        "layer_1": layer_1,
        "layer_2": layer_2,
        "energy": energy,
    }
    return data

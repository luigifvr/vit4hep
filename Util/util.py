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

"""
Some useful utility functions that don"t fit in anywhere else
"""


def load_params(path):
    """
    Method to load a parameter dict from a yaml file
    :param path: path to a *.yaml parameter file
    :return: the parameters as a dict
    """
    with open(path) as f:
        param = yaml.load(f, Loader=yaml.FullLoader)
        return param


def save_params(params, name="paramfile.yaml"):
    """
    Method to save a parameter dict to a yaml file
    :param params: the parameter dict
    :param name: the name of the yaml file
    """
    with open(name, 'w') as f:
        yaml.dump(params, f)


def get(dict, key, default):
    """
    Method to extract a key from a dict.
    If the key is not contained in the dict, the default value is returned and written into the dict.
    :param dict: the dictionary
    :param key: the key
    :param default: the default value of the key
    :return: the value of the key in the dict if it exists, the default value otherwise
    """

    if key in dict:
        return dict[key]
    else:
        dict[key] = default
        return default


def get_device():
    """Check whether cuda can be used"""
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    return device

def linear_beta_schedule(timesteps):
    """
    linear beta schedule for DDPM diffusion models
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine beta schedule for DDPM diffusion models
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def load_data(filename, particle_type,  xml_filename, threshold=1e-5, single_energy=None):
    """Loads the data for a dataset 1 from the calo challenge"""
    
    # Create a XML_handler to extract the layer boundaries. (Geometric setup is stored in the XML file)
    xml_handler = XMLHandler(particle_name=particle_type, 
    filename=xml_filename)
    
    layer_boundaries = np.unique(xml_handler.GetBinEdges())

    # Prepare a container for the loaded data
    data = {}

    # Load and store the data. Make sure to slice according to the layers.
    # Also normalize to 100 GeV (The scale of the original data is MeV)
    data_file = h5py.File(filename, 'r')
    #data["energy"] = data_file["incident_energies"][:]
    if single_energy is not None:
        energy_mask = data_file["incident_energies"][:] == single_energy
    else:
        energy_mask = np.full(len(data_file["incident_energies"]), True)

    data["energy"] = data_file["incident_energies"][:][energy_mask].reshape(-1, 1)
    for layer_index, (layer_start, layer_end) in enumerate(zip(layer_boundaries[:-1], layer_boundaries[1:])):
        data[f"layer_{layer_index}"] = data_file["showers"][..., layer_start:layer_end][energy_mask.flatten()]
    data_file.close()
    
    return data, layer_boundaries

def get_energy_and_sorted_layers(data):
    """returns the energy and the sorted layers from the data dict"""
    
    # Get the incident energies
    energy = data["energy"]

    # Get the number of layers layers from the keys of the data array
    number_of_layers = len(data)-1
    
    # Create a container for the layers
    layers = []

    # Append the layers such that they are sorted.
    for layer_index in range(number_of_layers):
        layer = f"layer_{layer_index}"
        
        layers.append(data[layer])
       
    layers = np.concatenate(layers, axis=1)
            
    return energy, layers

def get_transformations(transforms_list, doc=None):
    func = []
    for name, kwargs in transforms_list.items():
        if name == 'StandardizeFromFile' and doc is not None:
            kwargs['model_dir'] = doc.basedir
        func.append(getattr(transforms, name)(**kwargs))
    return func

def set_scheduler(optimizer, params, steps_per_epoch=1, last_epoch=-1):
    lr_sched_mode = params.get("lr_scheduler", "reduce_on_plateau")

    if lr_sched_mode == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size = params["lr_decay_epochs"],
            gamma = params["lr_decay_factor"],
            last_epoch=last_epoch,
        )
    elif lr_sched_mode == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor = 0.8, patience = 50, cooldown = 100,
            threshold = 5e-5, threshold_mode = "rel", verbose=True,
            last_epoch=last_epoch,
        )
    elif lr_sched_mode == "one_cycle_lr":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, params.get("max_lr", params["lr"]*10),
            epochs = params.get("cycle_epochs") or params["n_epochs"],
            steps_per_epoch=steps_per_epoch,
            pct_start=params.get("cycle_pct_start", 0.3),
            last_epoch=last_epoch,
        )
    elif lr_sched_mode == "cycle_lr":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr = params.get("lr", 1.0e-4),
            max_lr = params.get("max_lr", params["lr"]*10),
            step_size_up= params.get("step_size_up", 2000),
            mode = params.get("cycle_mode", "triangular"),
            cycle_momentum = False,
            last_epoch=last_epoch,
       )
    elif lr_sched_mode == "multi_step_lr":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[2730, 8190, 13650, 27300], gamma=0.5,
            last_epoch=last_epoch,
       )
    elif lr_sched_mode == "CosineAnnealing":
        n_epochs = params.get("cycle_epochs") or params["n_epochs"]
        eta_min = params.get( "eta_min", 0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs * steps_per_epoch, eta_min=eta_min,
            last_epoch=last_epoch,
       )
    elif lr_sched_mode == "CosineAnnealingWarmRestarts":
        n_epochs = params.get("cycle_epochs") or params["n_epochs"]
        eta_min = params.get( "eta_min", 0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=int((n_epochs * steps_per_epoch)/2), eta_min=eta_min,
            last_epoch=last_epoch,
       )
    else:
        raise ValueError(f"scheduler f\"{lr_sched_mode}\" not recognised.")

    return scheduler

def sumlogC( x , eps = 5e-3):
    '''
    Numerically stable implementation of
    sum of logarithm of Continous Bernoulli
    constant C, using Taylor 2nd degree approximation

    Parameter
    ----------
    x : Tensor of dimensions (batch_size, dim)
        x takes values in (0,1)
    '''
    x = torch.clamp(x, eps, 1.-eps)
    mask = torch.abs(x - 0.5).ge(eps)
    far = torch.masked_select(x, mask)
    close = torch.masked_select(x, ~mask)
    #far_values =  torch.log( (torch.log(1. - far) - torch.log(far)).div(1. - 2. * far) )
    far_values = torch.log( 2*torch.atanh(1-2*far)/(1-2*far) )
    #close_values = torch.log(torch.tensor((2.))) + torch.log(1. + torch.pow( 1. - 2. * close, 2)/3. )
    close_values = math.log(2.0) + (4.0 / 3.0 + 104.0 / 45.0 * x) * x 
    return far_values.sum() + close_values.sum()

def loss_cbvae(recon_x, x): # mu, logvar):
    '''
    Loss function for continuous bernoulli vae
    '''
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    LOGC = -sumlogC(recon_x)
    #return BCE + KLD + LOGC
    return BCE + LOGC

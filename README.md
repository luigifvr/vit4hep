# VisionTransformers4HEP
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pytorch](https://img.shields.io/badge/PyTorch_2.7-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
![python](https://img.shields.io/badge/python-3.12%2B-blue)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

A codebase for applying Vision Transformers (ViTs) to calorimeter data in high-energy physics (HEP). It supports training, evaluation, and fine-tuning of transformer-based models on calorimeter simulation datasets.

We modularize and extend the [CaloDREAM](https://github.com/luigifvr/calo_dreamer) package for easier training on different datasets for calorimeter shower generation and beyond.

## Installation

Clone the repository:

```bash
git clone git@github.com:luigifvr/vit4hep.git
cd vit4hep
```

Set up a Python environment (recommended: Python 3.12+):

```bash
python -m venv ~/venvs/vit
source ~/venvs/vit/bin/activate
pip install -r requirements.txt
```

Datasets have to be collected separately, and the various data paths in the provided config files should be changed to the directories containing those files.
For instance, [GitHub CaloChallenge](https://calochallenge.github.io/homepage/) provides links to the CaloChallenge datasets.

## Usage

A run trains only a single network. Therefore, the energy and shape networks are trained separately. The model type can be set in the main configuration file with `model_type=shape` or `model_type=energy`.

It is possible to train a shape network without sampling the energy ratios by setting `sample_us=false`. Beware that the conditions will be taken from the test file, and the evaluation can become very slow for large datasets.

If `sample_us=true`, the code expects a valid path to a trained energy model in `energy_model=path/to/model`.

### Training

Existing experiments can be run by setting the configuration file and its directory. For example, for the shape model for the CaloChallenge-ds2:

```bash
python main.py --config-dir configs/calochallenge/calochallenge_cfm/ -cn calochallenge_ds2
```

Keys defined in the configuration files can be overwritten simply by calling them, e.g.

```bash
python main.py --config-dir configs/calochallenge/calochallenge_cfm/ -cn calochallenge_ds2 model=cfm/cfm_ds2_electrons training.batchsize=128 training.iterations=1000 exp_name="calo_test"
```

Here, we explicitly call the model configuration file, set the batch size, the number of training iterations, and the name of the experiment.

Further, we use MLflow for tracking. You can start an MLflow server based on the saved results in runs/tracking/mlflow.db on port 4242 of your machine with the following command

```bash
mlflow ui --port 4242 --backend-store-uri sqlite:///runs/tracking/mlflow.db
```

### Validation
An existing run can be reloaded to perform additional tests with the trained model. For a previous run with exp_name=calo_test and run_name=calo_0000, one can run, for example

```bash
python main.py -cn config -cp runs/calo_test/calo_0000 train=false warm_start_idx=0
```

The warm_start_idx specifies which model in the models folder should be loaded and defaults to 0. 

It is also possible to rerun the evaluation of a saved sample by passing the file path to `load_sample`, following the previous example:

```bash
python main.py -cn config -cp runs/calo_test/calo_0000 train=false warm_start_idx=0 plot=false load_sample=<file_path>
```

The option `plot=false` skips the sampling function (soon will be changed to a more intuitive name).

## Structure

- `experiments/` — Experiment classes and training logic
- `nn/` — Model definitions (ViT, heads, etc.)
- `models/` — Base models for CFM and cINN 
- `configs/` — Hydra configuration files

## Training on new experiments

Using this code on a new dataset requires three dataset-specific components:

- A dataset class which handles the preprocessing, or the lazy data collection if necessary;
- A model which inherits from the provided base models, which can be useful for custom patching functions;
- The set of preprocessing steps implemented as classes (see the various `transforms.py` for examples).

### Configuration files

The main configuration file contains keys that are dataset-specific under `data`. In particular, we define the preprocessing steps, with the respective arguments, as a dictionary in `transforms`. The preprocessing can be readily changed by modifying such a dictionary.

Other keys that are expected to be always relevant are outside the `data` key. This includes the model type, `shape` or `energy`, the path to the energy model, the number of samples, the evaluation details, etc.

## Citation

If you use this codebase in your research, please consider citing:

```
@article{Favaro:2024rle,
 author = "Favaro, Luigi and Ore, Ayodele and Schweitzer, Sofia Palacios and Plehn, Tilman",
 title = "{CaloDREAM -- Detector Response Emulation via Attentive flow Matching}",
 eprint = "2405.09629",
 archivePrefix = "arXiv",
 primaryClass = "hep-ph",
 doi = "10.21468/SciPostPhys.18.3.088",
 journal = "SciPost Phys.",
 volume = "18",
 pages = "088",
 year = "2025"
}

@inproceedings{Favaro:2025ift,
 author = "Favaro, Luigi and Giammanco, Andrea and Krause, Claudius",
 title = "{Fast, accurate, and precise detector simulation with vision transformers}",
 booktitle = "{2nd European AI for Fundamental Physics Conference}",
 eprint = "2509.25169",
 archivePrefix = "arXiv",
 primaryClass = "hep-ph",
 reportNumber = "IRMP-CP3-25-33",
 month = "9",
 year = "2025"
}
```

## License

See `LICENSE` for details.

## Contact

For questions or contributions, [contact us](mailto:luigi.favaro@uclouvain.be).
# VisionTransformers4HEP
[![pytorch](https://img.shields.io/badge/PyTorch_2.2+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)

A codebase for applying Vision Transformers (ViTs) to calorimeter data in high-energy physics (HEP). It supports training, evaluation, and fine-tuning of transformer-based models on calorimeter simulation datasets, including the CaloChallenge datasets.

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

Datasets have to be collected separately and the various data paths in the provided config files should be changed to the directories containing those files.
For instance, [GitHub CaloChallenge](https://calochallenge.github.io/homepage/) provides links to the CaloChallenge datasets.

## Usage

### Training

Run an experiment using a configuration file:
```bash
python main.py -cn calochallenge model=cfm/cfm_ds2_electrons training.batchsize=128 training.iterations=1000 exp_name="calo_test"
```
Other keys defined in the configuration files can be overwritten in a similar fashion.

Further, we use mlflow for tracking. You can start a mlflow server based on the saved results in runs/tracking/mlflow.db on port 4242 of your machine with the following command

```bash
mlflow ui --port 4242 --backend-store-uri sqlite:///runs/tracking/mlflow.db
```

An existing run can be reloaded to perform additional tests with the trained model. For a previous run with exp_name=calo_test and run_name=calo_0000, one can run for example

```bash
python run.py -cn config -cp runs/calo_test/calo_0000 train=false warm_start_idx=0
```

The warm_start_idx specifies which model in the models folder should be loaded and defaults to 0. 

## Project Structure

- `experiments/` — Experiment classes and training logic
- `nn/` — Model definitions (ViT, heads, etc.)
- `models/` — Base models for CFM and cINN 
- `configs/` — Hydra configuration files

## Citation

If you use this codebase in your research, please cite:

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

@article{Favaro:2024rle,
    author = "Favaro, Luigi and Giammanco, Andrea and Krause, Claudius",
    title = "{A universal Vision Transformer for fast detector simulation}",
    eprint = "xxxx.xxxxx"
}
```

## License

See `LICENSE` for details.

## Contact

For questions or contributions, [contact us](mailto:luigi.favaro@uclouvain.be)
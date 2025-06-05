import hydra
from experiments.calochallenge.experiment import CaloChallenge
from experiments.calochallenge.experiment_finetuning import CaloChallengeFT


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg):
    if cfg.exp_type == "calochallenge":
        exp = CaloChallenge(cfg)
    elif cfg.exp_type == "calochallenge_ft":
        exp = CaloChallengeFT(cfg)
    else:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented")

    exp()


if __name__ == "__main__":
    main()

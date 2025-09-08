import hydra
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from experiments.calochallenge.experiment import CaloChallenge
from experiments.calochallenge.calochallenge_cfm.experiment_finetuning import (
    CaloChallengeFTCFM,
)
from experiments.calogan.experiment import CaloGAN
from experiments.calogan.experiment_finetuning import CaloGANFTCFM


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg):
    world_size = torch.cuda.device_count()
    if world_size > 1:
        mp.spawn(
            experiment,
            args=(world_size, cfg),
            nprocs=world_size,
        )
    else:
        experiment(0, world_size, cfg)


def experiment(rank, world_size, cfg):
    # Initialize the process group for distributed training
    if world_size > 1:
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )
        torch.cuda.set_device(rank)
    if cfg.exp_type == "calochallenge":
        exp = CaloChallenge(cfg, rank, world_size)
    elif cfg.exp_type == "calochallenge_ft_cfm":
        exp = CaloChallengeFTCFM(cfg, rank, world_size)
    elif cfg.exp_type == "calogan":
        exp = CaloGAN(cfg, rank, world_size)
    elif cfg.exp_type == "calogan_ft_cfm":
        exp = CaloGANFTCFM(cfg, rank, world_size)
    else:
        raise ValueError(f"exp_type {cfg.exp_type} not implemented")

    exp()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

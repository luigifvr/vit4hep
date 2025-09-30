import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from experiments.logger import LOGGER
from experiments.calo_utils.ugr_evaluation.evaluate import (
    DNN,
    ttv_split,
    train_and_evaluate_cls,
    evaluate_cls,
    load_classifier,
)


class args_class:
    def __init__(self, cfg):
        cfg = cfg.evaluation
        self.dataset = cfg.eval_dataset
        self.mode = cfg.eval_mode
        self.which_cuda = 0

        self.cls_n_layer = cfg.eval_cls_n_layer
        self.cls_n_hidden = cfg.eval_cls_n_hidden
        self.cls_dropout_probability = cfg.eval_cls_dropout
        self.cls_lr = cfg.eval_cls_lr
        self.cls_batch_size = cfg.eval_cls_batch_size
        self.cls_n_epochs = cfg.eval_cls_n_epochs
        self.save_mem = cfg.eval_cls_save_mem


def eval_ui_dists(source_array, reference_array, cfg):
    if not os.path.isdir(cfg.run_dir + f"/eval_{cfg.run_idx}/"):
        os.makedirs(cfg.run_dir + f"/eval_{cfg.run_idx}/")

    args = args_class(cfg)
    args.output_dir = cfg.run_dir + f"/eval_{cfg.run_idx}/"

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # add label in source array
    source_array = np.concatenate(
        (source_array, np.zeros(source_array.shape[0]).reshape(-1, 1)), axis=1
    )
    reference_array = np.concatenate(
        (reference_array, np.ones(reference_array.shape[0]).reshape(-1, 1)), axis=1
    )
    train_data, test_data, val_data = ttv_split(source_array, reference_array)

    # set up device
    args.device = torch.device(
        "cuda:" + str(args.which_cuda) if torch.cuda.is_available() else "cpu"
    )
    print("Using {}".format(args.device))

    # set up DNN classifier
    input_dim = train_data.shape[1] - 1
    DNN_kwargs = {
        "num_layer": args.cls_n_layer,  # 2
        "num_hidden": args.cls_n_hidden,  # 512
        "input_dim": input_dim,
        "dropout_probability": args.cls_dropout_probability,
    }  # 0
    classifier = DNN(**DNN_kwargs)
    classifier.to(args.device)
    print(classifier)
    total_parameters = sum(
        p.numel() for p in classifier.parameters() if p.requires_grad
    )

    print("{} has {} parameters".format(args.mode, int(total_parameters)))

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.cls_lr)

    train_data = TensorDataset(
        torch.tensor(train_data, dtype=torch.get_default_dtype()).to(args.device)
    )
    test_data = TensorDataset(
        torch.tensor(test_data, dtype=torch.get_default_dtype()).to(args.device)
    )
    val_data = TensorDataset(
        torch.tensor(val_data, dtype=torch.get_default_dtype()).to(args.device)
    )

    train_dataloader = DataLoader(
        train_data, batch_size=args.cls_batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_data, batch_size=args.cls_batch_size, shuffle=False
    )
    val_dataloader = DataLoader(val_data, batch_size=args.cls_batch_size, shuffle=False)

    train_and_evaluate_cls(
        classifier, train_dataloader, test_dataloader, optimizer, args
    )
    classifier = load_classifier(classifier, args)

    with torch.inference_mode():
        print("Now looking at independent dataset:")
        eval_acc, eval_auc, eval_JSD = evaluate_cls(
            classifier,
            val_dataloader,
            args,
            final_eval=True,
            calibration_data=test_dataloader,
        )
    LOGGER.info("Final result of classifier test (AUC / JSD):")
    LOGGER.info("{:.4f} / {:.4f}".format(eval_auc, eval_JSD))
    with open(
        os.path.join(
            args.output_dir, "classifier_{}_{}.txt".format(args.mode, args.dataset)
        ),
        "a",
    ) as f:
        f.write(
            "Final result of classifier test (AUC / JSD):\n"
            + "{:.4f} / {:.4f}\n\n".format(eval_auc, eval_JSD)
        )

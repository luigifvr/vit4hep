import os

import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

from experiments.calogan.utils import load_data

torch.set_default_dtype(torch.float64)

plt.rc("font", family="serif", size=16)
plt.rc("axes", titlesize="medium")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
plt.rc("text", usetex=True)


########## Functions and Classes ##########
class DNN(torch.nn.Module):
    """NN for vanilla classifier. Does not have sigmoid activation in last layer, should
    be used with torch.nn.BCEWithLogitsLoss()
    """

    def __init__(self, num_layer, num_hidden, input_dim, dropout_probability=0.0):
        super(DNN, self).__init__()

        self.dpo = dropout_probability

        self.inputlayer = torch.nn.Linear(input_dim, num_hidden)
        self.outputlayer = torch.nn.Linear(num_hidden, 1)

        all_layers = [self.inputlayer, torch.nn.LeakyReLU(), torch.nn.Dropout(self.dpo)]
        for _ in range(num_layer):
            all_layers.append(torch.nn.Linear(num_hidden, num_hidden))
            all_layers.append(torch.nn.LeakyReLU())
            all_layers.append(torch.nn.Dropout(self.dpo))

        all_layers.append(self.outputlayer)
        self.layers = torch.nn.Sequential(*all_layers)

    def forward(self, x):
        """Forward pass through the DNN"""
        x = self.layers(x)
        return x


def ttv_split(data1, data2, split=np.array([0.6, 0.2, 0.2])):
    """splits data1 and data2 in train/test/val according to split,
    returns shuffled and merged arrays
    """
    # assert len(data1) == len(data2)
    if len(data1) < len(data2):
        data2 = data2[: len(data1)]
    elif len(data1) > len(data2):
        data1 = data1[: len(data2)]
    else:
        assert len(data1) == len(data2)
    num_events = (len(data1) * split).astype(int)
    np.random.shuffle(data1)
    np.random.shuffle(data2)
    train1, test1, val1 = np.split(data1, num_events.cumsum()[:-1])
    train2, test2, val2 = np.split(data2, num_events.cumsum()[:-1])
    train = np.concatenate([train1, train2], axis=0)
    test = np.concatenate([test1, test2], axis=0)
    val = np.concatenate([val1, val2], axis=0)
    np.random.shuffle(train)
    np.random.shuffle(test)
    np.random.shuffle(val)
    print(len(train), len(test), len(val))
    return train, test, val


def load_classifier(constructed_model, parser_args):
    """loads a saved model"""
    filename = parser_args.mode + "_" + parser_args.dataset + ".pt"
    checkpoint = torch.load(
        os.path.join(parser_args.output_dir, filename), map_location=parser_args.device
    )
    constructed_model.load_state_dict(checkpoint["model_state_dict"])
    constructed_model.to(parser_args.device)
    constructed_model.eval()
    print("classifier loaded successfully")
    return constructed_model


def train_and_evaluate_cls(model, data_train, data_test, optim, arg):
    """train the model and evaluate along the way"""
    best_eval_acc = float("-inf")
    arg.best_epoch = -1
    try:
        for i in range(arg.cls_n_epochs):
            train_cls(model, data_train, optim, i, arg)
            with torch.inference_mode():
                eval_acc, _, _ = evaluate_cls(model, data_test, arg)
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                arg.best_epoch = i + 1
                filename = arg.mode + "_" + arg.dataset + ".pt"
                torch.save(
                    {"model_state_dict": model.state_dict()},
                    os.path.join(arg.output_dir, filename),
                )
            if eval_acc == 1.0:
                break
    except KeyboardInterrupt:
        # training can be cut short with ctrl+c, for example if overfitting between train/test set
        # is clearly visible
        pass


def train_cls(model, data_train, optim, epoch, arg):
    """train one step"""
    model.train()
    for i, data_batch in enumerate(data_train):
        if arg.save_mem:
            data_batch = data_batch[0].to(arg.device)
        else:
            data_batch = data_batch[0]
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector)
        criterion = torch.nn.BCEWithLogitsLoss()
        loss = criterion(output_vector, target_vector.unsqueeze(1))

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if i % (len(data_train) // 2) == 0:
            print(
                "Epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}".format(
                    epoch + 1, arg.cls_n_epochs, i, len(data_train), loss.item()
                )
            )
        # PREDICTIONS
        pred = torch.round(torch.sigmoid(output_vector.detach()))
        target = torch.round(target_vector.detach())
        if i == 0:
            res_true = target
            res_pred = pred
        else:
            res_true = torch.cat((res_true, target), 0)
            res_pred = torch.cat((res_pred, pred), 0)

    print("Accuracy on training set is", accuracy_score(res_true.cpu(), res_pred.cpu()))


def evaluate_cls(model, data_test, arg, final_eval=False, calibration_data=None):
    """evaluate on test set"""
    model.eval()
    for j, data_batch in enumerate(data_test):
        if arg.save_mem:
            data_batch = data_batch[0].to(arg.device)
        else:
            data_batch = data_batch[0]
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector)
        pred = output_vector.reshape(-1)
        target = target_vector.double()
        if j == 0:
            result_true = target
            result_pred = pred
        else:
            result_true = torch.cat((result_true, target), 0)
            result_pred = torch.cat((result_pred, pred), 0)
    BCE = torch.nn.BCEWithLogitsLoss()(result_pred, result_true)
    result_pred = torch.sigmoid(result_pred).cpu().numpy()
    result_true = result_true.cpu().numpy()
    eval_acc = accuracy_score(result_true, np.round(result_pred))
    print("Accuracy on test set is", eval_acc)
    eval_auc = roc_auc_score(result_true, result_pred)
    print("AUC on test set is", eval_auc)
    JSD = -BCE + np.log(2.0)
    print(
        "BCE loss of test set is {:.4f}, JSD of the two dists is {:.4f}".format(
            BCE, JSD / np.log(2.0)
        )
    )
    if final_eval:
        prob_true, prob_pred = calibration_curve(result_true, result_pred, n_bins=10)
        print("unrescaled calibration curve:", prob_true, prob_pred)
        calibrator = calibrate_classifier(model, calibration_data, arg)
        rescaled_pred = calibrator.predict(result_pred)
        eval_acc = accuracy_score(result_true, np.round(rescaled_pred))
        print("Rescaled accuracy is", eval_acc)
        eval_auc = roc_auc_score(result_true, rescaled_pred)
        print("rescaled AUC of dataset is", eval_auc)
        prob_true, prob_pred = calibration_curve(result_true, rescaled_pred, n_bins=10)
        print("rescaled calibration curve:", prob_true, prob_pred)
        # calibration was done after sigmoid, therefore only BCELoss() needed here:
        BCE = torch.nn.BCELoss()(
            torch.tensor(rescaled_pred, dtype=torch.get_default_dtype()),
            torch.tensor(result_true, dtype=torch.get_default_dtype()),
        )
        JSD = -BCE.cpu().numpy() + np.log(2.0)
        otp_str = (
            "rescaled BCE loss of test set is {:.4f}, "
            + "rescaled JSD of the two dists is {:.4f}"
        )
        print(otp_str.format(BCE, JSD / np.log(2.0)))
    return eval_acc, eval_auc, JSD / np.log(2.0)


def calibrate_classifier(model, calibration_data, arg):
    """reads in calibration data and performs a calibration with isotonic regression"""
    model.eval()
    assert calibration_data is not None, "Need calibration data for calibration!"
    for j, data_batch in enumerate(calibration_data):
        if arg.save_mem:
            data_batch = data_batch[0].to(arg.device)
        else:
            data_batch = data_batch[0]
        input_vector, target_vector = data_batch[:, :-1], data_batch[:, -1]
        output_vector = model(input_vector)
        pred = torch.sigmoid(output_vector).reshape(-1)
        target = target_vector.to(torch.float64)
        if j == 0:
            result_true = target
            result_pred = pred
        else:
            result_true = torch.cat((result_true, target), 0)
            result_pred = torch.cat((result_pred, pred), 0)
    result_true = result_true.cpu().numpy()
    result_pred = result_pred.cpu().numpy()
    iso_reg = IsotonicRegression(
        out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6
    ).fit(result_pred, result_true)
    return iso_reg


def eval_calogan_lowlevel(source_array, cfg):
    if not os.path.isdir(cfg.run_dir + f"/eval_{cfg.run_idx}/"):
        os.makedirs(cfg.run_dir + f"/eval_{cfg.run_idx}/")

    args = args_class(cfg)
    args.output_dir = cfg.run_dir + f"/eval_{cfg.run_idx}/"

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    reference_data = load_data(cfg.eval_hdf5_file)
    reference_array = np.hstack(
        (
            reference_data["layer_0"].reshape(-1, 288),
            reference_data["layer_1"].reshape(-1, 144),
            reference_data["layer_2"].reshape(-1, 72),
        ),
    )

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
    }
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
    print("Final result of classifier test (AUC / JSD):")
    print("{:.4f} / {:.4f}".format(eval_auc, eval_JSD))
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


class args_class:
    def __init__(self, cfg):
        self.dataset = cfg.eval_dataset
        self.mode = cfg.eval_mode
        self.cut = cfg.eval_cut
        self.reference_file = cfg.eval_hdf5_file
        self.which_cuda = 0

        self.cls_resnet_layers = cfg.eval_cls_resnet_layers
        self.cls_n_layer = cfg.eval_cls_n_layer
        self.cls_n_hidden = cfg.eval_cls_n_hidden
        self.cls_dropout_probability = cfg.eval_cls_dropout
        self.cls_lr = cfg.eval_cls_lr
        self.cls_batch_size = cfg.eval_cls_batch_size
        self.cls_n_epochs = cfg.eval_cls_n_epochs
        self.save_mem = cfg.eval_cls_save_mem

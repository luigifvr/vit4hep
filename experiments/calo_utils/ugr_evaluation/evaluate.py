import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

import experiments.calo_utils.ugr_evaluation.HighLevelFeatures as HLF
from experiments.calo_utils.ugr_evaluation.evaluate_plotting_helper import (
    plot_cell_dist,
    plot_E_layers,
    plot_ECEtas,
    plot_ECPhis,
    plot_ECWidthEtas,
    plot_ECWidthPhis,
    plot_Etot_Einc,
    plot_layer_comparison,
    plot_sparsity,
    plot_weighted_depth_a,
    plot_weighted_depth_r,
)
from experiments.calo_utils.ugr_evaluation.resnet import generate_model

torch.set_default_dtype(torch.float64)

plt.rc("font", family="serif", size=16)
plt.rc("axes", titlesize="medium")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
plt.rc("text", usetex=True)
# hardcoded labels for histograms
labels = ["ViT-CFM"]

########## Functions and Classes ##########


class DNN(torch.nn.Module):
    """NN for vanilla classifier. Does not have sigmoid activation in last layer, should
    be used with torch.nn.BCEWithLogitsLoss()
    """

    def __init__(self, num_layer, num_hidden, input_dim, dropout_probability=0.0):
        super().__init__()

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


def prepare_low_data_for_classifier(
    voxel_orig, E_inc_orig, hlf_class, label, cut=0.0, normed=False, single_energy=None
):
    """takes hdf5_file, extracts Einc and voxel energies, appends label, returns array"""
    voxel = voxel_orig.copy()
    E_inc = E_inc_orig.copy()
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
            [np.log10(E_inc), voxel, label * np.ones_like(E_inc)], axis=1
        )
    return ret


def prepare_high_data_for_classifier(
    voxel_orig, E_inc_orig, hlf_class, label, cut=0.0, single_energy=None
):
    """takes hdf5_file, extracts high-level features, appends label, returns array"""
    E_inc = E_inc_orig.copy()
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


def ttv_split(data1, data2, split=[0.6, 0.2, 0.2]):
    """splits data1 and data2 in train/test/val according to split,
    returns shuffled and merged arrays
    """
    split = np.array(split)
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
    if model.__class__.__name__ == "ResNet":
        n_epochs = arg.cls_resnet_epochs
    else:
        n_epochs = arg.cls_n_epochs
    try:
        for i in range(n_epochs):
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
                f"Epoch {epoch + 1:3d} / {arg.cls_n_epochs}, step {i:4d} / {len(data_train)}; loss {loss.item():.4f}"
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
        f"BCE loss of test set is {BCE:.4f}, JSD of the two dists is {JSD / np.log(2.0):.4f}"
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


def check_file(given_file, arg, which=None):
    """checks if the provided file has the expected structure based on the dataset"""
    print(
        "Checking if {} file has the correct form ...".format(
            which if which is not None else "provided"
        )
    )
    num_features = {
        "1-photons": 368,
        "1-pions": 533,
        "2": 6480,
        "3": 40500,
        "LEMURS": 6480,
    }[arg.dataset]
    num_events = given_file["incident_energies"].shape[0]
    assert (
        given_file["showers"].shape[0] == num_events
    ), "Number of energies provided does not match number of showers, {} != {}".format(
        num_events, given_file["showers"].shape[0]
    )
    assert (
        given_file["showers"].shape[1] == num_features
    ), "Showers have wrong shape, expected {}, got {}".format(
        num_features, given_file["showers"].shape[1]
    )

    print(f"Found {num_events} events in the file.")
    print(
        "Checking if {} file has the correct form: DONE \n".format(
            which if which is not None else "provided"
        )
    )


def extract_shower_and_energy(given_file, which, single_energy=None, max_len=-1):
    """reads .hdf5 file and returns samples and their energy"""
    print(f"Extracting showers from {which} file ...")
    if single_energy is not None:
        energy_mask = given_file["incident_energies"][:] == single_energy
        energy = given_file["incident_energies"][:][energy_mask].reshape(-1, 1)
        shower = given_file["showers"][:][energy_mask.flatten()]
    else:
        shower = given_file["showers"][:max_len]
        energy = given_file["incident_energies"][:max_len]
    print(f"Extracting showers from {which} file: DONE.\n")
    return shower.astype("float32", copy=False), energy.astype("float32", copy=False)


def plot_histograms(
    hlf_classes, reference_class, arg, labels, input_names="", p_label=""
):
    """plots histograms based with reference file as comparison"""
    plot_Etot_Einc(hlf_classes, reference_class, arg, labels, input_names, p_label)
    plot_E_layers(hlf_classes, reference_class, arg, labels, input_names, p_label)
    plot_ECEtas(hlf_classes, reference_class, arg, labels, input_names, p_label)
    plot_ECPhis(hlf_classes, reference_class, arg, labels, input_names, p_label)
    plot_ECWidthEtas(hlf_classes, reference_class, arg, labels, input_names, p_label)
    plot_ECWidthPhis(hlf_classes, reference_class, arg, labels, input_names, p_label)
    plot_sparsity(hlf_classes, reference_class, arg, labels, input_names, p_label)
    plot_weighted_depth_a(
        hlf_classes, reference_class, arg, labels, input_names, p_label
    )
    plot_weighted_depth_r(
        hlf_classes, reference_class, arg, labels, input_names, p_label
    )


class args_class:
    def __init__(self, cfg):
        cfg = cfg.evaluation
        self.dataset = cfg.eval_dataset
        self.mode = cfg.eval_mode
        self.cut = cfg.eval_cut
        self.reference_file = cfg.eval_hdf5_file
        self.which_cuda = 0
        self.p_label = cfg.eval_p_label

        # ResNet classifier parameters
        self.cls_resnet_layers = cfg.eval_cls_resnet_layers
        self.cls_resnet_lr = cfg.eval_cls_resnet_lr
        self.cls_resnet_epochs = cfg.eval_cls_resnet_n_epochs

        self.cls_n_layer = cfg.eval_cls_n_layer
        self.cls_n_hidden = cfg.eval_cls_n_hidden
        self.cls_dropout_probability = cfg.eval_cls_dropout
        self.cls_lr = cfg.eval_cls_lr
        self.cls_batch_size = cfg.eval_cls_batch_size
        self.cls_n_epochs = cfg.eval_cls_n_epochs
        self.save_mem = cfg.eval_cls_save_mem


def run_from_py(sample, energy, cfg):
    print("Running evaluation script run_from_py:")

    if not os.path.isdir(cfg.run_dir + f"/eval_{cfg.run_idx}/"):
        os.makedirs(cfg.run_dir + f"/eval_{cfg.run_idx}/")

    args = args_class(cfg)
    args.output_dir = cfg.run_dir + f"/eval_{cfg.run_idx}/"
    print("Input sample of shape: ")
    print(sample.shape)
    particle = {
        "1-photons": "photon",
        "1-pions": "pion",
        "2": "electron",
        "3": "electron",
        "LEMURS": "gamma",
    }[args.dataset]
    args.particle = particle

    args.min_energy = {
        "1-photons": 0.001,
        "1-pions": 0.001,
        "2": 0.5e-3 / 0.033,
        "3": 0.5e-3 / 0.033,
        "LEMURS": 0.5e-3 / 0.033,
    }[args.dataset]

    hlf = HLF.HighLevelFeatures(particle, filename=cfg.data.xml_filename)

    # Checking for negative values, nans and infinities
    print("Checking for negative values, number of negative energies: ")
    print("input: ", (sample < 0.0).sum(), "\n")
    print("Checking for nans in the generated sample, number of nans: ")
    print("input: ", np.isnan(sample).sum(), "\n")
    print("Checking for infs in the generated sample, number of infs: ")
    print("input: ", np.isinf(sample).sum(), "\n")
    np.nan_to_num(sample, copy=False, nan=0.0, neginf=0.0, posinf=0.0)

    # Using a cut everywhere
    print(f"Using Everywhere a cut of {args.cut}")
    sample[sample < args.cut] = 0.0

    # get reference folder and name of file
    args.source_dir, args.reference_file_name = os.path.split(args.reference_file)
    args.reference_file_name = os.path.splitext(args.reference_file_name)[0]

    reference_file = h5py.File(args.reference_file, "r")
    check_file(reference_file, args, which="reference")

    reference_shower, reference_energy = extract_shower_and_energy(
        reference_file, which="reference", max_len=len(sample)
    )
    reference_shower[reference_shower < args.cut] = 0.0
    reference_hlf = HLF.HighLevelFeatures(particle, filename=cfg.data.xml_filename)
    reference_hlf.Einc = reference_energy

    args.x_scale = "log"

    if args.mode in ["all", "no-cls", "avg"]:
        print("Plotting average shower next to reference...")
        plot_layer_comparison(
            hlf,
            sample.mean(axis=0, keepdims=True),
            reference_hlf,
            reference_shower.mean(axis=0, keepdims=True),
            args,
        )
        print("Plotting average shower next to reference: DONE.\n")
        print("Plotting average shower...")
        hlf.DrawAverageShower(
            sample,
            filename=os.path.join(
                args.output_dir, f"average_shower_dataset_{args.dataset}.png"
            ),
            title="Shower average",
        )
        if hasattr(reference_hlf, "avg_shower"):
            pass
        else:
            reference_hlf.avg_shower = reference_shower.mean(axis=0, keepdims=True)
        hlf.DrawAverageShower(
            reference_hlf.avg_shower,
            filename=os.path.join(
                args.output_dir,
                f"reference_average_shower_dataset_{args.dataset}.png",
            ),
            title="Shower average reference dataset",
        )
        print("Plotting average shower: DONE.\n")

        print("Plotting randomly selected reference and generated shower: ")
        hlf.DrawSingleShower(
            sample[:5],
            filename=os.path.join(
                args.output_dir, f"single_shower_dataset_{args.dataset}.png"
            ),
            title="Single shower",
        )
        hlf.DrawSingleShower(
            reference_shower[:5],
            filename=os.path.join(
                args.output_dir,
                f"reference_single_shower_dataset_{args.dataset}.png",
            ),
            title="Reference single shower",
        )

    if args.mode in ["all", "no-cls", "avg-E"]:
        print("Plotting average showers for different energies ...")
        if "1" in args.dataset:
            target_energies = 2 ** np.linspace(8, 23, 16)
            plot_title = [
                f"shower average at E = {int(en)} MeV" for en in target_energies
            ]
        else:
            target_energies = 10 ** np.linspace(3, 6, 4)
            plot_title = []
            for i in range(3, 7):
                plot_title.append(
                    f"shower average for E in [{10**i}, {10 ** (i + 1)}] MeV"
                )
        for i in range(len(target_energies) - 1):
            filename = (
                f"average_shower_dataset_{args.dataset}_E_{target_energies[i]}.png"
            )
            which_showers = (
                (energy >= target_energies[i]) & (energy < target_energies[i + 1])
            ).squeeze()
            hlf.DrawAverageShower(
                sample[which_showers],
                filename=os.path.join(args.output_dir, filename),
                title=plot_title[i],
            )
            if hasattr(reference_hlf, "avg_shower_E"):
                pass
            else:
                reference_hlf.avg_shower_E = {}
            if target_energies[i] in reference_hlf.avg_shower_E:
                pass
            else:
                which_showers = (
                    (reference_hlf.Einc >= target_energies[i])
                    & (reference_hlf.Einc < target_energies[i + 1])
                ).squeeze()
                reference_hlf.avg_shower_E[target_energies[i]] = reference_shower[
                    which_showers
                ].mean(axis=0, keepdims=True)

            hlf.DrawAverageShower(
                reference_hlf.avg_shower_E[target_energies[i]],
                filename=os.path.join(args.output_dir, "reference_" + filename),
                title="reference " + plot_title[i],
            )

        print("Plotting average shower for different energies: DONE.\n")

    if args.mode in ["all", "no-cls", "hist-p", "hist-chi", "hist"]:
        print("Calculating high-level features for histograms ...")
        hlf.CalculateFeatures(sample)
        hlf.Einc = energy

        if reference_hlf.E_tot is None:
            reference_hlf.CalculateFeatures(reference_shower)

        print("Calculating high-level features for histograms: DONE.\n")

        if args.mode in ["all", "no-cls", "hist-chi", "hist"]:
            with open(
                os.path.join(args.output_dir, f"histogram_chi2_{args.dataset}.txt"),
                "w",
            ) as f:
                f.write(
                    "List of chi2 of the plotted histograms,"
                    + " see eq. 15 of 2009.03796 for its definition.\n"
                )
        print("Plotting histograms ...")
        if args.dataset == "1-photons":
            p_label = r"$\gamma$ ds-1"
        elif args.dataset == "1-pions":
            p_label = r"$\pi^{+}$ ds-1"
        elif args.dataset == "2":
            p_label = r"$e^{-}$ ds-2"
        elif args.dataset == "3":
            p_label = r"$e^{-}$ ds-3"
        else:
            p_label = f"{args.p_label}"

        plot_histograms(
            [
                hlf,
            ],
            reference_hlf,
            args,
            labels,
            [
                "",
            ],
            p_label=p_label,
        )
        plot_cell_dist(
            [
                sample,
            ],
            reference_shower,
            args,
            labels,
            [
                "",
            ],
            p_label,
        )
        print("Plotting histograms: DONE. \n")

    if args.mode in [
        "all",
        "all-cls",
        "cls-low",
        "cls-high",
        "cls-low-normed",
        "cls-resnet",
    ]:
        if args.mode in ["all", "all-cls"]:
            list_cls = ["cls-low", "cls-high", "cls-resnet"]
        else:
            list_cls = [args.mode]

        print("Calculating high-level features for classifier ...")

        print(f"Using {args.cut} as cut for the showers ...")
        # set a cut on low energy voxels !only low level!
        cut = args.cut

        hlf.CalculateFeatures(sample)
        hlf.Einc = energy

        if reference_hlf.E_tot is None:
            reference_hlf.CalculateFeatures(reference_shower)

        print("Calculating high-level features for classifer: DONE.\n")
        for key in list_cls:
            if (args.mode in ["cls-low", "cls-resnet"]) or (
                key in ["cls-low", "cls-resnet"]
            ):
                source_array = prepare_low_data_for_classifier(
                    sample, energy, hlf, 0.0, cut=cut, normed=False
                )
                reference_array = prepare_low_data_for_classifier(
                    reference_shower,
                    reference_energy,
                    reference_hlf,
                    1.0,
                    cut=cut,
                    normed=False,
                )
            elif (args.mode in ["cls-low-normed"]) or (key in ["cls_low_normed"]):
                source_array = prepare_low_data_for_classifier(
                    sample, energy, hlf, 0.0, cut=cut, normed=True
                )
                reference_array = prepare_low_data_for_classifier(
                    reference_shower,
                    reference_energy,
                    reference_hlf,
                    1.0,
                    cut=cut,
                    normed=True,
                )
            elif (args.mode in ["cls-high"]) or (key in ["cls-high"]):
                source_array = prepare_high_data_for_classifier(
                    sample, energy, hlf, 0.0, cut=cut
                )
                reference_array = prepare_high_data_for_classifier(
                    reference_shower, reference_energy, reference_hlf, 1.0, cut=cut
                )

            train_data, test_data, val_data = ttv_split(source_array, reference_array)

            # set up device
            args.device = torch.device(
                "cuda:" + str(args.which_cuda) if torch.cuda.is_available() else "cpu"
            )
            print(f"Using {args.device}")

            if key in ["all", "cls-low", "cls-low-normed", "cls-high"]:
                # set up DNN classifier
                input_dim = train_data.shape[1] - 1
                DNN_kwargs = {
                    "num_layer": args.cls_n_layer,
                    "num_hidden": args.cls_n_hidden,
                    "input_dim": input_dim,
                    "dropout_probability": args.cls_dropout_probability,
                }
                classifier = DNN(**DNN_kwargs)
            elif key in ["cls-resnet"]:
                classifier = generate_model(
                    args.cls_resnet_layers,
                    img_shape={
                        "2": (45, 16, 9),
                        "3": (45, 50, 18),
                        "LEMURS": (45, 16, 9),
                    }[args.dataset],
                )

            classifier.to(args.device)
            print(classifier)
            total_parameters = sum(
                p.numel() for p in classifier.parameters() if p.requires_grad
            )

            print(f"{args.mode} has {int(total_parameters)} parameters")

            if key == "cls-resnet":
                optimizer = torch.optim.AdamW(
                    classifier.parameters(), lr=args.cls_resnet_lr
                )  # usually smaller lr for resnet
            else:
                optimizer = torch.optim.Adam(classifier.parameters(), lr=args.cls_lr)

            if args.save_mem:
                train_data = TensorDataset(
                    torch.tensor(train_data, dtype=torch.get_default_dtype())
                )
                test_data = TensorDataset(
                    torch.tensor(test_data, dtype=torch.get_default_dtype())
                )
                val_data = TensorDataset(
                    torch.tensor(val_data, dtype=torch.get_default_dtype())
                )
            else:
                train_data = TensorDataset(
                    torch.tensor(train_data, dtype=torch.get_default_dtype()).to(
                        args.device
                    )
                )
                test_data = TensorDataset(
                    torch.tensor(test_data, dtype=torch.get_default_dtype()).to(
                        args.device
                    )
                )
                val_data = TensorDataset(
                    torch.tensor(val_data, dtype=torch.get_default_dtype()).to(
                        args.device
                    )
                )

            train_dataloader = DataLoader(
                train_data, batch_size=args.cls_batch_size, shuffle=True
            )
            test_dataloader = DataLoader(
                test_data, batch_size=args.cls_batch_size, shuffle=False
            )
            val_dataloader = DataLoader(
                val_data, batch_size=args.cls_batch_size, shuffle=False
            )

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
            print(f"{eval_auc:.4f} / {eval_JSD:.4f}")
            with open(
                os.path.join(
                    args.output_dir,
                    f"classifier_{args.mode}_{key}_{args.dataset}.txt",
                ),
                "a",
            ) as f:
                f.write(
                    "Final result of classifier test (AUC / JSD):\n"
                    + f"{eval_auc:.4f} / {eval_JSD:.4f}\n\n"
                )

            if args.mode in ["all", "fpd", "kpd"]:
                import jetnet

                print("Calculating high-level features for FPD/KPD ...")
                hlf.CalculateFeatures(sample)
                hlf.Einc = energy

                if reference_hlf.E_tot is None:
                    reference_hlf.CalculateFeatures(reference_shower)

                print("Calculating high-level features for FPD/KPD: DONE.\n")

                # get high level features and remove class label
                source_array = prepare_high_data_for_classifier(
                    sample, energy, hlf, 0.0, cut=cut
                )[:, :-1]
                reference_array = prepare_high_data_for_classifier(
                    reference_shower, reference_energy, reference_hlf, 1.0, cut=cut
                )[:, :-1]

                fpd_val, fpd_err = jetnet.evaluation.fpd(
                    reference_array, source_array, min_samples=10000
                )
                kpd_val, kpd_err = jetnet.evaluation.kpd(
                    reference_array, source_array, batch_size=10000
                )

                result_str = (
                    f"FPD (x10^3): {fpd_val*1e3:.4f} ± {fpd_err*1e3:.4f}\n"
                    f"KPD (x10^3): {kpd_val*1e3:.4f} ± {kpd_err*1e3:.4f}"
                )

                print(result_str)
                with open(
                    os.path.join(args.output_dir, f"fpd_kpd_{args.dataset}.txt"),
                    "w",
                ) as f:
                    f.write(result_str)

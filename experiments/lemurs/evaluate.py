# pylint: disable=invalid-name
"""
Evaluation script for the LEMURS dataset inspired by the CaloChallenge.


input:
    - path to a folder containing .hdf5 samples.
      The script loads only files with *samples.hdf5 in the name
output:
    - metrics for evaluation (plots, classifier scores, etc.)

usage:
    -i --input_file: path of the input files to be evaluated.
    -r --reference_file: Name and path of the reference .hdf5 file.
    -m --mode: Which metric to look at. Choices are
               'all': does all of the below (with low-level classifier).
               'avg': plots the average shower of the whole dataset.
               'avg-E': plots the average showers at different energy (ranges).
               'hist-p': plots histograms of high-level features.
               'hist-chi': computes the chi2 difference of the histograms.
               'hist': plots histograms and computes chi2.
               'all-cls': only run classifiers in list_cls
               'no-cls': does all of the above (no classifier).
               'cls-low': trains a classifier on low-level features (voxels).
               'cls-low-normed': trains a classifier on normalized voxels.
               'cls-high': trains a classifier on high-level features (same as histograms).
    -d --dataset: Which dataset the evaluation is for. Choices are
                  '1-photons', '1-pions', '2', '3'
       --output_dir: Folder in which the evaluation results (plots, scores) are saved.
       --save_mem: If included, data is moved to the GPU batch by batch instead of once.
                   This reduced the memory footprint a lot, especially for datasets 2 and 3.

       --no_cuda: if added, code will not run on GPU, even if available.
       --which_cuda: Which GPU to use if multiple are available.

additional options for the classifier start with --cls_ and can be found below.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import h5py
import torch
from torch.utils.data import TensorDataset, DataLoader

from experiments.calo_utils.ugr_evaluation.evaluate_plotting_helper import *
import experiments.calo_utils.ugr_evaluation.HighLevelFeatures as HLF
from experiments.calo_utils.ugr_evaluation.resnet import generate_model
from experiments.calo_utils.ugr_evaluation.evaluate import (
    DNN,
    prepare_low_data_for_classifier,
    prepare_high_data_for_classifier,
    ttv_split,
    load_classifier,
    train_and_evaluate_cls,
    evaluate_cls,
)
from experiments.logger import LOGGER

torch.set_default_dtype(torch.float64)

plt.rc("font", family="serif", size=16)
plt.rc("axes", titlesize="medium")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
plt.rc("text", usetex=True)

########## Parser Setup ##########


def define_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate calorimeter showers of the " + "Fast Calorimeter Challenge 2022."
        )
    )

    parser.add_argument(
        "--input_file", "-i", help="Path of the inputs file to be evaluated."
    )
    parser.add_argument(
        "--reference_file",
        "-r",
        help="Name and path of the .hdf5 file to be used as reference. ",
    )
    parser.add_argument(
        "--mode",
        "-m",
        default="all",
        choices=[
            "all",
            "all-cls",
            "no-cls",
            "avg",
            "avg-E",
            "hist-p",
            "hist-chi",
            "hist",
            "cls-low",
            "cls-low-normed",
            "cls-high",
        ],
        help=(
            "What metric to evaluate: "
            + "'avg' plots the shower average;"
            + "'avg-E' plots the shower average for energy ranges;"
            + "'hist-p' plots the histograms;"
            + "'hist-chi' evaluates a chi2 of the histograms;"
            + "'hist' evaluates a chi2 of the histograms and plots them;"
            + "'cls-low' trains a classifier on the low-level feautures;"
            + "'cls-low-normed' trains a classifier on the low-level feautures"
            + " with calorimeter layers normalized to 1;"
            + "'cls-high' trains a classifier on the high-level features;"
            + "'all' does the full evaluation, ie all of the above"
            + " with low-level classifier."
        ),
    )
    parser.add_argument(
        "--dataset",
        "-d",
        choices=["1-photons", "1-pions", "2", "3"],
        help="Which dataset is evaluated.",
    )
    parser.add_argument(
        "--output_dir",
        default="evaluation_results/",
        help="Where to store evaluation output files (plots and scores).",
    )

    parser.add_argument("--cut", type=float)
    parser.add_argument("--energy", type=float, default=None)

    parser.add_argument(
        "--cls_resnet_layers",
        type=int,
        default=18,
        help="Number of layers in the ResNet classifier, default is 18.",
    )
    parser.add_argument(
        "--cls_n_layer",
        type=int,
        default=2,
        help="Number of hidden layers in the classifier, default is 2.",
    )
    parser.add_argument(
        "--cls_n_hidden",
        type=int,
        default="512",
        help="Hidden nodes per layer of the classifier, default is 512.",
    )
    parser.add_argument(
        "--cls_dropout_probability",
        type=float,
        default=0.0,
        help="Dropout probability of the classifier, default is 0.",
    )

    parser.add_argument(
        "--cls_batch_size",
        type=int,
        default=1000,
        help="Classifier batch size, default is 1000.",
    )
    parser.add_argument(
        "--cls_n_epochs",
        type=int,
        default=50,
        help="Number of epochs to train classifier, default is 50.",
    )
    parser.add_argument(
        "--cls_lr",
        type=float,
        default=2e-4,
        help="Learning rate of the classifier, default is 2e-4.",
    )

    # CUDA parameters
    parser.add_argument("--no_cuda", action="store_true", help="Do not use cuda.")
    parser.add_argument(
        "--which_cuda", default=0, type=int, help="Which cuda device to use"
    )

    parser.add_argument(
        "--save_mem",
        action="store_true",
        help="Data is moved to GPU batch by batch instead of once in total.",
    )
    return parser


def extract_shower_and_energy(given_file, which, max_len=-1):
    """reads .hdf5 file and returns samples and their energy"""
    print("Extracting showers from {} file ...".format(which))
    shower = given_file["showers"][:max_len]
    energy = given_file["incident_energy"][:max_len]
    theta = given_file["incident_theta"][:max_len]
    phi = given_file["incident_phi"][:max_len]
    print("Extracting showers from {} file: DONE.\n".format(which))
    return (
        shower.astype("float32", copy=False),
        energy.astype("float32", copy=False),
        theta.astype("float32", copy=False),
        phi.astype("float32", copy=False),
    )


def plot_histograms(hlf_classes, reference_class, arg, input_names="", p_label=""):
    """plots histograms based with reference file as comparison"""
    plot_Etot_Einc_scaled(
        hlf_classes, reference_class, arg, arg.labels, input_names, p_label
    )
    plot_E_layers(hlf_classes, reference_class, arg, arg.labels, input_names, p_label)
    plot_ECEtas(hlf_classes, reference_class, arg, arg.labels, input_names, p_label)
    plot_ECPhis(hlf_classes, reference_class, arg, arg.labels, input_names, p_label)
    plot_ECWidthEtas(
        hlf_classes, reference_class, arg, arg.labels, input_names, p_label
    )
    plot_ECWidthPhis(
        hlf_classes, reference_class, arg, arg.labels, input_names, p_label
    )
    plot_sparsity(hlf_classes, reference_class, arg, arg.labels, input_names, p_label)
    plot_weighted_depth_a(
        hlf_classes, reference_class, arg, arg.labels, input_names, p_label
    )
    plot_weighted_depth_r(
        hlf_classes, reference_class, arg, arg.labels, input_names, p_label
    )
    plot_z_profile(hlf_classes, reference_class, arg, arg.labels, input_names, p_label)
    plot_r_profile(hlf_classes, reference_class, arg, arg.labels, input_names, p_label)


def plot_conditions(sample_conds, ref_conds, arg, labels, input_names, p_label):
    filename = os.path.join(arg.output_dir, "conditions.pdf")
    with PdfPages(filename) as pdf:
        for n in range(sample_conds.shape[1]):
            fig, ax = plt.subplots(
                3,
                1,
                figsize=(4.5, 4),
                gridspec_kw={"height_ratios": (4, 1, 1), "hspace": 0.0},
                sharex=True,
            )
            combined = np.concatenate((sample_conds[:, n], ref_conds[:, n]))
            data_min = combined.min()
            data_max = combined.max()
            bins = np.linspace(data_min - 1, data_max + 1, 41)

            counts_ref, bins = np.histogram(ref_conds[:, n], bins=bins, density=False)
            bin_width = bins[1] - bins[0]
            counts_ref_norm = counts_ref / counts_ref.sum()
            geant_error = counts_ref_norm / np.sqrt(counts_ref)
            geant_ratio_error = geant_error / counts_ref_norm
            geant_ratio_error_isnan = np.isnan(geant_ratio_error)
            geant_ratio_error[geant_ratio_error_isnan] = 0.0
            geant_delta_err = geant_ratio_error * 100
            ax[0].step(
                bins,
                dup(counts_ref_norm),
                label="Geant4",
                linestyle="-",
                alpha=0.8,
                linewidth=1.0,
                color="k",
                where="post",
            )
            ax[0].fill_between(
                bins,
                dup(counts_ref_norm + geant_error),
                dup(counts_ref_norm - geant_error),
                step="post",
                color="k",
                alpha=0.2,
            )
            ax[1].fill_between(
                bins,
                dup(1 - geant_ratio_error),
                dup(1 + geant_ratio_error),
                step="post",
                color="k",
                alpha=0.2,
            )
            ax[2].errorbar(
                (bins[:-1] + bins[1:]) / 2,
                np.zeros_like(bins[:-1]),
                yerr=geant_delta_err,
                ecolor="grey",
                color="grey",
                elinewidth=0.5,
                linewidth=1.0,
                fmt=".",
                capsize=2,
            )
            counts, _ = np.histogram(sample_conds[:, n], bins=bins, density=False)
            counts_data, bins = np.histogram(
                sample_conds[:, n], bins=bins, density=False
            )
            counts_data_norm = counts_data / counts_data.sum()
            ax[0].step(
                bins,
                dup(counts_data_norm),
                label=labels[0],
                where="post",
                linewidth=1.0,
                alpha=1.0,
                color="tab:blue",
                linestyle="-",
            )
            y_ref_err = counts_data_norm / np.sqrt(counts)
            ax[0].fill_between(
                bins,
                dup(counts_data_norm + y_ref_err),
                dup(counts_data_norm - y_ref_err),
                step="post",
                color="tab:blue",
                alpha=0.2,
            )

            ratio = counts_data / counts_ref
            ratio_err = y_ref_err / counts_ref_norm
            ratio_isnan = np.isnan(ratio)
            ratio[ratio_isnan] = 1.0
            ratio_err[ratio_isnan] = 0.0
            ax[1].step(
                bins,
                dup(ratio),
                linewidth=1.0,
                alpha=1.0,
                color="tab:blue",
                where="post",
            )
            ax[1].fill_between(
                bins,
                dup(ratio - ratio_err),
                dup(ratio + ratio_err),
                step="post",
                color="tab:blue",
                alpha=0.2,
            )

            delta = np.fabs(ratio - 1) * 100
            delta_err = ratio_err * 100
            markers, caps, bars = ax[2].errorbar(
                (bins[:-1] + bins[1:]) / 2,
                delta,
                yerr=delta_err,
                ecolor="tab:blue",
                color="tab:blue",
                elinewidth=0.5,
                linewidth=1.0,
                fmt=".",
                capsize=2,
            )

            ax[1].hlines(
                1.0,
                bins[0],
                bins[-1],
                linewidth=1.0,
                alpha=0.8,
                linestyle="-",
                color="k",
            )
            ax[1].set_yticks((0.7, 1.0, 1.3))
            ax[1].set_ylim(0.5, 1.5)
            ax[0].set_xlim(bins[0] - bin_width, bins[-1] + bin_width)

            ax[1].axhline(0.7, c="k", ls="--", lw=0.5)
            ax[1].axhline(1.3, c="k", ls="--", lw=0.5)

            ax[2].set_ylim((0.05, 50))
            ax[2].set_yscale("log")
            ax[2].set_yticks([0.1, 1.0, 10.0])
            ax[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
            ax[2].set_yticks(
                [
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    20.0,
                    30.0,
                    40.0,
                ],
                minor=True,
            )

            ax[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
            # ax[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
            ax[2].set_ylabel(r"$\delta [\%]$")

            # ax[0].set_title("Energy deposited in layer {}".format(key))
            ax[0].set_ylabel(r"a.u.")
            ax[1].set_ylabel(r"$\frac{\text{Model}}{\text{Geant4}}$")
            ax[2].set_xlabel(f"cond {n}")
            ax[0].set_yscale("log")
            ax[0].legend(
                loc="lower right",
                frameon=False,
                title=p_label,
                handlelength=1.2,
                fontsize=16,
                title_fontsize=18,
            )
            fig.tight_layout(
                pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98)
            )
            plt.savefig(pdf, dpi=300, format="pdf")
            plt.close()


########## Alternative Main ############


class args_class:
    def __init__(self, cfg):
        cfg = cfg.evaluation
        self.dataset = cfg.eval_dataset
        self.mode = cfg.eval_mode
        self.cut = cfg.eval_cut
        self.reference_file = cfg.eval_hdf5_file
        self.labels = cfg.eval_labels
        self.which_cuda = 0

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


def run_from_py(sample, energy, theta, phi, cfg):
    LOGGER.info("Running evaluation script:")

    if not os.path.isdir(cfg.run_dir + f"/eval_{cfg.run_idx}/"):
        os.makedirs(cfg.run_dir + f"/eval_{cfg.run_idx}/")

    args = args_class(cfg)
    args.output_dir = cfg.run_dir + f"/eval_{cfg.run_idx}/"
    print("Input sample of shape: ")
    print(sample.shape)

    args.min_energy = 0.5e-3 / 0.033
    particle = "electron"
    args.particle = particle
    args.num_voxels = np.prod(sample.shape[1:])
    args.dataset = "LEMURS"

    hlf = HLF.HighLevelFeatures(particle, filename=cfg.data.xml_filename)

    # match the CaloChallenge convention
    sample = sample.transpose(0, 3, 2, 1)
    sample = sample.reshape(-1, args.num_voxels)
    # Checking for negative values, nans and infinities
    print("Checking for negative values, number of negative energies: ")
    print("input: ", (sample < 0.0).sum(), "\n")
    print("Checking for nans in the generated sample, number of nans: ")
    print("input: ", np.isnan(sample).sum(), "\n")
    print("Checking for infs in the generated sample, number of infs: ")
    print("input: ", np.isinf(sample).sum(), "\n")
    np.nan_to_num(sample, copy=False, nan=0.0, neginf=0.0, posinf=0.0)

    # Using a cut everywhere
    print("Using Everywhere a cut of {}".format(args.cut))
    sample[sample < args.cut] = 0.0
    sample_conds = np.concatenate((energy, theta, phi), axis=1)

    # get reference folder and name of file
    args.source_dir, args.reference_file_name = os.path.split(args.reference_file)
    args.reference_file_name = os.path.splitext(args.reference_file_name)[0]

    reference_file = h5py.File(args.reference_file, "r")
    reference_file = reference_file["events"][:]

    reference_shower, reference_energy, reference_theta, reference_phi = (
        extract_shower_and_energy(
            reference_file, which="reference", max_len=len(sample)
        )
    )
    # match the CaloChallenge convention
    reference_shower = reference_shower.transpose(0, 3, 2, 1)
    reference_shower = reference_shower.reshape(-1, args.num_voxels)

    reference_shower[reference_shower < args.cut] = 0.0
    reference_hlf = HLF.HighLevelFeatures(particle, filename=cfg.data.xml_filename)
    reference_hlf.Einc = reference_energy
    reference_conds = np.concatenate(
        (reference_energy, reference_theta, reference_phi), axis=1
    )

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
                args.output_dir, "average_shower_dataset_{}.png".format(args.dataset)
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
                "reference_average_shower_dataset_{}.png".format(args.dataset),
            ),
            title="Shower average reference dataset",
        )
        print("Plotting average shower: DONE.\n")

        print("Plotting randomly selected reference and generated shower: ")
        hlf.DrawSingleShower(
            sample[:5],
            filename=os.path.join(
                args.output_dir, "single_shower_dataset_{}.png".format(args.dataset)
            ),
            title="Single shower",
        )
        hlf.DrawSingleShower(
            reference_shower[:5],
            filename=os.path.join(
                args.output_dir,
                "reference_single_shower_dataset_{}.png".format(args.dataset),
            ),
            title="Reference single shower",
        )

    if args.mode in ["all", "no-cls", "avg-E"]:
        print("Plotting average showers for different energies ...")
        target_energies = 10 ** np.linspace(3, 6, 4)
        plot_title = []
        for i in range(3, 7):
            plot_title.append(
                "shower average for E in [{}, {}] MeV".format(10**i, 10 ** (i + 1))
            )
        for i in range(len(target_energies) - 1):
            filename = "average_shower_dataset_{}_E_{}.png".format(
                args.dataset, target_energies[i]
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
                os.path.join(
                    args.output_dir, "histogram_chi2_{}.txt".format(args.dataset)
                ),
                "w",
            ) as f:
                f.write(
                    "List of chi2 of the plotted histograms,"
                    + " see eq. 15 of 2009.03796 for its definition.\n"
                )
        print("Plotting histograms ...")
        p_label = "LEMURS"

        plot_histograms(
            [
                hlf,
            ],
            reference_hlf,
            args,
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
            args.labels,
            [
                "",
            ],
            p_label,
        )
        plot_conditions(sample_conds, reference_conds, args, args.labels, [""], p_label)
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

        print("Using {} as cut for the showers ...".format(args.cut))
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
            print("Using {}".format(args.device))

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
                    img_shape=(45, 16, 9),
                )

            classifier.to(args.device)
            print(classifier)
            total_parameters = sum(
                p.numel() for p in classifier.parameters() if p.requires_grad
            )

            LOGGER.info("{} has {} parameters".format(args.mode, int(total_parameters)))

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
            LOGGER.info("Final result of classifier test (AUC / JSD):")
            LOGGER.info("{:.4f} / {:.4f}".format(eval_auc, eval_JSD))
            with open(
                os.path.join(
                    args.output_dir,
                    "classifier_{}_{}_{}.txt".format(args.mode, key, args.dataset),
                ),
                "a",
            ) as f:
                f.write(
                    "Final result of classifier test (AUC / JSD):\n"
                    + "{:.4f} / {:.4f}\n\n".format(eval_auc, eval_JSD)
                )

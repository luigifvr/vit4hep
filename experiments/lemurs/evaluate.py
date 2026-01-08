import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.data import DataLoader, TensorDataset

import experiments.calo_utils.ugr_evaluation.HighLevelFeatures as HLF
from experiments.calo_utils.ugr_evaluation.evaluate import (
    DNN,
    evaluate_cls,
    load_classifier,
    train_and_evaluate_cls,
    ttv_split,
)
from experiments.calo_utils.ugr_evaluation.evaluate_plotting_helper import (
    dup,
    plot_cell_dist,
    plot_E_layers,
    plot_ECEtas,
    plot_ECPhis,
    plot_ECWidthEtas,
    plot_ECWidthPhis,
    plot_Etot_Einc_scaled,
    plot_layer_comparison,
    plot_r_profile,
    plot_sparsity,
    plot_weighted_depth_a,
    plot_weighted_depth_r,
    plot_z_profile,
)
from experiments.calo_utils.ugr_evaluation.resnet import generate_model
from experiments.lemurs.utils import (
    prepare_high_data_for_classifier,
    prepare_low_data_for_classifier,
)
from experiments.logger import LOGGER

torch.set_default_dtype(torch.float64)

plt.rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 24})
plt.rc("axes", titlesize="medium")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
plt.rc("text", usetex=True)


def extract_shower_and_energy(
    given_file, which, max_len=-1, energy_bin=None, theta_bin=None, phi_bin=None
):
    """reads .hdf5 file and returns samples and their energy"""
    print(f"Extracting showers from {which} file ...")
    shower = given_file["showers"][:]
    energy = given_file["incident_energy"][:]
    theta = given_file["incident_theta"][:]
    phi = given_file["incident_phi"][:]
    print(f"Extracting showers from {which} file: DONE.\n")
    if energy_bin is not None:
        energy_mask = (energy >= energy_bin[0]) & (energy < energy_bin[1])
    else:
        energy_mask = np.ones_like(energy, dtype=bool)
    if theta_bin is not None:
        theta_mask = (theta >= theta_bin[0]) & (theta < theta_bin[1])
    else:
        theta_mask = np.ones_like(energy, dtype=bool)
    if phi_bin is not None:
        phi_mask = (phi >= phi_bin[0]) & (phi < phi_bin[1])
    else:
        phi_mask = np.ones_like(energy, dtype=bool)
    full_mask = np.ones_like(energy, dtype=bool)
    full_mask = (full_mask & energy_mask & theta_mask & phi_mask).squeeze()
    shower = shower[full_mask][:max_len]
    energy = energy[full_mask][:max_len]
    theta = theta[full_mask][:max_len]
    phi = phi[full_mask][:max_len]
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
            ax[2].set_ylabel(r"$\delta [\%]$")
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
        self.p_label = cfg.eval_p_label
        self.which_cuda = 0

        # slice parameters
        self.energy_bin = cfg.eval_energy_bin
        self.theta_bin = cfg.eval_theta_bin
        self.phi_bin = cfg.eval_phi_bin

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
    print(f"Using Everywhere a cut of {args.cut}")
    sample[sample < args.cut] = 0.0
    sample_conds = np.concatenate((energy, theta, phi), axis=1)

    # get reference folder and name of file
    args.source_dir, args.reference_file_name = os.path.split(args.reference_file)
    args.reference_file_name = os.path.splitext(args.reference_file_name)[0]

    reference_file = h5py.File(args.reference_file, "r")
    reference_file = reference_file["events"][:]

    print("Extracting showers from reference file ...")
    print(
        f"slicing with energy bin: {args.energy_bin}, theta bin: {args.theta_bin}, phi bin: {args.phi_bin}"
    )
    reference_shower, reference_energy, reference_theta, reference_phi = (
        extract_shower_and_energy(
            reference_file,
            which="reference",
            max_len=len(sample),
            energy_bin=args.energy_bin,
            theta_bin=args.theta_bin,
            phi_bin=args.phi_bin,
        )
    )
    print("Number of showers in reference after slicing: ", len(reference_energy))
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
        target_energies = 10 ** np.linspace(3, 6, 4)
        plot_title = []
        for i in range(3, 7):
            plot_title.append(f"shower average for E in [{10**i}, {10 ** (i + 1)}] MeV")
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
        p_label = f"{args.p_label}"

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

    if args.mode in ["all", "fpd", "kpd"]:
        import jetnet

        print("Calculating high-level features for FPD/KPD ...")
        hlf.CalculateFeatures(sample)
        hlf.Einc = energy

        reference_angles = np.concatenate((reference_theta, reference_phi), axis=1)
        angles = np.concatenate((theta, phi), axis=1)
        cut = args.cut

        if reference_hlf.E_tot is None:
            reference_hlf.CalculateFeatures(reference_shower)

        print("Calculating high-level features for FPD/KPD: DONE.\n")

        # get high level features and remove class label
        source_array = prepare_high_data_for_classifier(
            sample, energy, angles, hlf, 0.0, cut=cut
        )
        reference_array = prepare_high_data_for_classifier(
            reference_shower,
            reference_energy,
            reference_angles,
            reference_hlf,
            1.0,
            cut=cut,
        )

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
            os.path.join(args.output_dir, f"fpd_kpd_{args.dataset}.txt"), "w"
        ) as f:
            f.write(result_str)

    if args.mode in [
        "all",
        "all-cls",
        "cls-low",
        "cls-high",
        "cls-low-normed",
        "cls-resnet",
    ]:
        # TODO: angles are not currenlty used!!!
        reference_angles = np.concatenate((reference_theta, reference_phi), axis=1)
        angles = np.concatenate((theta, phi), axis=1)
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
                    sample, energy, angles, hlf, 0.0, cut=cut, normed=False
                )
                reference_array = prepare_low_data_for_classifier(
                    reference_shower,
                    reference_energy,
                    reference_angles,
                    reference_hlf,
                    1.0,
                    cut=cut,
                    normed=False,
                )
            elif (args.mode in ["cls-low-normed"]) or (key in ["cls_low_normed"]):
                source_array = prepare_low_data_for_classifier(
                    sample, energy, angles, hlf, 0.0, cut=cut, normed=True
                )
                reference_array = prepare_low_data_for_classifier(
                    reference_shower,
                    reference_energy,
                    reference_angles,
                    reference_hlf,
                    1.0,
                    cut=cut,
                    normed=True,
                )
            elif (args.mode in ["cls-high"]) or (key in ["cls-high"]):
                source_array = prepare_high_data_for_classifier(
                    sample, energy, angles, hlf, 0.0, cut=cut
                )
                reference_array = prepare_high_data_for_classifier(
                    reference_shower,
                    reference_energy,
                    reference_angles,
                    reference_hlf,
                    1.0,
                    cut=cut,
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
                    img_shape=(45, 16, 9),
                )

            classifier.to(args.device)
            print(classifier)
            total_parameters = sum(
                p.numel() for p in classifier.parameters() if p.requires_grad
            )

            LOGGER.info(f"{args.mode} has {int(total_parameters)} parameters")

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
            LOGGER.info(f"{eval_auc:.4f} / {eval_JSD:.4f}")
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

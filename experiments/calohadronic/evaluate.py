import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.calo_utils.ugr_evaluation.evaluate import (
    DNN,
    evaluate_cls,
    load_classifier,
    train_and_evaluate_cls,
    ttv_split,
)
from experiments.calo_utils.ugr_evaluation.evaluate_plotting_helper import _separation_power, dup
from experiments.calohadronic.utils import load_data
from experiments.logger import LOGGER

torch.set_default_dtype(torch.float64)

plt.rc("font", family="serif", size=16)
plt.rc("axes", titlesize="medium")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
plt.rc("text", usetex=True)

colors = ["#0000cc", "#b40000"]


def plot_histograms(features_gen, features_g4, all_gen, all_g4, arg, output_dir=""):
    """plot histograms specific for CaloHadronic"""

    plot_feature(
        features_gen[:, 0],
        features_g4[:, 0],
        arg,
        title="cog_x",
        label=r"$\langle x \rangle$ [mm]",
        title_label="CaloHad.",
        output_dir=output_dir,
    )
    plot_feature(
        features_gen[:, 1],
        features_g4[:, 1],
        arg,
        title="cog_y",
        label=r"$\langle y \rangle$ [mm]",
        title_label="CaloHad.",
        output_dir=output_dir,
    )
    plot_feature(
        features_gen[:, 2],
        features_g4[:, 2],
        arg,
        title="cog_z",
        label=r"$\langle z \rangle$ layer number",
        title_label="CaloHad.",
        output_dir=output_dir,
    )
    plot_feature(
        features_gen[:, 3],
        features_g4[:, 3],
        arg,
        title="energy",
        label=r"$E_\text{tot}/E_\text{inc}$",
        title_label="CaloHad.",
        output_dir=output_dir,
    )
    plot_feature(
        features_gen[:, 4],
        features_g4[:, 4],
        arg,
        title="nhits",
        label=r"$\langle \lambda \rangle$",
        title_label="CaloHad.",
        output_dir=output_dir,
    )
    plot_feature(
        np.log10(all_gen.flatten()),
        np.log10(all_g4.flatten()),
        arg,
        title="voxels",
        label=r"$x$ [GeV]",
        title_label="CaloHad.",
        output_dir=output_dir,
    )


def get_centroid_z(ecal, hcal):
    ecal_avg = ecal.mean((-1, -2))
    hcal_avg = hcal.mean((-1, -2))
    showers = np.concatenate((ecal_avg, hcal_avg), axis=1)
    x_var = np.arange(0, showers.shape[1])[None, :]
    centroid = (x_var * showers).sum(1) / showers.sum(-1)
    return centroid


def get_centroid_x(ecal, hcal, ecalmm=5.1):
    ecal_mm = ecalmm
    hcal_mm = 30
    ecal_avg = ecal.mean((-1, -3))
    hcal_avg = hcal.mean((-1, -3))
    x_pos_ecal = (np.arange(0, ecal.shape[2]) * ecal_mm)[None, :]
    x_pos_hcal = (np.arange(0, hcal.shape[2]) * hcal_mm)[None, :]
    ecal_center = ecal_avg * x_pos_ecal
    hcal_center = hcal_avg * x_pos_hcal
    showers_weighted = np.concatenate((ecal_center, hcal_center), axis=1)
    showers_sum = np.concatenate((ecal_avg, hcal_avg), axis=1)
    centroid = (showers_weighted).sum(1) / showers_sum.sum(-1)
    return centroid - 430.0  # shift to center around 0


def get_centroid_y(ecal, hcal, ecalmm=5.1):
    ecal_mm = ecalmm
    hcal_mm = 30
    ecal_avg = ecal.mean((-2, -3))
    hcal_avg = hcal.mean((-2, -3))
    x_pos_ecal = (np.arange(0, ecal.shape[3]) * ecal_mm)[None, :]
    x_pos_hcal = (np.arange(0, hcal.shape[3]) * hcal_mm)[None, :]
    ecal_center = ecal_avg * x_pos_ecal
    hcal_center = hcal_avg * x_pos_hcal
    showers_weighted = np.concatenate((ecal_center, hcal_center), axis=1)
    showers_sum = np.concatenate((ecal_avg, hcal_avg), axis=1)
    centroid = (showers_weighted).sum(1) / showers_sum.sum(-1)
    return centroid - 430.0  # shift to center around 0


def get_total_energy(ecal, hcal):
    ecal_total = ecal.sum((-1, -2, -3))
    hcal_total = hcal.sum((-1, -2, -3))
    total_energy = ecal_total + hcal_total
    return total_energy


def get_n_hits(ecal, hcal, threshold=1.0e-6):
    ecal_hits = (ecal > threshold).sum((-1, -2, -3))
    hcal_hits = (hcal > threshold).sum((-1, -2, -3))
    total_hits = ecal_hits + hcal_hits
    return total_hits


class args_class:
    """
    Create a class with evaluation arguments.
    """

    def __init__(self, cfg):
        cfg = cfg.evaluation
        self.dataset = cfg.eval_dataset
        self.mode = cfg.eval_mode
        self.cut = cfg.eval_cut
        self.cls_n_epochs = cfg.eval_cls_n_epochs
        self.save_mem = cfg.eval_cls_save_mem


def run_from_py(ecal, hcal, energy, cfg):
    """Main function to be called from other scripts."""
    LOGGER.info("Running evaluation script:")
    if not os.path.isdir(cfg.run_dir + f"/eval_{cfg.run_idx}/"):
        os.makedirs(cfg.run_dir + f"/eval_{cfg.run_idx}/")

    output_dir = cfg.run_dir + f"/eval_{cfg.run_idx}/"
    args = args_class(cfg)
    args.output_dir = output_dir
    print("Input ecal sample of shape: ")
    print(ecal.shape)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    test_data_dir = cfg.evaluation.eval_hdf5_file
    test_file = h5py.File(test_data_dir, "r")
    test_data_dict = load_data(test_file, local_index=None, dtype=torch.float32)
    avg_pool = torch.nn.AvgPool3d((3, 12, 12))
    ecal_g4, hcal_g4, energy_g4 = (
        (avg_pool(test_data_dict["ecal"]) * 3 * 12 * 12).numpy(),
        test_data_dict["hcal"].numpy(),
        test_data_dict["energy"].numpy(),
    )

    min_energy = 1.0e-6
    ecal[ecal < min_energy] = 0.0
    hcal[hcal < min_energy] = 0.0
    ecal_g4[ecal_g4 < min_energy] = 0.0
    hcal_g4[hcal_g4 < min_energy] = 0.0

    cog_x_gen = get_centroid_x(ecal, hcal, ecalmm=5.1 * 12)
    cog_y_gen = get_centroid_y(ecal, hcal, ecalmm=5.1 * 12)
    cog_z_gen = get_centroid_z(ecal, hcal)
    energy_gen = get_total_energy(ecal, hcal) / energy.flatten()
    n_hits_gen = get_n_hits(ecal, hcal, threshold=min_energy)
    all_voxels_gen = np.concatenate((ecal.flatten(), hcal.flatten()), axis=0)

    cog_x_g4 = get_centroid_x(ecal_g4, hcal_g4, ecalmm=5.1 * 12)
    cog_y_g4 = get_centroid_y(ecal_g4, hcal_g4, ecalmm=5.1 * 12)
    cog_z_g4 = get_centroid_z(ecal_g4, hcal_g4)
    energy_g4 = get_total_energy(ecal_g4, hcal_g4) / energy_g4.flatten()
    n_hits_g4 = get_n_hits(ecal_g4, hcal_g4, threshold=min_energy)
    all_voxels_g4 = np.concatenate((ecal_g4.flatten(), hcal_g4.flatten()), axis=0)

    features_gen = np.stack((cog_x_gen, cog_y_gen, cog_z_gen, energy_gen, n_hits_gen), axis=1)
    features_g4 = np.stack((cog_x_g4, cog_y_g4, cog_z_g4, energy_g4, n_hits_g4), axis=1)
    plot_histograms(
        features_gen,
        features_g4,
        all_voxels_gen,
        all_voxels_g4,
        cfg,
        output_dir=output_dir,
    )

    cog_x_gen_std = (cog_x_gen - cog_x_gen.mean()) / cog_x_gen.std()
    cog_y_gen_std = (cog_y_gen - cog_y_gen.mean()) / cog_y_gen.std()
    cog_z_gen_std = (cog_z_gen - cog_z_gen.mean()) / cog_z_gen.std()
    energy_gen_std = (energy_gen - energy_gen.mean()) / energy_gen.std()
    n_hits_gen_std = (n_hits_gen - n_hits_gen.mean()) / n_hits_gen.std()

    cog_x_g4_std = (cog_x_g4 - cog_x_gen.mean()) / cog_x_gen.std()
    cog_y_g4_std = (cog_y_g4 - cog_y_gen.mean()) / cog_y_gen.std()
    cog_z_g4_std = (cog_z_g4 - cog_z_gen.mean()) / cog_z_gen.std()
    energy_g4_std = (energy_g4 - energy_gen.mean()) / energy_gen.std()
    n_hits_g4_std = (n_hits_g4 - n_hits_gen.mean()) / n_hits_gen.std()

    # extract layer energies
    ecal_energies = ecal.sum(axis=(-1, -2))
    hcal_energies = hcal.sum(axis=(-1, -2))
    ecal_energies_g4 = ecal_g4.sum(axis=(-1, -2))
    hcal_energies_g4 = hcal_g4.sum(axis=(-1, -2))

    g4_labels = np.zeros(features_g4.shape[0])
    gen_labels = np.ones(features_gen.shape[0])
    features_gen = np.concatenate(
        (
            cog_x_gen_std[:, None],
            cog_y_gen_std[:, None],
            cog_z_gen_std[:, None],
            energy_gen_std[:, None],
            n_hits_gen_std[:, None],
            ecal_energies,
            hcal_energies,
            gen_labels[:, None],
        ),
        axis=1,
    )
    features_g4 = np.concatenate(
        (
            cog_x_g4_std[:, None],
            cog_y_g4_std[:, None],
            cog_z_g4_std[:, None],
            energy_g4_std[:, None],
            n_hits_g4_std[:, None],
            ecal_energies_g4,
            hcal_energies_g4,
            g4_labels[:, None],
        ),
        axis=1,
    )
    train_data, test_data, val_data = ttv_split(features_gen, features_g4)
    # set up DNN classifier
    input_dim = train_data.shape[1] - 1
    DNN_kwargs = {
        "num_layer": cfg.evaluation.eval_cls_n_layer,
        "num_hidden": cfg.evaluation.eval_cls_n_hidden,
        "input_dim": input_dim,
        "dropout_probability": cfg.evaluation.eval_cls_dropout,
    }
    classifier = DNN(**DNN_kwargs)
    classifier.to(device)
    print(classifier)
    total_parameters = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

    print(f"Classifier has {int(total_parameters)} parameters")
    train_data = TensorDataset(torch.tensor(train_data, dtype=torch.get_default_dtype()).to(device))
    test_data = TensorDataset(torch.tensor(test_data, dtype=torch.get_default_dtype()).to(device))
    val_data = TensorDataset(torch.tensor(val_data, dtype=torch.get_default_dtype()).to(device))
    train_dataloader = DataLoader(
        train_data, batch_size=cfg.evaluation.eval_cls_batch_size, shuffle=True
    )
    test_dataloader = DataLoader(
        test_data, batch_size=cfg.evaluation.eval_cls_batch_size, shuffle=False
    )
    val_dataloader = DataLoader(
        val_data, batch_size=cfg.evaluation.eval_cls_batch_size, shuffle=False
    )
    optimizer = torch.optim.Adam(classifier.parameters(), lr=cfg.evaluation.eval_cls_lr)

    train_and_evaluate_cls(classifier, train_dataloader, test_dataloader, optimizer, args)
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
            output_dir,
            "classifier.txt",
        ),
        "a",
    ) as f:
        f.write(
            "Final result of classifier test (AUC / JSD):\n"
            + f"{eval_auc:.4f} / {eval_JSD:.4f}\n\n"
        )


def plot_feature(feature_gen, feature_g4, arg, title="", label="", title_label="", output_dir=""):
    gen_label = arg.evaluation.label
    g4_min = np.nanmin(feature_g4[np.isfinite(feature_g4)])
    g4_max = np.nanmax(feature_g4[np.isfinite(feature_g4)])
    bins = np.linspace(g4_min, g4_max, 41)
    fig, ax = plt.subplots(
        3,
        1,
        figsize=(4.5, 4),
        gridspec_kw={"height_ratios": (4, 1, 1), "hspace": 0.0},
        sharex=True,
    )

    counts_ref, bins = np.histogram(
        feature_g4,
        bins=bins,
        density=False,
    )
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
        dup(1 - geant_error / counts_ref_norm),
        dup(1 + geant_error / counts_ref_norm),
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
    counts, _ = np.histogram(feature_gen, bins=bins, density=False)
    counts_data, bins = np.histogram(feature_gen, bins=bins, density=False)
    counts_data_norm = counts_data / counts_data.sum()
    ax[0].step(
        bins,
        dup(counts_data_norm),
        label=gen_label,
        where="post",
        linewidth=1.0,
        alpha=1.0,
        color=colors[0],
        linestyle="-",
    )

    y_ref_err = counts_data_norm / np.sqrt(counts)
    ax[0].fill_between(
        bins,
        dup(counts_data_norm + y_ref_err),
        dup(counts_data_norm - y_ref_err),
        step="post",
        color=colors[0],
        alpha=0.2,
    )

    ratio = counts_data_norm / counts_ref_norm
    ratio_err = y_ref_err / counts_ref_norm
    ratio_isnan = np.isnan(ratio)
    ratio[ratio_isnan] = 1.0
    ratio_err[ratio_isnan] = 0.0
    ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=colors[0], where="post")
    ax[1].fill_between(
        bins,
        dup(ratio - ratio_err),
        dup(ratio + ratio_err),
        step="post",
        color=colors[0],
        alpha=0.2,
    )
    delta = np.fabs(ratio - 1) * 100
    delta_err = ratio_err * 100
    markers, caps, bars = ax[2].errorbar(
        (bins[:-1] + bins[1:]) / 2,
        delta,
        yerr=delta_err,
        ecolor=colors[0],
        color=colors[0],
        elinewidth=0.5,
        linewidth=1.0,
        fmt=".",
        capsize=2,
    )

    seps = _separation_power(counts_ref_norm, counts_data_norm, None)
    print(f"Separation power of {title} histogram: {seps}")
    with open(
        os.path.join(
            output_dir,
            "histogram_chi2.txt",
        ),
        "a",
    ) as f:
        f.write(f"{title}: \n")
        f.write(str(seps))
        f.write("\n\n")

    ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle="-", color="k")
    ax[1].set_yticks((0.7, 1.0, 1.3))
    ax[1].set_ylim(0.5, 1.5)
    ax[0].set_xlim(bins[0], bins[-1])

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

    ax[2].set_xlabel(f"{label}")
    ax[0].set_ylabel(r"a.u.")
    ax[1].set_ylabel(r"$\frac{\text{Model}}{\text{Geant4}}$")
    ax[0].legend(
        loc="best",
        frameon=False,
        title=title_label,
        handlelength=1.2,
        fontsize=16,
        title_fontsize=18,
    )
    fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
    filename = os.path.join(output_dir, f"{title}.pdf")
    fig.savefig(filename, dpi=300, format="pdf")
    plt.close()

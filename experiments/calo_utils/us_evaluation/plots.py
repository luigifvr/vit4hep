import os
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


def plot_ui_dists(
    gen_us,
    ref_us,
    xlim=(-0.05, 1.05),
    ratio_ylim=(0.5, 1.5),
    num_bins=64,
    color="#0000cc",
    quantile_bins=False,
    skip_quantiles=0,
    cfg=None,
):
    """
    Function that plots the u_i energy ratios for both generated and reference samples.
    """

    # iterate layers
    for i, (ref, gen) in enumerate(zip(ref_us.T, gen_us.T)):

        # create figure and subaxes
        fig, ax = plt.subplots(
            3,
            1,
            figsize=(4.5, 4),
            gridspec_kw={"height_ratios": (4, 1, 1), "hspace": 0.0},
            sharex=True,
        )

        # set binning
        if quantile_bins:
            total = np.hstack([ref])
            quantiles = np.linspace(0, 1, num_bins + 1)
            bins = np.quantile(
                total,
                (
                    quantiles[skip_quantiles:-skip_quantiles]
                    if skip_quantiles
                    else quantiles
                ),
            )
        else:
            if xlim == "auto":
                xlim = ref.min(), ref.max()
            bins = np.linspace(xlim[0], (2 if i == 0 else 1) * xlim[1], num_bins)
        bin_centers = (bins[1:] + bins[:-1]) / 2
        bin_widths = bins[1:] - bins[:-1]

        ## MAIN AXIS ##
        # plot central values
        ref_vals, _, _ = ax[0].hist(
            ref,
            bins=bins,
            density=True,
            histtype="step",
            linestyle="-",
            alpha=0.8,
            linewidth=1.0,
            color="k",
            label="Geant",
        )
        gen_vals, _, _ = ax[0].hist(
            gen,
            bins=bins,
            density=True,
            histtype="step",
            linestyle="-",
            alpha=0.8,
            linewidth=1.0,
            color=color,
            label="CaloDREAM",
        )
        # plot error bars
        ref_stds = np.sqrt(ref_vals * len(ref) / bin_widths) / len(ref)
        gen_stds = np.sqrt(gen_vals * len(gen) / bin_widths) / len(gen)
        ax[0].fill_between(
            bin_centers,
            ref_vals - ref_stds,
            ref_vals + ref_stds,
            alpha=0.2,
            step="mid",
            color="k",
        )
        ax[0].fill_between(
            bin_centers,
            gen_vals - gen_stds,
            gen_vals + gen_stds,
            alpha=0.2,
            step="mid",
            color=color,
        )
        ax[0].semilogy()
        # ax[0].set_ylim(None, min(ax[0].get_ylim()[1], 200))
        ax[0].set_ylim(max(ax[0].get_ylim()[0], 1.1e-4), None)
        ax[0].set_ylabel(f"Prob. density")

        ## RATIO AXIS ##
        norm = ref_vals
        # plot central values
        ax[1].step(
            bin_centers,
            ref_vals / norm,
            where="mid",
            linestyle="-",
            alpha=0.8,
            linewidth=1.0,
            color="k",
        )
        ax[1].step(
            bin_centers,
            gen_vals / norm,
            where="mid",
            linestyle="-",
            alpha=0.8,
            linewidth=1.0,
            color=color,
        )
        ax[1].fill_between(
            bin_centers,
            (ref_vals - ref_stds) / norm,
            (ref_vals + ref_stds) / norm,
            alpha=0.2,
            step="mid",
            color="k",
        )
        ax[1].fill_between(
            bin_centers,
            (gen_vals - gen_stds) / norm,
            (gen_vals + gen_stds) / norm,
            alpha=0.2,
            step="mid",
            color=color,
        )
        delta = np.fabs(gen_vals / norm - 1) * 100
        delta_err = gen_stds / norm * 100
        geant_delta_err = ref_stds / norm * 100
        ax[2].errorbar(
            bin_centers,
            np.zeros_like(bin_centers),
            yerr=geant_delta_err,
            ecolor="grey",
            color="grey",
            elinewidth=0.5,
            linewidth=1.0,
            fmt=".",
            capsize=2,
        )

        markers, caps, bars = ax[2].errorbar(
            bin_centers,
            delta,
            yerr=delta_err,
            ecolor=color,
            color=color,
            elinewidth=0.5,
            linewidth=1.0,
            fmt=".",
            capsize=2,
        )

        ax[1].set_ylim(*ratio_ylim)
        ax[1].set_xlim(*ax[0].get_xlim())
        ax[1].set_xlabel(f"$u_{{{i}}}$")
        ax[0].set_xticklabels([])

        ax[1].hlines(
            1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle="-", color="k"
        )
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

        ax[0].set_ylabel(r"a.u.")
        ax[2].set_xlabel(f"$u_{{{i}}}$")
        ax[0].set_xlim(*xlim)
        ax[0].set_yscale("log")
        ax[1].set_ylabel(r"$\frac{\text{Model}}{\text{Geant}}$")

        handle1 = matplotlib.lines.Line2D([], [], c="k", label="Geant")
        handle2 = matplotlib.lines.Line2D([], [], c="#0000cc", label="CaloDREAM")
        ax[0].legend(
            handles=[handle1, handle2],
            loc="upper right",
            frameon=False,
            handlelength=1.2,
            title_fontsize=18,
            fontsize=16,
        )
        fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))

        if cfg is not None:
            if not os.path.isdir(cfg.run_dir + f"/eval_{cfg.run_idx}/"):
                os.makedirs(cfg.run_dir + f"/eval_{cfg.run_idx}/")

            fig.savefig(
                cfg.run_dir + f"/eval_{cfg.run_idx}/" + f"u{i}_dist.pdf",
                dpi=200,
                bbox_inches="tight",
            )
        else:
            plt.show()

        plt.close(fig)

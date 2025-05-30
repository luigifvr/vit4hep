# pylint: disable=invalid-name
""" helper file containing plotting functions to evaluate contributions to the
    Fast Calorimeter Challenge 2022.

    by C. Krause

    modified by L. Favaro, A. Ore, S. Palacios for CaloDREAM
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.backends.backend_pdf import PdfPages

dup = lambda a: np.append(a, a[-1])
# settings for the various plots. These should be larger than the number of hlf files
colors = ["#0000cc", "#b40000"]

plt.rc("font", family="serif", size=22)
plt.rc("axes", titlesize="medium")
plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
plt.rc("text", usetex=True)
 
def plot_layer_comparison(hlf_class, data, reference_class, reference_data,
                          arg, input_name='', show=False):
    """ plots showers of of data and reference next to each other, for comparison """

    filename = os.path.join(arg.output_dir,
                                'Average_Layer_dataset_{}_{}.pdf'.format(arg.dataset, input_name))
    num_layer = len(reference_class.relevantLayers)
    vmax = np.max(reference_data)
    layer_boundaries = np.unique(reference_class.bin_edges)
    with PdfPages(filename) as pdf:
        for idx, layer_id in enumerate(reference_class.relevantLayers):
            plt.figure(figsize=(6, 4))
            reference_data_processed = reference_data\
                [:, layer_boundaries[idx]:layer_boundaries[idx+1]]
            reference_class._DrawSingleLayer(reference_data_processed,
                                             idx, filename=None,
                                             title='Reference Layer '+str(layer_id),
                                             fig=plt.gcf(), subplot=(1, 2, 1), vmax=vmax,
                                             colbar='None')
            data_processed = data[:, layer_boundaries[idx]:layer_boundaries[idx+1]]
            hlf_class._DrawSingleLayer(data_processed,
                                       idx, filename=None,
                                       title='Generated Layer '+str(layer_id),
                                       fig=plt.gcf(), subplot=(1, 2, 2), vmax=vmax, colbar='both')

            plt.savefig(pdf, dpi=300, format='pdf')
            if show:
                plt.show()
            plt.close()

def plot_Etot_Einc_discrete(hlf_class, reference_class, arg):
    """ plots Etot normalized to Einc histograms for each Einc in ds1 """
    # hardcode boundaries?
    bins = np.linspace(0.4, 1.4, 21)
    plt.figure(figsize=(10, 10))
    target_energies = 2**np.linspace(8, 23, 16)
    for i in range(len(target_energies)-1):
        if i > 3 and 'photons' in arg.dataset:
            bins = np.linspace(0.9, 1.1, 21)
        energy = target_energies[i]
        which_showers_ref = ((reference_class.Einc.squeeze() >= target_energies[i]) & \
                             (reference_class.Einc.squeeze() < target_energies[i+1])).squeeze()
        which_showers_hlf = ((hlf_class.Einc.squeeze() >= target_energies[i]) & \
                             (hlf_class.Einc.squeeze() < target_energies[i+1])).squeeze()
        ax = plt.subplot(4, 4, i+1)
        counts_ref, _, _ = ax.hist(reference_class.GetEtot()[which_showers_ref] /\
                                   reference_class.Einc.squeeze()[which_showers_ref],
                                   bins=bins, label='reference', linestyle='--', density=True,
                                   histtype='stepfilled', alpha=0.2, linewidth=1.5, color=hlf_class.color)
        counts_data, _, _ = ax.hist(hlf_class.GetEtot()[which_showers_hlf] /\
                                    hlf_class.Einc.squeeze()[which_showers_hlf], bins=bins,
                                    label='generated', histtype='step', linewidth=1.5, alpha=1.,
                                    density=True, color=reference_class.color)
        if i in [0, 1, 2]:
            energy_label = 'E = {:.0f} MeV'.format(energy)
        elif i in np.arange(3, 12):
            energy_label = 'E = {:.1f} GeV'.format(energy/1e3)
        else:
            energy_label = 'E = {:.1f} TeV'.format(energy/1e6)
        ax.text(0.95, 0.95, energy_label, ha='right', va='top',
                transform=ax.transAxes)
        ax.set_xlabel(r'$E_{\text{tot}} / E_{\text{inc}}$')
        ax.xaxis.set_label_coords(1., -0.15)
        ax.set_ylabel('counts')
        ax.yaxis.set_ticklabels([])
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        seps = _separation_power(counts_ref, counts_data, bins)
        print("Separation power of Etot / Einc at E = {} histogram: {}".format(energy, seps))
        with open(os.path.join(arg.output_dir, 'histogram_chi2_{}.txt'.format(arg.dataset)),
                  'a') as f:
            f.write('Etot / Einc at E = {}: \n'.format(energy))
            f.write(str(seps))
            f.write('\n\n')
        h, l = ax.get_legend_handles_labels()
    ax = plt.subplot(4, 4, 16)
    ax.legend(h, l, loc='center', fontsize=16)
    ax.axis('off')
    filename = os.path.join(arg.output_dir, 'Etot_Einc_dataset_{}_E_i.pdf'.format(arg.dataset))
    plt.savefig(filename, dpi=300, format='pdf')
    plt.close()

def plot_Etot_Einc(hlfs, reference_class, arg, labels, input_names, p_label):
    """ plots Etot normalized to Einc histogram """

    bins = np.linspace(0.5, 1.5, 31)
    fig, ax = plt.subplots(3,1, figsize=(4.5, 4), gridspec_kw = {"height_ratios": (4,1,1), "hspace": 0.0}, sharex = True)
        
    counts_ref, bins = np.histogram(reference_class.GetEtot() / reference_class.Einc.squeeze(), bins=bins, density=False)
    counts_ref_norm = counts_ref/counts_ref.sum()
    geant_error = counts_ref_norm/np.sqrt(counts_ref)
    geant_ratio_error = geant_error/counts_ref_norm
    geant_ratio_error_isnan = np.isnan(geant_ratio_error)
    geant_ratio_error[geant_ratio_error_isnan] = 0.
    geant_delta_err = geant_ratio_error*100
    ax[0].step(bins, dup(counts_ref_norm), label='Geant4', linestyle='-',
                   alpha=0.8, linewidth=1.0, color='k', where='post')
    ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', color='k', alpha=0.2)
    ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', color='k', alpha=0.2 )
    ax[2].errorbar((bins[:-1]+bins[1:])/2, np.zeros_like(bins[:-1]), yerr=geant_delta_err, ecolor='grey',
                           color='grey', elinewidth=0.5, linewidth=1., fmt=".", capsize=2)
    for i in range(len(hlfs)):
        counts, _ = np.histogram(hlfs[i].GetEtot() / hlfs[i].Einc.squeeze(), bins=bins, density=False)
        counts_data, bins = np.histogram(hlfs[i].GetEtot() / hlfs[i].Einc.squeeze(), bins=bins, density=False)
        counts_data_norm = counts_data/counts_data.sum()
        ax[0].step(bins, dup(counts_data_norm), label=labels[i], where='post',
                linewidth=1., alpha=1., color=colors[i], linestyle='-')

        y_ref_err = counts_data_norm/np.sqrt(counts)
        ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post', color=colors[i], alpha=0.2)
    
        ratio = counts_data / counts_ref
        ratio_err = y_ref_err/counts_ref_norm
        ratio_isnan = np.isnan(ratio)
        ratio[ratio_isnan] = 1.
        ratio_err[ratio_isnan] = 0.
        ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=colors[i], where='post')
        ax[1].fill_between(bins, dup(ratio-ratio_err), dup(ratio+ratio_err), step='post', color=colors[i], alpha=0.2)
        delta = np.fabs(ratio - 1)*100
        delta_err = ratio_err*100
        markers, caps, bars = ax[2].errorbar((bins[:-1]+bins[1:])/2, delta,
                                    yerr=delta_err, ecolor=colors[i], color=colors[i], elinewidth=0.5,
                                    linewidth=1.0, fmt=".", capsize=2)
 
        seps = _separation_power(counts_ref_norm, counts_data_norm, None)
        print("Separation power of Etot / Einc histogram: {}".format(seps))
        with open(os.path.join(arg.output_dir, 'histogram_chi2_{}_{}.txt'
                               .format(arg.dataset, input_names[i])), 'a') as f:
            f.write('Etot / Einc: \n')
            f.write(str(seps))
            f.write('\n\n')

    ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
    ax[1].set_yticks((0.7, 1.0, 1.3))
    ax[1].set_ylim(0.5, 1.5)
    ax[0].set_xlim(bins[0], bins[-1])

    ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
    ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
             
    ax[2].set_ylim((0.05, 50))
    ax[2].set_yscale("log")
    ax[2].set_yticks([0.1, 1.0, 10.0])
    ax[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
    ax[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               2., 3., 4., 5., 6., 7., 8., 9., 20., 30., 40.,], minor=True)

    ax[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
    #ax[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
    ax[2].set_ylabel(r"$\delta [\%]$")
   
    ax[2].set_xlabel(r'$E_{\text{tot}} / E_{\text{inc}}$')
    ax[0].set_ylabel(r'a.u.')
    ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{Geant4}}$')
    ax[0].legend(loc='best', frameon=False, title=p_label, handlelength=1.2, fontsize=16, title_fontsize=18)
    fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
    filename = os.path.join(arg.output_dir, 'Etot_Einc_dataset_{}.pdf'
                            .format(arg.dataset))
    fig.savefig(filename, dpi=300, format='pdf')
    plt.close()

def plot_E_layers(hlfs, reference_class, arg, labels, input_names, p_label):
    """ plots energy deposited in each layer """
    filename = os.path.join(arg.output_dir, 'E_layer_dataset_{}.pdf'.format(
                arg.dataset))
    with PdfPages(filename) as pdf:
        for key in reference_class.GetElayers().keys():
            fig, ax = plt.subplots(3, 1, figsize=(4.5, 4), gridspec_kw={"height_ratios": (4,1,1), "hspace": 0.0}, sharex=True)
            if arg.x_scale == 'log':
                bins = np.logspace(np.log10(arg.min_energy),
                                   np.log10(reference_class.GetElayers()[key].max()),
                                   40)
            else:
                bins = 40
            
            counts_ref, bins = np.histogram(reference_class.GetElayers()[key], bins=bins, density=False)
            counts_ref_norm = counts_ref/counts_ref.sum()
            geant_error = counts_ref_norm/np.sqrt(counts_ref)
            geant_ratio_error = geant_error/counts_ref_norm
            geant_ratio_error_isnan = np.isnan(geant_ratio_error)
            geant_ratio_error[geant_ratio_error_isnan] = 0.
            geant_delta_err = geant_ratio_error*100
            ax[0].step(bins, dup(counts_ref_norm), label='Geant4', linestyle='-',
                            alpha=0.8, linewidth=1.0, color='k', where='post')
            ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', color='k', alpha=0.2)
            ax[1].fill_between(bins, dup(1-geant_ratio_error), dup(1+geant_ratio_error), step='post', color='k', alpha=0.2 )
            ax[2].errorbar((bins[:-1]+bins[1:])/2, np.zeros_like(bins[:-1]), yerr=geant_delta_err, ecolor='grey',
                           color='grey', elinewidth=0.5, linewidth=1., fmt=".", capsize=2)
            for i in range(len(hlfs)):
                counts, _ = np.histogram(hlfs[i].GetElayers()[key], bins=bins, density=False)
                counts_data, bins = np.histogram(hlfs[i].GetElayers()[key], bins=bins, density=False)
                counts_data_norm = counts_data/counts_data.sum()
                ax[0].step(bins, dup(counts_data_norm), label=labels[i], where='post',
                       linewidth=1., alpha=1., color=colors[i], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post', color=colors[i], alpha=0.2)
            
                ratio = counts_data / counts_ref
                ratio_err = y_ref_err/counts_ref_norm
                ratio_isnan = np.isnan(ratio)
                ratio[ratio_isnan] = 1.
                ratio_err[ratio_isnan] = 0.
                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=colors[i], where='post')
                ax[1].fill_between(bins, dup(ratio-ratio_err), dup(ratio+ratio_err), step='post', color=colors[i], alpha=0.2)
                    
                delta = np.fabs(ratio - 1)*100
                delta_err = ratio_err*100
                markers, caps, bars = ax[2].errorbar((bins[:-1]+bins[1:])/2, delta,
                                        yerr=delta_err, ecolor=colors[i], color=colors[i], elinewidth=0.5,
                                        linewidth=1.0, fmt=".", capsize=2)
                    
                seps = _separation_power(counts_ref_norm, counts_data_norm, None)
                print("Separation power of E layer {} histogram: {}".format(key, seps))
                with open(os.path.join(arg.output_dir, 'histogram_chi2_{}_{}.txt'
                                        .format(arg.dataset, input_names[i])), 'a') as f:
                    f.write('E layer {}: \n'.format(key))
                    f.write(str(seps))
                    f.write('\n\n')

            ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
            ax[1].set_yticks((0.7, 1.0, 1.3))
            ax[1].set_ylim(0.5, 1.5)
            ax[0].set_xlim(bins[0], bins[-1])

            ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
            ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
            
            ax[2].set_ylim((0.05, 50))
            ax[2].set_yscale("log")
            ax[2].set_yticks([0.1, 1.0, 10.0])
            ax[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
            ax[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               2., 3., 4., 5., 6., 7., 8., 9., 20., 30., 40.,], minor=True)

            ax[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
            #ax[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
            ax[2].set_ylabel(r"$\delta [\%]$")

            #ax[0].set_title("Energy deposited in layer {}".format(key))
            ax[0].set_ylabel(r'a.u.')
            ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{Geant4}}$')
            ax[2].set_xlabel(f'$E_{{{key}}}$ [MeV]')
            ax[0].set_yscale('log'), ax[0].set_xscale('log')
            ax[1].set_xscale('log')
            ax[0].legend(loc='lower right', frameon=False, title=p_label, handlelength=1.2, fontsize=16, title_fontsize=18)
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
            plt.savefig(pdf, dpi=300, format='pdf')
            plt.close()

def plot_ECEtas(hlfs, reference_class, arg, labels, input_names, p_label):
    """ plots center of energy in eta """
    filename = os.path.join(arg.output_dir,
                'ECEta_layer_dataset_{}.pdf'.format(arg.dataset))
    with PdfPages(filename) as pdf:
        for key in reference_class.GetECEtas().keys():
            if arg.dataset in ['2', '3']:
                lim = (-30., 30.)
            elif key in [12, 13]:
                lim = (-500., 500.)
            else:
                lim = (-100., 100.)
            fig, ax = plt.subplots(3, 1, figsize=(4.5, 4), gridspec_kw={"height_ratios": (4,1,1), "hspace": 0.0}, sharex=True)
            bins = np.linspace(*lim, 51)

            counts_ref, bins = np.histogram(reference_class.GetECEtas()[key], bins=bins, density=False)
            counts_ref_norm = counts_ref/counts_ref.sum()
            geant_error = counts_ref_norm/np.sqrt(counts_ref)
            geant_ratio_error = geant_error/counts_ref_norm
            geant_ratio_error_isnan = np.isnan(geant_ratio_error)
            geant_ratio_error[geant_ratio_error_isnan] = 0.
            geant_delta_err = geant_ratio_error*100
            ax[0].step(bins, dup(counts_ref_norm), label='Geant4', linestyle='-',
                            alpha=0.8, linewidth=1.0, color='k', where='post')
            ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', color='k', alpha=0.2)
            ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', color='k', alpha=0.2 )
            ax[2].errorbar((bins[:-1]+bins[1:])/2, np.zeros_like(bins[:-1]), yerr=geant_delta_err, ecolor='grey',
                           color='grey', elinewidth=0.5, linewidth=1., fmt=".", capsize=2)
            for i in range(len(hlfs)):
                counts, _ = np.histogram(hlfs[i].GetECEtas()[key], bins=bins, density=False)
                counts_data, bins = np.histogram(hlfs[i].GetECEtas()[key], bins=bins, density=False)
                    
                counts_data_norm = counts_data/counts_data.sum()
                ax[0].step(bins, dup(counts_data_norm), label=labels[i], where='post',
                        linewidth=1., alpha=1., color=colors[i], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post', color=colors[i], alpha=0.2)
            
                ratio = counts_data / counts_ref
                ratio_err = y_ref_err/counts_ref_norm
                ratio_isnan = np.isnan(ratio)
                ratio[ratio_isnan] = 1.
                ratio_err[ratio_isnan] = 0.

                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=colors[i], where='post')
                ax[1].fill_between(bins, dup(ratio-ratio_err), dup(ratio+ratio_err), step='post', color=colors[i], alpha=0.2)
                delta = np.fabs(ratio - 1)*100
                delta_err = ratio_err*100
                markers, caps, bars = ax[2].errorbar((bins[:-1]+bins[1:])/2, delta,
                                        yerr=delta_err, ecolor=colors[i], color=colors[i], elinewidth=0.5,
                                        linewidth=1.0, fmt=".", capsize=2)
                
                seps = _separation_power(counts_ref_norm, counts_data_norm, None)
                
                print("Separation power of EC Eta layer {} histogram: {}".format(key, seps))
                with open(os.path.join(arg.output_dir, 'histogram_chi2_{}_{}.txt'
                                       .format(arg.dataset, input_names[i])),'a') as f:
                    f.write('EC Eta layer {}: \n'.format(key))
                    f.write(str(seps))
                    f.write('\n\n')

            ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
            ax[1].set_yticks((0.7, 1.0, 1.3))
            ax[1].set_ylim(0.5, 1.5)
            ax[0].set_xlim(bins[0], bins[-1])

            ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
            ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
            
            ax[2].set_ylim((0.05, 50))
            ax[2].set_yscale("log")
            ax[2].set_yticks([0.1, 1.0, 10.0])
            ax[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
            ax[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               2., 3., 4., 5., 6., 7., 8., 9., 20., 30., 40.,], minor=True)

            ax[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
            #ax[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
            ax[2].set_ylabel(r"$\delta [\%]$")

            #ax[0].set_title(r"Center of Energy in $\Delta\eta$ in layer {}".format(key))
            ax[0].set_ylabel(r'a.u.')
            ax[2].set_xlabel(f'$\\langle\\eta\\rangle_{{{key}}}$ [mm]')
            ax[0].set_xlim(*lim)
            ax[0].set_yscale('log')
            ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{Geant4}}$')
            ax[0].legend(loc='lower center', frameon=False, title=p_label, handlelength=1.2, title_fontsize=18, fontsize=16)
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))

            plt.savefig(pdf, dpi=300, format='pdf')
            plt.close()

def plot_ECPhis(hlfs, reference_class, arg, labels, input_names, p_label):
    """ plots center of energy in phi """
    filename = os.path.join(arg.output_dir,
                'ECPhi_layer_dataset_{}.pdf'.format(arg.dataset))
    with PdfPages(filename) as pdf:
        for key in reference_class.GetECPhis().keys():
            if arg.dataset in ['2', '3']:
                lim = (-30., 30.)
            elif key in [12, 13]:
                lim = (-500., 500.)
            else:
                lim = (-100., 100.)
            fig, ax = plt.subplots(3, 1, figsize=(4.5, 4), gridspec_kw={"height_ratios": (4,1,1), "hspace": 0.0}, sharex=True)
            bins = np.linspace(*lim, 51)
            
            counts_ref, bins = np.histogram(reference_class.GetECPhis()[key], bins=bins, density=False)
            counts_ref_norm = counts_ref/counts_ref.sum()
            geant_error = counts_ref_norm/np.sqrt(counts_ref)
            geant_ratio_error = geant_error/counts_ref_norm
            geant_ratio_error_isnan = np.isnan(geant_ratio_error)
            geant_ratio_error[geant_ratio_error_isnan] = 0.
            geant_delta_err = geant_ratio_error*100
            ax[0].step(bins, dup(counts_ref_norm), label='Geant4', linestyle='-',
                            alpha=0.8, linewidth=1.0, color='k', where='post')
            ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', color='k', alpha=0.2)
            ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', color='k', alpha=0.2 )
            ax[2].errorbar((bins[:-1]+bins[1:])/2, np.zeros_like(bins[:-1]), yerr=geant_delta_err, ecolor='grey',
                           color='grey', elinewidth=0.5, linewidth=1., fmt=".", capsize=2)
            for i in range(len(hlfs)):
                counts, _ = np.histogram(hlfs[i].GetECPhis()[key], bins=bins, density=False)
                counts_data, bins = np.histogram(hlfs[i].GetECPhis()[key], bins=bins, density=False)
                    
                counts_data_norm = counts_data/counts_data.sum()
                ax[0].step(bins, dup(counts_data_norm), label=labels[i], where='post',
                       linewidth=1., alpha=1., color=colors[i], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post', color=colors[i], alpha=0.2)
            
                ratio = counts_data / counts_ref
                ratio_err = y_ref_err/counts_ref_norm
                ratio_isnan = np.isnan(ratio)
                ratio[ratio_isnan] = 1.
                ratio_err[ratio_isnan] = 0.

                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=colors[i], where='post')
                ax[1].fill_between(bins, dup(ratio-ratio_err), dup(ratio+ratio_err), step='post', color=colors[i], alpha=0.2)
                
                delta = np.fabs(ratio - 1)*100
                delta_err = ratio_err*100
                markers, caps, bars = ax[2].errorbar((bins[:-1]+bins[1:])/2, delta,
                                        yerr=delta_err, ecolor=colors[i],  color=colors[i], elinewidth=0.5,
                                        linewidth=1.0, fmt=".", capsize=2)
 
                seps = _separation_power(counts_ref_norm, counts_data_norm, None)
                print("Separation power of EC Phi layer {} histogram: {}".format(key, seps))
                with open(os.path.join(arg.output_dir, 'histogram_chi2_{}_{}.txt'
                                        .format(arg.dataset, input_names[i])),'a') as f:
                    f.write('EC Phi layer {}: \n'.format(key))
                    f.write(str(seps))
                    f.write('\n\n')

            ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
            ax[1].set_yticks((0.7, 1.0, 1.3))
            ax[1].set_ylim(0.5, 1.5)
            ax[0].set_xlim(bins[0], bins[-1])

            ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
            ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
            
            ax[2].set_ylim((0.05, 50))
            ax[2].set_yscale("log")
            ax[2].set_yticks([0.1, 1.0, 10.0])
            ax[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
            ax[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               2., 3., 4., 5., 6., 7., 8., 9., 20., 30., 40.,], minor=True)

            ax[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
            #ax[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
            ax[2].set_ylabel(r"$\delta [\%]$")

            #ax[0].set_title(r"Center of Energy in $\Delta\phi$ in layer {}".format(key))
            ax[0].set_ylabel(r'a.u.')
            ax[2].set_xlabel(f"$\\langle\\phi\\rangle_{{{key}}}$ [mm]")
            ax[0].set_xlim(*lim)
            ax[0].set_yscale('log')
            ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{Geant4}}$')
            ax[0].legend(loc='lower center', frameon=False, title=p_label, handlelength=1.2, title_fontsize=18, fontsize=16)
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))

            plt.savefig(pdf, dpi=300, format='pdf')
            plt.close()

def plot_ECWidthEtas(hlfs, reference_class, arg, labels, input_names, p_label):
    """ plots width of center of energy in eta """
    filename = os.path.join(arg.output_dir,
                'WidthEta_layer_dataset_{}.pdf'.format(arg.dataset))
    with PdfPages(filename) as pdf:
        for key in reference_class.GetWidthEtas().keys():
            if arg.dataset in ['2', '3']:
                lim = (0., 30.)
            elif key in [12, 13]:
                lim = (0., 400.)
            else:
                lim = (0., 100.)
            fig, ax = plt.subplots(3,1, figsize=(4.5, 4), gridspec_kw={"height_ratios": (4,1,1), "hspace": 0.0}, sharex=True)
            bins = np.linspace(*lim, 51)
            
            counts_ref, bins = np.histogram(reference_class.GetWidthEtas()[key], bins=bins, density=False)
            counts_ref_norm = counts_ref/counts_ref.sum()
            geant_error = counts_ref_norm/np.sqrt(counts_ref)
            geant_ratio_error = geant_error/counts_ref_norm
            geant_ratio_error_isnan = np.isnan(geant_ratio_error)
            geant_ratio_error[geant_ratio_error_isnan] = 0.
            geant_delta_err = geant_ratio_error*100
            ax[0].step(bins, dup(counts_ref_norm), label='Geant4', linestyle='-',
                            alpha=0.8, linewidth=1.0, color='k', where='post')
            ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', color='k', alpha=0.2)
            ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', color='k', alpha=0.2 )
            ax[2].errorbar((bins[:-1]+bins[1:])/2, np.zeros_like(bins[:-1]), yerr=geant_delta_err, ecolor='grey',
                           color='grey', elinewidth=0.5, linewidth=1., fmt=".", capsize=2)
            for i in range(len(hlfs)):
                counts, _ = np.histogram(hlfs[i].GetWidthEtas()[key], bins=bins, density=False)
                counts_data, bins = np.histogram(hlfs[i].GetWidthEtas()[key], bins=bins, density=False)
                    
                counts_data_norm = counts_data/counts_data.sum()
                ax[0].step(bins, dup(counts_data_norm), label=labels[i], where='post',
                       linewidth=1., alpha=1., color=colors[i], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post', color=colors[i], alpha=0.2)
            
                ratio = counts_data / counts_ref
                ratio_err = y_ref_err/counts_ref_norm
                ratio_isnan = np.isnan(ratio)
                ratio[ratio_isnan] = 1.
                ratio_err[ratio_isnan] = 0.

                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=colors[i], where='post')
                ax[1].fill_between(bins, dup(ratio-ratio_err), dup(ratio+ratio_err), step='post', color=colors[i], alpha=0.2)
                
                delta = np.fabs(ratio - 1)*100
                delta_err = ratio_err*100
                markers, caps, bars = ax[2].errorbar((bins[:-1]+bins[1:])/2, delta,
                                        yerr=delta_err, ecolor=colors[i], color=colors[i], elinewidth=0.5,
                                        linewidth=1.0, fmt=".", capsize=2)
      
                seps = _separation_power(counts_ref_norm, counts_data_norm, None)
                print("Separation power of Width Eta layer {} histogram: {}".format(key, seps))
                with open(os.path.join(arg.output_dir, 'histogram_chi2_{}_{}.txt'
                                       .format(arg.dataset, input_names[i])),'a') as f:
                    f.write('Width Eta layer {}: \n'.format(key))
                    f.write(str(seps))
                    f.write('\n\n')

            ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
            ax[1].set_yticks((0.7, 1.0, 1.3))
            ax[1].set_ylim(0.5, 1.5)
            ax[0].set_xlim(bins[0], bins[-1])

            ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
            ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
            
            ax[2].set_ylim((0.05, 50))
            ax[2].set_yscale("log")
            ax[2].set_yticks([0.1, 1.0, 10.0])
            ax[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
            ax[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               2., 3., 4., 5., 6., 7., 8., 9., 20., 30., 40.,], minor=True)

            ax[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
            #ax[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
            ax[2].set_ylabel(r"$\delta [\%]$")
           
            ax[0].set_ylabel(r'a.u.')
            ax[2].set_xlabel(r"$\sigma_{\langle\eta\rangle_{" + str(key) + "}}$ [mm]")
            #ax[0].set_title(r"Width of Center of Energy in $\Delta\eta$ in layer {}".format(key))
            ax[0].set_xlim(*lim)
            ax[0].set_yscale('log')
            ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{Geant4}}$')
            ax[0].legend(loc='lower left', frameon=False, title=p_label, handlelength=1.2, fontsize=16, title_fontsize=18)
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
     
            plt.savefig(pdf, dpi=300, format='pdf')
            plt.close()

def plot_ECWidthPhis(hlfs, reference_class, arg, labels, input_names, p_label):
    """ plots width of center of energy in phi """
    filename = os.path.join(arg.output_dir,
                    'WidthPhi_layer_dataset_{}.pdf'.format(arg.dataset))
    with PdfPages(filename) as pdf:
        for key in reference_class.GetWidthPhis().keys():
            if arg.dataset in ['2', '3']:
                lim = (0., 30.)
            elif key in [12, 13]:
                lim = (0., 400.)
            else:
                lim = (0., 100.)
            fig, ax = plt.subplots(3, 1, figsize=(4.5, 4), gridspec_kw={"height_ratios": (4,1,1), "hspace": 0.0}, sharex=True)
            bins = np.linspace(*lim, 51)
            
            counts_ref, bins = np.histogram(reference_class.GetWidthPhis()[key], bins=bins, density=False)
            counts_ref_norm = counts_ref/counts_ref.sum()
            geant_error = counts_ref_norm/np.sqrt(counts_ref)
            geant_ratio_error = geant_error/counts_ref_norm
            geant_ratio_error_isnan = np.isnan(geant_ratio_error)
            geant_ratio_error[geant_ratio_error_isnan] = 0.
            geant_delta_err = geant_ratio_error*100
            ax[0].step(bins, dup(counts_ref_norm), label='Geant4', linestyle='-',
                            alpha=0.8, linewidth=1.0, color='k', where='post')
            ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', color='k', alpha=0.2)
            ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', color='k', alpha=0.2 )
            ax[2].errorbar((bins[:-1]+bins[1:])/2, np.zeros_like(bins[:-1]), yerr=geant_delta_err, ecolor='grey',
                           color='grey', elinewidth=0.5, linewidth=1., fmt=".", capsize=2)
            for i in range(len(hlfs)):
                counts, _ = np.histogram(hlfs[i].GetWidthPhis()[key], bins=bins, density=False)
                counts_data, bins = np.histogram(hlfs[i].GetWidthPhis()[key], bins=bins, density=False)
                    
                counts_data_norm = counts_data/counts_data.sum()
                ax[0].step(bins, dup(counts_data_norm), label=labels[i], where='post',
                        linewidth=1., alpha=1., color=colors[i], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post', color=colors[i], alpha=0.2)
            
                ratio = counts_data / counts_ref
                ratio_err = y_ref_err/counts_ref_norm
                ratio_isnan = np.isnan(ratio)
                ratio[ratio_isnan] = 1.
                ratio_err[ratio_isnan] = 0.

                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=colors[i], where='post')
                ax[1].fill_between(bins, dup(ratio-ratio_err), dup(ratio+ratio_err), step='post', color=colors[i], alpha=0.2)
                
                delta = np.fabs(ratio - 1)*100
                delta_err = ratio_err*100
                markers, caps, bars = ax[2].errorbar((bins[:-1]+bins[1:])/2, delta,
                                        yerr=delta_err, ecolor=colors[i],  color=colors[i], elinewidth=0.5,
                                        linewidth=1.0, fmt=".", capsize=2)
 
                seps = _separation_power(counts_ref_norm, counts_data_norm, None)
                print("Separation power of Width Phi layer {} histogram: {}".format(key, seps))
                with open(os.path.join(arg.output_dir, 'histogram_chi2_{}_{}.txt'
                                       .format(arg.dataset, input_names[i])),'a') as f:
                    f.write('Width Phi layer {}: \n'.format(key))
                    f.write(str(seps))
                    f.write('\n\n')

            ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
            ax[1].set_yticks((0.7, 1.0, 1.3))
            ax[1].set_ylim(0.5, 1.5)
            ax[0].set_xlim(bins[0], bins[-1])

            ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
            ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
            
            ax[2].set_ylim((0.05, 50))
            ax[2].set_yscale("log")
            ax[2].set_yticks([0.1, 1.0, 10.0])
            ax[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
            ax[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               2., 3., 4., 5., 6., 7., 8., 9., 20., 30., 40.,], minor=True)

            ax[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
            #ax[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
            ax[2].set_ylabel(r"$\delta [\%]$")

            ax[0].set_ylabel(r'a.u.')
            ax[2].set_xlabel(r"$\sigma_{\langle\phi\rangle_{" + str(key) + "}}$ [mm]")
            #ax[0].set_title(r"Width of Center of Energy in $\Delta\phi$ in layer {}".format(key))
            ax[0].set_xlim(*lim)
            ax[0].set_yscale('log')
            ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{Geant4}}$')
            ax[0].legend(loc='lower left', frameon=False, title=p_label, handlelength=1.2, fontsize=16, title_fontsize=18)
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))

            plt.savefig(pdf, dpi=300, format='pdf')
            plt.close()

def plot_weighted_depth_r(hlfs, reference_hlf, arg, labels, input_names, p_label, l=1):
    """ Plot weighted depth"""
    filename = os.path.join(arg.output_dir,
                'Weighted_Depth_slice_dataset_{}_groups_{}.pdf'.format(arg.dataset, l))
    g = 0
    with PdfPages(filename) as pdf:
        func_ref = reference_hlf.GetWeightedDepthR() if l==1 else reference_hlf.GetGroupedWeightedDepthR()
        for n, key in enumerate(func_ref.keys()):
            bins = np.linspace(g*len(reference_hlf.relevantLayers)/l,
                               (g+1)*len(reference_hlf.relevantLayers)/l, 40)
            fig, ax = plt.subplots(3, 1, figsize=(4.5, 4), gridspec_kw={"height_ratios": (4,1,1), "hspace": 0.0}, sharex=True)
            counts_ref, bins = np.histogram(func_ref[key], bins=bins, density=False)
            counts_ref_norm = counts_ref/counts_ref.sum()
            geant_error = counts_ref_norm/np.sqrt(counts_ref)
            geant_ratio_error = geant_error/counts_ref_norm
            geant_ratio_error_isnan = np.isnan(geant_ratio_error)
            geant_ratio_error[geant_ratio_error_isnan] = 0.
            geant_delta_err = geant_ratio_error*100
            ax[0].step(bins, dup(counts_ref_norm), label='Geant4', linestyle='-',
                            alpha=0.8, linewidth=1.0, color='k', where='post')
            ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', color='k', alpha=0.2)
            ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', color='k', alpha=0.2 )
            ax[2].errorbar((bins[:-1]+bins[1:])/2, np.zeros_like(bins[:-1]), yerr=geant_delta_err, ecolor='grey',
                           color='grey', elinewidth=0.5, linewidth=1., fmt=".", capsize=2)
            for j, file in enumerate(hlfs):
                func_hlf = hlfs[j].GetWeightedDepthR() if l==1 else hlfs[j].GetGroupedWeightedDepthR()
                counts, _ = np.histogram(func_hlf[key], bins=bins, density=False)
                counts_data, bins = np.histogram(func_hlf[key], bins=bins, density=False)
                counts_data_norm = counts_data/counts_data.sum()
                
                ax[0].step(bins, dup(counts_data_norm), label=labels[j], where='post',
                           linewidth=1., alpha=1., color=colors[j], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post', color=colors[j], alpha=0.2)
            
                ratio = counts_data / counts_ref
                ratio_err = y_ref_err/counts_ref_norm
                ratio_isnan = np.isnan(ratio)
                ratio[ratio_isnan] = 1.
                ratio_err[ratio_isnan] = 0.

                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=colors[j], where='post')
                ax[1].fill_between(bins, dup(ratio-ratio_err), dup(ratio+ratio_err), step='post', color=colors[j], alpha=0.2)

                delta = np.fabs(ratio - 1)*100
                delta_err = ratio_err*100
                markers, caps, bars = ax[2].errorbar((bins[:-1]+bins[1:])/2, delta,
                                        yerr=delta_err, ecolor=colors[j],  color=colors[j], elinewidth=0.5,
                                        linewidth=1.0, fmt=".", capsize=2)
 
                seps = _separation_power(counts_ref_norm, counts_data_norm, None)
                print("Separation power of Weighted depth slice {} histogram: {}".format(key, seps))
                with open(os.path.join(arg.output_dir, 'histogram_chi2_{}_{}.txt'
                                       .format(arg.dataset, input_names[j])), 'a') as f:
                    f.write('Weighted depth slice {}: \n'.format(key))
                    f.write(str(seps))
                    f.write('\n\n')
            
            ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
            ax[1].set_yticks((0.7, 1.0, 1.3))
            ax[1].set_ylim(0.5, 1.5)
            ax[0].set_xlim(bins[0], bins[-1])
            
            ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
            ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
                
            ax[2].set_ylim((0.05, 50))
            ax[2].set_yscale("log")
            ax[2].set_yticks([0.1, 1.0, 10.0])
            ax[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
            ax[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               2., 3., 4., 5., 6., 7., 8., 9., 20., 30., 40.,], minor=True)

            ax[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
            #ax[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
            ax[2].set_ylabel(r"$\delta [\%]$")
 
            ax[0].set_ylabel(r'a.u.')
            ax[2].set_xlabel(r"$d_{\alpha_{"+str(key)+"}}$")
            #ax[0].set_xlim(*lim)
            ax[0].set_yscale('log')
            ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{Geant4}}$')
            ax[0].legend(loc='upper right', frameon=False, title=p_label, handlelength=1.2, fontsize=16, title_fontsize=18)
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
            if (n+1)%(reference_hlf.num_alpha[0]) == 0:
                g += 1
            plt.savefig(pdf, format='pdf')
            plt.close()

def plot_weighted_depth_a(hlfs, reference_class, arg, labels, input_names, p_label, l=1):
    """ Plot weighted depth"""
    g = 0
    filename = os.path.join(arg.output_dir,
            'Weighted_Depth_ring_dataset_{}_groups_{}.pdf'.format(arg.dataset, l))

    with PdfPages(filename) as pdf:
        func_ref = reference_class.GetWeightedDepthA() if l==1 else reference_class.GetGroupedWeightedDepthA()
        for n, key in enumerate(func_ref.keys()):
            bins = np.linspace(g*len(reference_class.relevantLayers)/l,
                               (g+1)*len(reference_class.relevantLayers)/l, 40)
            fig, ax = plt.subplots(3, 1, figsize=(4.5, 4), gridspec_kw={"height_ratios": (4,1,1), "hspace": 0.0}, sharex=True)
            counts_ref, bins = np.histogram(func_ref[key], bins=bins, density=False)
            counts_ref_norm = counts_ref/counts_ref.sum()
            geant_error = counts_ref_norm/np.sqrt(counts_ref)
            geant_ratio_error = geant_error/counts_ref_norm
            geant_ratio_error_isnan = np.isnan(geant_ratio_error)
            geant_ratio_error[geant_ratio_error_isnan] = 0.
            geant_delta_err = geant_ratio_error*100
            ax[0].step(bins, dup(counts_ref_norm), label='Geant4', linestyle='-',
                            alpha=0.8, linewidth=1.0, color='k', where='post')
            ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', color='k', alpha=0.2)
            ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', color='k', alpha=0.2 )
            ax[2].errorbar((bins[:-1]+bins[1:])/2, np.zeros_like(bins[:-1]), yerr=geant_delta_err, ecolor='grey',
                           color='grey', elinewidth=0.5, linewidth=1., fmt=".", capsize=2)
            for j, file in enumerate(hlfs):
                func_hlf = hlfs[j].GetWeightedDepthA() if l==1 else hlfs[j].GetGroupedWeightedDepthA()
                counts, _ = np.histogram(func_hlf[key], bins=bins, density=False)
                counts_data, bins = np.histogram(func_hlf[key], bins=bins, density=False)
                counts_data_norm = counts_data/counts_data.sum()
                
                ax[0].step(bins, dup(counts_data_norm), label=labels[j], where='post',
                           linewidth=1., alpha=1., color=colors[j], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post', color=colors[j], alpha=0.2)
            
                ratio = counts_data / counts_ref
                ratio_err = y_ref_err/counts_ref_norm
                ratio_isnan = np.isnan(ratio)
                ratio[ratio_isnan] = 1.
                ratio_err[ratio_isnan] = 0.

                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=colors[j], where='post')
                ax[1].fill_between(bins, dup(ratio-ratio_err), dup(ratio+ratio_err), step='post', color=colors[j], alpha=0.2)

                delta = np.fabs(ratio - 1)*100
                delta_err = ratio_err*100
                markers, caps, bars = ax[2].errorbar((bins[:-1]+bins[1:])/2, delta,
                                        yerr=delta_err, ecolor=colors[j], color=colors[j], elinewidth=0.5,
                                        linewidth=1.0, fmt=".", capsize=2)
 
                seps = _separation_power(counts_ref_norm, counts_data_norm, None)
                print("Separation power of Weighted depth ring {} histogram: {}".format(key, seps))
                with open(os.path.join(arg.output_dir, 'histogram_chi2_{}_{}.txt'
                                       .format(arg.dataset, input_names[j])), 'a') as f:
                    f.write('Weighted depth ring {}: \n'.format(key))
                    f.write(str(seps))
                    f.write('\n\n')

            ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
            ax[1].set_yticks((0.7, 1.0, 1.3))
            ax[1].set_ylim(0.5, 1.5)
            ax[0].set_xlim(bins[0], bins[-1])
            
            ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
            ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
                        
            ax[2].set_ylim((0.05, 50))
            ax[2].set_yscale("log")
            ax[2].set_yticks([0.1, 1.0, 10.0])
            ax[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
            ax[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               2., 3., 4., 5., 6., 7., 8., 9., 20., 30., 40.,], minor=True)

            ax[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
            #ax[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
            ax[2].set_ylabel(r"$\delta [\%]$")
 
            ax[0].set_ylabel(r'a.u.')
            ax[2].set_xlabel(r"$d_{r_{"+str(key)+"}}$")
            #ax[0].set_xlim(*lim)
            ax[0].set_yscale('log')
            ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{Geant4}}$')
            ax[0].legend(loc='upper right', frameon=False, title=p_label, handlelength=1.2, fontsize=16, title_fontsize=18)
            fig.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
            if (n+1)%(len(reference_class.r_edges[0])-1)==0:
                g += 1

            plt.savefig(pdf, format='pdf')
            plt.close()

def plot_sparsity(hlfs, reference_class, arg, labels, input_names, p_label):
    """ Plot sparsity of relevant layers"""
    filename = os.path.join(arg.output_dir,
                'Sparsity_layer_dataset_{}.pdf'.format(arg.dataset))
    with PdfPages(filename) as pdf:
        for key in reference_class.GetSparsity().keys():
            lim = (0, 1)
            fig, ax = plt.subplots(3, 1, figsize=(4.5, 4), gridspec_kw={"height_ratios": (4,1,1), "hspace": 0.0}, sharex=True)
            bins = np.linspace(*lim, 20)
            
            counts_ref, bins = np.histogram(1-reference_class.GetSparsity()[key], bins=bins, density=False)
            counts_ref_norm = counts_ref/counts_ref.sum()
            geant_error = counts_ref_norm/np.sqrt(counts_ref)
            geant_ratio_error = geant_error/counts_ref_norm
            geant_ratio_error_isnan = np.isnan(geant_ratio_error)
            geant_ratio_error[geant_ratio_error_isnan] = 0.
            geant_delta_err = geant_ratio_error*100
            ax[0].step(bins, dup(counts_ref_norm), label='Geant4', linestyle='-',
                            alpha=0.8, linewidth=1.0, color='k', where='post')
            ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', color='k', alpha=0.2)
            ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', color='k', alpha=0.2 )
            ax[2].errorbar((bins[:-1]+bins[1:])/2, np.zeros_like(bins[:-1]), yerr=geant_delta_err, ecolor='grey',
                           color='grey', elinewidth=0.5, linewidth=1., fmt=".", capsize=2)
            for i in range(len(hlfs)):
                counts, _ = np.histogram(1-hlfs[i].GetSparsity()[key], bins=bins, density=False)
                counts_data, bins = np.histogram(1-hlfs[i].GetSparsity()[key], bins=bins, density=False)
                    
                counts_data_norm = counts_data/counts_data.sum()
                ax[0].step(bins, dup(counts_data_norm), label=labels[i], where='post',
                       linewidth=1., alpha=1., color=colors[i], linestyle='-')
                y_ref_err = counts_data_norm/np.sqrt(counts)
                ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post', color=colors[i], alpha=0.2)
            
                ratio = counts_data / counts_ref
                ratio_err = y_ref_err/counts_ref_norm
                ratio_isnan = np.isnan(ratio)
                ratio[ratio_isnan] = 1.
                ratio_err[ratio_isnan] = 0.

                ax[1].step(bins, dup(ratio), linewidth=1.0, alpha=1.0, color=colors[i], where='post')
                ax[1].fill_between(bins, dup(ratio-ratio_err), dup(ratio+ratio_err), step='post', color=colors[i], alpha=0.2)
                
                delta = np.fabs(ratio - 1)*100
                delta_err = ratio_err*100
                markers, caps, bars = ax[2].errorbar((bins[:-1]+bins[1:])/2, delta,
                                        yerr=delta_err, ecolor=colors[i],  color=colors[i], elinewidth=0.5,
                                        linewidth=1.0, fmt=".", capsize=2)
 
                seps = _separation_power(counts_ref_norm, counts_data_norm, None)
                print("Separation power of Sparsity layer {} histogram: {}".format(key, seps))
                with open(os.path.join(arg.output_dir, 'histogram_chi2_{}_{}.txt'
                                       .format(arg.dataset, input_names[i])), 'a') as f:
                    f.write('Sparsity {}: \n'.format(key))
                    f.write(str(seps))
                    f.write('\n\n')

            ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
            ax[1].set_yticks((0.7, 1.0, 1.3))
            ax[1].set_ylim(0.5, 1.5)
            ax[0].set_xlim(bins[0], bins[-1])

            ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
            ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
             
            ax[2].set_ylim((0.05, 50))
            ax[2].set_yscale("log")
            ax[2].set_yticks([0.1, 1.0, 10.0])
            ax[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
            ax[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                               2., 3., 4., 5., 6., 7., 8., 9., 20., 30., 40.,], minor=True)

            ax[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
            #ax[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
            ax[2].set_ylabel(r"$\delta [\%]$")

            ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{Geant4}}$')
            ax[0].set_ylabel(r'a.u.')
            ax[2].set_xlabel(f"$\\lambda_{{{key}}}$")
            #plt.yscale('log')
            ax[1].set_xlim(*lim)
            ax[0].legend(loc='upper center', frameon=False, title=p_label, handlelength=1.2, fontsize=16, title_fontsize=18)
            fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
            plt.savefig(pdf, format='pdf')
            plt.close()

def plot_cell_dist(list_showers, ref_shower_arr, arg, labels, input_names, p_label):
    """ plots voxel energies across all layers """
    fig, ax = plt.subplots(3,1, figsize=(4.5, 4), gridspec_kw={"height_ratios": (4,1,1), "hspace": 0.0}, sharex=True)
    if arg.x_scale == 'log':
        bins = np.logspace(np.log10(arg.min_energy),
                           np.log10(ref_shower_arr.max()),
                           50)
    else:
        bins = 50

    counts_ref, bins = np.histogram(ref_shower_arr, bins=bins, density=False)
    counts_ref_norm = counts_ref/counts_ref.sum()
    geant_error = counts_ref_norm/np.sqrt(counts_ref)
    geant_ratio_error = geant_error/counts_ref_norm
    geant_ratio_error_isnan = np.isnan(geant_ratio_error)
    geant_ratio_error[geant_ratio_error_isnan] = 0.
    geant_delta_err = geant_ratio_error*100
    ax[0].step(bins, dup(counts_ref_norm), label='Geant4', linestyle='-',
                        alpha=0.8, linewidth=1.0, color='k', where='post')
    ax[0].fill_between(bins, dup(counts_ref_norm+geant_error), dup(counts_ref_norm-geant_error), step='post', color='k', alpha=0.2)
    ax[1].fill_between(bins, dup(1-geant_error/counts_ref_norm), dup(1+geant_error/counts_ref_norm), step='post', color='k', alpha=0.2 )
    ax[2].errorbar((bins[:-1]+bins[1:])/2, np.zeros_like(bins[:-1]), yerr=geant_delta_err, ecolor='grey',
                           color='grey', elinewidth=0.5, linewidth=1., fmt=".", capsize=2)
    for i in range(len(list_showers)):
        counts, _ = np.histogram(list_showers[i].flatten(), bins=bins, density=False)
        counts_data, bins = np.histogram(list_showers[i].flatten(), bins=bins, density=False)

        counts_data_norm = counts_data/counts_data.sum()
        ax[0].step(bins, dup(counts_data_norm), label=labels[i], where='post',
                   linewidth=1.5, alpha=1., color=colors[i], linestyle='-')

        y_ref_err = counts_data_norm/np.sqrt(counts)
        ax[0].fill_between(bins, dup(counts_data_norm+y_ref_err), dup(counts_data_norm-y_ref_err), step='post', color=colors[i], alpha=0.2)
    
        ratio = counts_data_norm / counts_ref_norm
        ratio_err = y_ref_err/counts_ref_norm
        ratio_isnan = np.isnan(ratio)
        ratio[ratio_isnan] = 1.
        ratio_err[ratio_isnan] = 0.

        ax[1].step(bins, dup(ratio), linewidth=1.5, alpha=1.0, color=colors[i], where='post')
        ax[1].fill_between(bins, dup(ratio-y_ref_err/counts_ref), dup(ratio+y_ref_err/counts_ref), step='post', color=colors[i], alpha=0.2)
        delta = np.fabs(ratio - 1)*100
        delta_err = ratio_err*100
        markers, caps, bars = ax[2].errorbar((bins[:-1]+bins[1:])/2, delta,
                                yerr=delta_err, ecolor=colors[i], color=colors[i], elinewidth=0.5,
                                linewidth=1.0, fmt=".", capsize=2)
 
        seps = _separation_power(counts_ref_norm, counts_data_norm, bins=None)
        print("Separation power of voxel distribution histogram: {}".format(seps))
        with open(os.path.join(arg.output_dir,
                           'histogram_chi2_{}_{}.txt'.format(arg.dataset, input_names[i])), 'a') as f:
            f.write('Voxel distribution: \n')
            f.write(str(seps))
            f.write('\n\n')

    ax[1].hlines(1.0, bins[0], bins[-1], linewidth=1.0, alpha=0.8, linestyle='-', color='k')
    ax[1].set_yticks((0.7, 1.0, 1.3))
    ax[1].set_ylim(0.5, 1.5)
    ax[0].set_xlim(bins[0]+2*arg.min_energy, bins[-1])
    ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{Geant4}}$')

    ax[1].axhline(0.7, c='k', ls='--', lw=0.5)
    ax[1].axhline(1.3, c='k', ls='--', lw=0.5)
    ax[2].set_ylim((0.05, 50))
    ax[2].set_yscale("log")
    ax[2].set_yticks([0.1, 1.0, 10.0])
    ax[2].set_yticklabels([r"$0.1$", r"$1.0$", "$10.0$"])
    ax[2].set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
                    2., 3., 4., 5., 6., 7., 8., 9., 20., 30., 40.,], minor=True)
    
    ax[2].axhline(y=1.0, linewidth=0.5, linestyle="--", color="grey")
    #ax[2].axhspan(0, 1.0, facecolor="#cccccc", alpha=0.3)
    ax[2].set_ylabel(r"$\delta [\%]$")

    ax[1].set_ylabel(r'$\frac{\text{Model}}{\text{Geant4}}$')    
    ax[0].set_ylabel(r'a.u.')
    ax[2].set_xlabel(r'$E$ [MeV]')
    ax[0].set_yscale('log')
    if arg.x_scale == 'log':
        ax[1].set_xscale('log')
    #plt.xlim(*lim)
    ax[0].legend(loc='best', frameon=False, title=p_label, handlelength=1.2, fontsize=16, title_fontsize=18)
    fig.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0, rect=(0.01, 0.01, 0.98, 0.98))
    filename = os.path.join(arg.output_dir,
                                'voxel_energy_dataset_{}.pdf'.format(arg.dataset))
    plt.savefig(filename, dpi=300, format='pdf')
    plt.close()

def _separation_power(hist1, hist2, bins):
    """ computes the separation power aka triangular discrimination (cf eq. 15 of 2009.03796)
        Note: the definition requires Sum (hist_i) = 1, so if hist1 and hist2 come from
        plt.hist(..., density=True), we need to multiply hist_i by the bin widhts

        If bins=None, the histograms are already properly normalized
    """
    if bins is not None:
        hist1, hist2 = hist1*np.diff(bins), hist2*np.diff(bins)
    ret = (hist1 - hist2)**2
    ret /= hist1 + hist2 + 1e-16
    return 0.5 * ret.sum()


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn
import h5py
import scipy.stats
import os


def plot_latent_distributions(results, alpha=0.1, color="grey", save=False, path="./"):
    lmean, lstd = results['lmean'][:], results['lstd'][:]
    for idx in range(lmean.shape[1]):
        f, ax = plt.subplots()
        ax.set_title("Latent %i" % idx)
        ax.set_xlabel("Latent value")
        plt.subplots_adjust(top=0.92)
        for i in range(len(lmean)):
            mu, sigma = lmean[i, idx], lstd[i, idx]
            x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            ax.plot(x, scipy.stats.norm.pdf(x, mu, sigma), alpha=alpha, c=color)
        if save:
            plt.savefig(path + "latent_distribution_%i.png" % idx)
            plt.close()


def corner_latents(results, s=5, height=3, save=False, path="./"):
    lmean, lstd = results['lmean'][:], results['lstd'][:]
    data = pd.DataFrame(scipy.stats.norm.rvs(lmean, lstd))
    cplot = seaborn.pairplot(data, corner=True, plot_kws=dict(s=s), height=height)
    if save:
        plt.savefig(path + "latents.png")
    return cplot


def plot_profiles(results, index, fig=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
        plt.loglog()
        ax.set_xlabel("Radius [kpc/$h$]")
        ax.set_ylabel(r"Density [$\mathrm{M_\odot} \, h^{2} / \mathrm{kpc}^{3}$]")
        ax.plot([], [], color="k", label="truth")
        ax.plot([], [], color="k", ls="--", label="predicted")
        ax.legend(loc="best")

    ax.plot(results['r'][index], 10 ** results['rho_pred'][index], color="C" + str(index))
    ax.plot(results['r'][index], 10 ** results['rho_true'][index], ls="--", color="C" + str(index))
    return fig, ax


def plot_true_vs_predicted(results, fig=None, ax=None, s=1, figsize=(14, 6.5), num_rows=2, apply_logxy=False,
                           xkey='rho_true', ykey='rho_pred', clabel='r', idx=None, rbins=None, save=False, path="./",
                           xlabel=r"$\log_{10} [\rho_\mathrm{true}/\rho_\mathrm{M}]$",
                           ylabel=r"$\log_{10} [\rho_\mathrm{pred}/\rho_\mathrm{M}]$"):
    if rbins is None:
        rbins = range(results[clabel].shape[1])
    cols = int(np.ceil(len(rbins)/num_rows))
    if ax is None:
        fig, ax = plt.subplots(num_rows, cols, figsize=figsize)
        fig.text(0.5, 0.02, xlabel, ha='center', fontsize=23)
        fig.text(0.02, 0.5, ylabel, va='center', rotation='vertical', fontsize=23)

    if idx is None:
        idx = ~np.any(results[xkey][:] <=0, axis=1)
    ax1 = ax.flatten()
    x, y, c = results[xkey][idx, :], results[ykey][idx, :], results[clabel][idx, :]

    if apply_logxy:
        rho_c0 = 277.536627245708
        omegaM0 = 0.3089
        x, y = np.log10(x/(omegaM0 * rho_c0)), np.log10(y/(omegaM0 * rho_c0))

    for axnum, i in enumerate(rbins):
        ax1[axnum].plot([x[:, i].min(), x[:, i].max()], [x[:, i].min(), x[:, i].max()], ls="--", color="dimgrey")
        ax1[axnum].scatter(x[:, i], y[:, i], c=plt.cm.viridis(plt.Normalize()(c[:, i])), s=s)
        ax1[axnum].set_title("Radial bin %i" % axnum, fontsize=18)

    plt.subplots_adjust(hspace=0.4, wspace=0.4, left=0.08, right=0.97)
    if save:
        plt.savefig(path + "predictions_truth_predicted.png")
    return fig, ax


def histogram_residuals_bins(results, fig=None, ax=None, figsize=(14, 6.5), num_rows=2, bins=100,
                           xkey='rho_true', ykey='rho_pred', clabel='r', idx=None, rbins=None, save=False, path="./",
                           xlabel=r"$\log_{10} [\rho_\mathrm{pred}/[\rho_\mathrm{true}]$", color=None, label=None, **kwargs):
    if rbins is None:
        rbins = range(results[clabel].shape[1])
    cols = int(np.ceil(len(rbins)/num_rows))
    if ax is None:
        fig, ax = plt.subplots(num_rows, cols, figsize=figsize)
        fig.text(0.5, 0.02, xlabel, ha='center', fontsize=23)

    if idx is None:
        idx = ~np.any(results[xkey][:] <=0, axis=1)
    ax1 = ax.flatten()
    x, y, c = results[xkey][idx, :], results[ykey][idx, :], results[clabel][idx, :]
    for axnum, i in enumerate(rbins):
        ax1[axnum].hist(y[:, i] - x[:, i], bins=bins, histtype="step", lw=1.5, color=color, label=label, **kwargs)
        ax1[axnum].set_title("Radial bin %i" % axnum, fontsize=18)
        ax1[axnum].axvline(x=0, color="dimgrey")

    plt.subplots_adjust(hspace=0.4, wspace=0.4, left=0.08, right=0.97)
    if save:
        plt.savefig(path + "histogram_residuals.png")
    return fig, ax


def plot_residuals_true_vs_predicted(results, fig=None, ax=None, figsize=None, apply_logxy=False, color=None,
                                     xkey='rho_true', ykey='rho_pred', clabel='r', idx=None, rho_c0=277.536627245708,
                                     xlabel=r"$r [\mathrm{kpc}/h]$", omegaM0=0.3089, label="IVE", rbins=None, marker="o",
                                     ylabel=r"$\log_{10} [\rho_\mathrm{pred}/\rho_\mathrm{M}]$", xnoise=0.):
    if rbins is None:
        rbins = range(results[clabel].shape[1])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale('log')
        ax.axhline(y=0, color="k")

    if idx is None:
        idx = ~np.any(results[xkey][:] <=0, axis=1)

    x, y, c = results[xkey][idx, :], results[ykey][idx, :], results[clabel][idx, :]
    if apply_logxy:
        x, y = np.log10(x/(omegaM0 * rho_c0)), np.log10(y/(omegaM0 * rho_c0))

    #xrbins = np.mean(c[:, rbins], axis=0) + np.diff(np.mean(c[:, rbins], axis=0))[0] * xnoise
    xrbins = np.mean(c[:, rbins], axis=0) + np.mean(c[:, rbins], axis=0) * xnoise
    ax.errorbar(xrbins, np.mean(y[:, rbins] - x[:, rbins], axis=0), yerr=np.std(y[:,rbins] - x[:,rbins], axis=0),
                marker=marker, label=label, fmt=marker, color=color)
    ax.legend(loc="best")
    return fig, ax


def plot_loss_curves(logfile, figsize=(12, 5)):
    import pandas as pd
    data = pd.read_csv(logfile, sep=",", header=0)
    metrics = [x for x in list(data.keys()) if not x.startswith('val_') and x != 'epoch']
    f, axes = plt.subplots(1, len(metrics), figsize=figsize)
    axes = axes.flatten()
    for i, metric in enumerate(metrics):
        axes[i].plot(data['epoch'], data[metric], color="C" + str(i), label=metric)
        axes[i].plot(data['epoch'], data["val_" + metric], ls="--", color="C" + str(i))
        axes[i].legend(loc="best")
        axes[i].set_xlabel("Epoch")
    plt.subplots_adjust(wspace=0.4, left=0.08)


def plot_MI(results, save=False, ax=None, path="./", bw=None):
    if bw is None:
        MI_mean, MI_std = results['MI_latents_truth_mean'][:], results['MI_latents_truth_std'][:]
    else:
        MI_mean = results['MI_latents_truth_mean_bw%.1f' % bw][:]
    reff = np.mean(results['r'], axis=0)
    if ax is None:
        f, ax = plt.subplots()
    if bw is None:
        for l in range(MI_mean.shape[0]):
            ax.errorbar(reff, MI_mean[l], yerr=MI_std[l], label="Latent %i" % l, marker="o", ms=5)
    else:
        for l in range(MI_mean.shape[1]):
            ax.errorbar(reff, MI_mean[0, l], label="Latent %i" % l, marker="o", ms=5)
    ax.set_xlabel(r"$r_\mathrm{eff} [\mathrm{kpc}\, / \, h]$")
    ax.set_ylabel(r"MI (latent, $\rho_\mathrm{true} (r))$")
    ax.legend(loc="best")
    ax.set_xscale("log")
    if save:
        plt.savefig(path + "MI_latents_truth.png")

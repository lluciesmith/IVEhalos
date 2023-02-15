import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KernelDensity
import scipy.integrate as integrate
from multiprocessing import Pool
import multiprocessing
import itertools
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def KL_mi_truth_params(truth, params, nsamples=50, num_radial_bins=16, bandwidth=0.2, epsrel=1.49e-08, pool=False):
    mi_kls = np.zeros((nsamples, len(params), num_radial_bins))
    for i in range(nsamples):
        for j in range(len(params)):
            p = normalise(params[j])
            if pool is True:
                mi_kls[i, j, :] = pool_mi_sampleslatent_trutheachrbin(p, truth, bandwidth, num_radial_bins, epsrel)
            else:
                mi_kls[i, j, :] = mi_samples_latent_vs_truth_rbins(p, truth, bandwidth, num_radial_bins, epsrel)
    return mi_kls


# Mutual information between ground truth in every radial bin and latents

def sklearn_mi_truth_latents(truth, l_mean, l_std, nsamples=50, num_radial_bins=16,
                             n_neighbors=2, seed=None, discrete_features='auto'):
    assert all(l_mean[:num_radial_bins, 0] == l_mean[0, 0])
    lm = l_mean[::num_radial_bins, :]
    lstd = l_std[::num_radial_bins, :]
    dim_latent = l_mean.shape[1]

    r = np.zeros((nsamples, dim_latent, num_radial_bins))
    for i in range(nsamples):
        for j in range(dim_latent):
            if seed is not None:
                np.random.seed(seed)
            samples = np.random.normal(lm[:, j], lstd[:, j], len(lm[:, j])).reshape(len(lm[:, j]), )
            for k in range(num_radial_bins):
                r[i, j, k] = mutual_info_regression(truth[k::num_radial_bins].reshape(-1, 1), samples,
                                                    n_neighbors=n_neighbors, discrete_features=discrete_features)
    return r


def rearrange_arrays(truth, l_mean, l_std, num_radial_bins):
    assert all(l_mean[:num_radial_bins, 0] == l_mean[0, 0])
    print("Re-arranging arrays of truth and latents for MI calculation")
    lm, lstd = l_mean[::num_radial_bins, :], l_std[::num_radial_bins, :]
    truth2d = truth.reshape((len(lm), num_radial_bins))
    return truth2d, lm, lstd


def KL_mi_truth_latents(truth, l_mean, l_std, nsamples=50, num_radial_bins=None, bandwidth=0.2, epsrel=1.49e-08, pool=False):
    if len(truth.shape) == 1:
        truth, l_mean, l_std = rearrange_arrays(truth, l_mean, l_std, num_radial_bins)

    dim_latent, num_radial_bins = l_mean.shape[1], truth.shape[1]
    mi_kls = np.zeros((nsamples, dim_latent, num_radial_bins))
    for i in range(nsamples):
        for j in range(dim_latent):
            samples = normalise(np.random.normal(l_mean[:, j], l_std[:, j], len(l_mean[:, j])))
            if pool is True:
                mi_kls[i, j, :] = pool_mi_sampleslatent_trutheachrbin(samples, truth, bandwidth, epsrel)
            else:
                mi_kls[i, j, :] = mi_samples_latent_vs_truth_rbins(samples, truth, bandwidth, epsrel)
    return mi_kls


def mi_samples_latent_vs_truth_rbins(samples_latents, truth, bandwidth=0.2, epsrel=1.49e-08):
    num_radial_bins = truth.shape[1]
    m = np.zeros(num_radial_bins, )
    for k in range(num_radial_bins):
        print("Radial bin " + str(k))
        t = normalise(truth[:, k])
        m[k] = mutual_information_cont(t, samples_latents, bandwidth=bandwidth, xlow=t.min(), xhigh=t.max(),
                                       ylow=samples_latents.min(), yhigh=samples_latents.max(), epsrel=epsrel)[0]
    return m


def pool_mi_sampleslatent_trutheachrbin(samples_latents, truth, bandwidth=0.2, epsrel=1.49e-08):
    num_radial_bins = truth.shape[1]
    p = Pool(min(num_radial_bins, multiprocessing.cpu_count()))
    args = list(zip(range(num_radial_bins), itertools.repeat(truth, num_radial_bins),
                    itertools.repeat(samples_latents, num_radial_bins), np.tile(bandwidth, num_radial_bins),
                    np.tile(epsrel, num_radial_bins)))
    m = p.map(mi_truth_samples_pool, args)
    return m


def mi_truth_samples_pool(args):
    k, truth, samples, bw, epsrel = args
    t = normalise(truth[:, k])
    mis = mutual_information_cont(t, samples, bandwidth=bw, xlow=t.min(), xhigh=t.max(), ylow=samples.min(),
                                  yhigh=samples.max(), epsrel=epsrel)[0]
    return mis


def normalise(X):
    return (X - np.mean(X))/np.std(X)


# Mutual information between latent variables

def KL_mi_between_latents(truth, l_mean, l_std, nsamples=50, num_radial_bins=16, bandwidth=0.2, epsrel=1.49e-08):
    assert all(l_mean[:num_radial_bins, 0] == l_mean[0, 0])
    lm, lstd = l_mean[::num_radial_bins, :], l_std[::num_radial_bins, :]
    dim_latent = l_mean.shape[1]

    mi_kls = np.zeros((nsamples, dim_latent, dim_latent))
    print(mi_kls.shape)
    for k in range(nsamples):
        samples = [normalise(np.random.normal(lm[:, j], lstd[:, j], len(lm[:, j]))) for j in range(dim_latent)]
        combos = np.unique(np.sort(list(itertools.product(np.arange(dim_latent), repeat=2))), axis=0)
        args = list(zip(combos, itertools.repeat(samples, len(combos)), np.tile(bandwidth, len(combos)), np.tile(epsrel, len(combos))))

        p = Pool(len(combos))
        mis = p.map(mi_between_latents_pool, args)
        p.close()
        p.join()
        assert len(mis) == len(combos)

        for num, n in enumerate(combos):
            i, j = n
            print(i, j)
            mi_kls[k, i, j] = mis[num]
            mi_kls[k, j, i] = mis[num]
    return mi_kls


def mi_between_latents_pool(args):
    combo, latent_samples, bw, epsrel = args
    i, j = combo
    a, b = latent_samples[i], latent_samples[j]
    return mutual_information_cont(a, b, bandwidth=bw, xlow=a.min(), xhigh=a.max(), ylow=b.min(), yhigh=b.max(), epsrel=epsrel)[0]


# Mutual information between latent variables conditional on ground truth

def mi_latents_conditioned_truth(truth, l_mean, l_std, nsamples=50, num_radial_bins=16, bandwidth=0.2,
                                 epsrel=1.49e-08, epsabs=1.49e-08):
    assert all(l_mean[:num_radial_bins, 0] == l_mean[0, 0])
    lm, lstd = l_mean[::num_radial_bins, :], l_std[::num_radial_bins, :]
    dim_latent = l_mean.shape[1]

    samples = [normalise(np.random.normal(lm[:, j], lstd[:, j], len(lm[:, j]))) for j in range(dim_latent)]
    combos = np.unique(np.sort(list(itertools.combinations(np.arange(dim_latent), 2))), axis=0)

    mi_kls = np.zeros((nsamples, dim_latent, dim_latent, num_radial_bins))
    for i in range(nsamples):
        for idx0, idx1 in combos:
            print(idx0, idx1)
            ts = [truth[k::num_radial_bins] for k in range(num_radial_bins)]
            args = list(zip(itertools.repeat(samples[idx0], num_radial_bins),
                            itertools.repeat(samples[idx1], num_radial_bins), ts, np.tile(bandwidth, num_radial_bins),
                            np.tile(epsrel, num_radial_bins), np.tile(epsabs, num_radial_bins)))

            p = Pool(num_radial_bins)
            mis = p.map(cmi_between_latents_conditioned_truth, args)
            p.close()
            p.join()
            mi_kls[i, idx0, idx1, :] = mis
            mi_kls[i, idx1, idx0, :] = mis

    return mi_kls


def cmi_between_latents_conditioned_truth(args):
    latent0, latent1, truthi, bandwidth, epsrel, absrel = args
    xlim, ylim, zlim = (latent0.min(), latent0.max()), (latent1.min(), latent1.max()), (truthi.min(), truthi.max())
    cmi = conditional_mutual_information_cont(latent0, latent1, truthi, bandwidth=bandwidth, xrange=xlim, yrange=ylim,
                                              zrange=zlim, epsrel=epsrel, epsabs=absrel)[0]
    return cmi


# Mutual information between latent variables conditional on ground truth

def total_mi_latents_truth(truth, l_mean, l_std, nsamples=50, num_radial_bins=16, bandwidth=0.2,
                           epsrel=1.49e-08, epsabs=1.49e-08):
    assert all(l_mean[:num_radial_bins, 0] == l_mean[0, 0])
    lm, lstd = l_mean[::num_radial_bins, :], l_std[::num_radial_bins, :]
    dim_latent = l_mean.shape[1]

    samples = [normalise(np.random.normal(lm[:, j], lstd[:, j], len(lm[:, j]))) for j in range(dim_latent)]
    truths = [truth[k::num_radial_bins] for k in range(num_radial_bins)]
    if dim_latent > 2:
        raise ValueError("Only implemented for latent dim 2, yours is " + print(dim_latent))
    else:
        mi_kls = np.zeros((nsamples, num_radial_bins))
        for i in range(nsamples):
            args = list(zip(itertools.repeat(samples[0], num_radial_bins),
                            itertools.repeat(samples[1], num_radial_bins),
                            truths, np.tile(bandwidth, num_radial_bins),
                            np.tile(epsrel, num_radial_bins), np.tile(epsabs, num_radial_bins)))

            p = Pool(num_radial_bins)
            mis = p.map(totalmi_between_latents_and_truth, args)
            p.close()
            p.join()
            mi_kls[i, :] = mis
            mi_kls[i, :] = mis

        return mi_kls


def totalmi_between_latents_and_truth(args):
    latent0, latent1, truthi, bandwidth, epsrel, absrel = args
    xlim, ylim, zlim = (latent0.min(), latent0.max()), (latent1.min(), latent1.max()), (truthi.min(), truthi.max())
    cmi = total_mutual_information_cont(latent0, latent1, truthi, bandwidth=bandwidth, xrange=xlim, yrange=ylim,
                                        zrange=zlim, epsrel=epsrel, epsabs=absrel)[0]
    return cmi


# Conditional mutual information: MI between X and Y conditioned on variable Z

def conditional_mutual_information_cont(x, y, z, bandwidth=0.1, method=None, N=100000, xrange=(9, 16), yrange=(9, 16),
                                        zrange=(9, 16), epsrel=1.49e-08, epsabs=1.49e-08):
    """  Mutual information of two continuous variables X and Y, conditioned on variable Z """
    pxyz = kde3D(x, y, z, bandwidth)
    pxz = kde2D(x, z, bandwidth)
    pyz = kde2D(y, z, bandwidth)
    pz = kde1D(z, bandwidth)
    print("Fitted 3D kde to joint distribution of x, y, z,  1D kdes for z and 2D kdes for x,z and y,z distributions")
    if method == "MC":
        mi = cond_MI_MC_integration(pxyz, pxz, pyz, pz, N=N)
    else:
        xlow, xhigh, ylow, yhigh, zlow, zhigh = xrange[0], xrange[1], yrange[0], yrange[1], zrange[0], zrange[1]
        mi = cond_MI_integration(pxyz, pxz, pyz, pz, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh,
                                 zlow=zlow, zhigh=zhigh, epsrel=epsrel, epsabs=epsabs)
    return mi


def integrand_conditional_MI(fitted_pxyz, fitted_pxz, fitted_pyz, fitted_pz, x, y, z):
    pxyz = evaluate_kde3d(fitted_pxyz, x, y, z)
    pz = evaluate_kde1d(fitted_pz, z)
    pxzpyz = evaluate_kde2d(fitted_pxz, x, z) * evaluate_kde2d(fitted_pyz, y, z)
    if pxzpyz == 0:
        print("Product of marginals is 0.. KL divergence is undefined.")
    return pxyz * np.log(pxyz * pz/pxzpyz, where=pxyz != 0)


def cond_MI_integration(fitted_pxyz, fitted_pxz, fitted_pyz, fitted_pz,
                        xlow=9, xhigh=16, ylow=9, yhigh=16, zlow=9, zhigh=16, epsrel=1.49e-08, epsabs=1.49e-08):
    func2 = lambda x, y, z: integrand_conditional_MI(fitted_pxyz, fitted_pxz, fitted_pyz, fitted_pz, x, y, z)
    return integrate.tplquad(func2, zlow, zhigh, lambda z: ylow, lambda z: yhigh, lambda z, y: xlow, lambda z, y: xhigh,
                             epsrel=epsrel, epsabs=epsabs)


def cond_MI_MC_integration(fitted_pxyz, fitted_pxz, fitted_pyz, fitted_pz, N=100000):
    samples = fitted_pxyz.sample(N)
    x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]
    pxyz = evaluate_kde3d(fitted_pxyz, x, y, z)
    pz = evaluate_kde1d(fitted_pz, z)
    pxzpyz = evaluate_kde2d(fitted_pxz, x, z) * evaluate_kde2d(fitted_pyz, y, z)
    integrand = np.log(pxyz * pz/pxzpyz, where=pxyz != 0)
    return np.mean(integrand)


# Total mutual information: MI between X and Y conditioned on variable Z

def total_mutual_information_cont(x, y, z, bandwidth=0.1, xrange=(9, 16), yrange=(9, 16), zrange=(9, 16),
                                  epsrel=1.49e-08, epsabs=1.49e-08):
    """  Total information of between 2D space (X,Y) and 1D variable Z. """
    pxyz = kde3D(x, y, z, bandwidth)
    pxy = kde2D(x, y, bandwidth)
    pz = kde1D(z, bandwidth)
    print("Fitted 3D kde to joint distribution of x, y, z,  1D kdes for z and 2D kdes for x,z and y,z distributions")
    xlow, xhigh, ylow, yhigh, zlow, zhigh = xrange[0], xrange[1], yrange[0], yrange[1], zrange[0], zrange[1]
    mi = total_MI_integration(pxyz, pxy, pz, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh,  zlow=zlow, zhigh=zhigh,
                              epsrel=epsrel, epsabs=epsabs)
    return mi


def integrand_total_MI(fitted_pxyz, fitted_pxy, fitted_pz, x, y, z):
    pxyz = evaluate_kde3d(fitted_pxyz, x, y, z)
    pxy = evaluate_kde2d(fitted_pxy, x, y)
    pz = evaluate_kde1d(fitted_pz, z)
    if pxy * pz == 0:
        print("Product of marginals is 0.. KL divergence is undefined.")
    return pxyz * np.log(pxyz/(pxy * pz), where=pxyz != 0)


def total_MI_integration(fitted_pxyz, fitted_pxy, fitted_pz, xlow=9, xhigh=16, ylow=9, yhigh=16, zlow=9, zhigh=16,
                         epsrel=1.49e-08, epsabs=1.49e-08):
    func2 = lambda x, y, z: integrand_total_MI(fitted_pxyz, fitted_pxy, fitted_pz, x, y, z)
    return integrate.tplquad(func2, zlow, zhigh, lambda z: ylow, lambda z: yhigh, lambda z, y: xlow, lambda z, y: xhigh,
                             epsrel=epsrel, epsabs=epsabs)


# Kernel density estimation


def kde1D(x, bandwidth, **kwargs):
    """Build 1D kernel density estimate (KDE)."""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl = kde_skl.fit(x.reshape(-1, 1))
    return kde_skl


def evaluate_kde1d(fitted_kde, xx):
    """Evaluate fitted 1D KDE at position xx."""
    z = np.exp(fitted_kde.score_samples(np.array([xx]).reshape(-1, 1)))
    return z


def kde2D(x, y, bandwidth, **kwargs):
    """Build 2D kernel density estimate (KDE)."""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(np.vstack([y, x]).T)
    return kde_skl


def evaluate_kde2d(fitted_kde, xx, yy):
    """Evaluate fitted 2D KDE at position xx, yy."""
    xy_sample = np.vstack([yy, xx]).T
    z = np.exp(fitted_kde.score_samples(xy_sample))
    return z


def kde3D(x, y, z, bandwidth, **kwargs):
    """Build 3D kernel density estimate (KDE)."""
    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(np.vstack([z, y, x]).T)
    return kde_skl


def evaluate_kde3d(fitted_kde, xx, yy, zz, pool=False):
    """Evaluate fitted 3D KDE at position xx, yy, zz."""
    xy_sample = np.vstack([zz, yy, xx]).T
    z = np.exp(fitted_kde.score_samples(xy_sample))
    return z


# Compute Mutual information between X and Y using quad integration method

def mutual_information_cont(x, y, bandwidth=0.1, xlow=9, xhigh=16, ylow=9, yhigh=16, epsrel=1.49e-08, epsabs=1.49e-08):
    # KL divergence between joint probability distribution p(x, y) and product of marginal p(x)p(y)
    pxy = kde2D(x, y, bandwidth)
    # print("Fitted 2D kde to joint distributions")
    px = kde1D(x, bandwidth)
    py = kde1D(y, bandwidth)
    # print("Fitted 1D kdes for each marginal distributions")
    return KL_div_continuous(pxy, px, py, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh, epsrel=epsrel, epsabs=epsabs)


def integrand(fitted_pxy, fitted_px, fitted_py, x, y):
    pxy = evaluate_kde2d(fitted_pxy, x, y)
    px = evaluate_kde1d(fitted_px, x)
    py = evaluate_kde1d(fitted_py, y)
    pxpy = px * py
    if pxpy == 0:
        print("Product of marginals is 0.. KL divergence is undefined.")
    return pxy * np.log(pxy/pxpy, where=pxy != 0)


def KL_div_continuous(fitted_pxy, fitted_px, fitted_py, xlow=9, xhigh=16, ylow=9, yhigh=16, epsrel=1.49e-08, epsabs=1.49e-08):
    func = lambda x, y: integrand(fitted_pxy, fitted_px, fitted_py, x, y)
    return integrate.dblquad(func, ylow, yhigh, lambda x: xlow, lambda x: xhigh, epsrel=epsrel, epsabs=epsabs)


# Compute Mutual information between X and Y using Romberg integration method

def mutual_information_cont_romberg(x, y, bandwidth=0.1, xlow=9, xhigh=16, ylow=9, yhigh=16):
    pxy = kde2D(x, y, bandwidth)
    px = kde1D(x, bandwidth)
    py = kde1D(y, bandwidth)
    print("Fitted 2D kde to joint distribution and 1D kdes for each marginal distributions")
    return KL_div_continuous_romberg(pxy, px, py, xlow=xlow, xhigh=xhigh, ylow=ylow, yhigh=yhigh)


def KL_div_continuous_romberg(fitted_pxy, fitted_px, fitted_py, xlow=9, xhigh=16, ylow=9, yhigh=16):
    sol_int = integrate.romberg(lambda x: integrate.romberg(lambda y: integrand(fitted_pxy, fitted_px, fitted_py, x, y), xlow, xhigh)[0], ylow, yhigh)
    return sol_int


# Compute Mutual information between X and Y using Monte-Carlo integration method

def mutual_information_cont_MC(x, y, bandwidth=0.1, N=100000):
    # KL divergence between joint probability distribution p(x, y) and
    pxy = kde2D(x, y, bandwidth)
    px = kde1D(x, bandwidth)
    py = kde1D(y, bandwidth)
    print("Fitted 2D kde to joint distribution and 1D kdes for each marginal distributions")
    return KL_div_MC(pxy, px, py, N=N)


def KL_div_MC(fitted_pxy, fitted_px, fitted_py, N=100000):
    samples = fitted_pxy.sample(N)
    x, y = samples[:, 0], samples[:, 1]
    pxy = evaluate_kde2d(fitted_pxy, x, y)
    px = evaluate_kde1d(fitted_px, x)
    py = evaluate_kde1d(fitted_py, y)
    nsz = pxy != 0
    integral = np.log(pxy[nsz]/(px[nsz]*py[nsz]))
    # might have to select only the positive ones, which is a fudge indeed...
    return np.mean(integral)


# Plot 2D kernel density estimate distribution

def plot_kde2D(x, y, bandwidth, xbins=100j, ybins=100j, **kwargs):
    """Build 2D kernel density estimate (KDE)."""

    # create grid of sample locations (default: 100x100)
    xx, yy = np.mgrid[x.min():x.max():xbins,
                      y.min():y.max():ybins]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T
    xy_train = np.vstack([y, x]).T

    kde_skl = KernelDensity(bandwidth=bandwidth, **kwargs)
    kde_skl.fit(xy_train)

    # score_samples() returns the log-likelihood of the samples
    z = np.exp(kde_skl.score_samples(xy_sample))
    return xx, yy, np.reshape(z, xx.shape)


def latents_corner_plots(latents, title=None, save=None, mi=None):
    # latents should be a list of arrays, where each array is a set of samples from one latent variable
    l0 = np.vstack(latents).T
    assert np.allclose(l0[:, 0], latents[0])
    num_latents = l0.shape[1]
    pd_latents = pd.DataFrame(np.vstack(latents).T, columns=["Latent " + str(i) + "" for i in range(num_latents)])
    g = sns.pairplot(pd_latents, corner=True, plot_kws=dict(marker="+", linewidth=0.1))
    if mi is not None:
        for i,j in [(1, 0), (2, 1), (2, 0)]:
            g.axes[i][j].text(g.axes[i][j].get_xlim()[0]+0.5, g.axes[i][j].get_ylim()[1]-0.5, "MI = %.1e" % mi[i, j])
    if title is not None:
        g.fig.suptitle(title)
    if save is not None:
        plt.savefig(save)
    return g


# Discrete mutual information

def mutual_information_discrete(x, y, bins=100):
    """ Mutual information of two discrete variables """
    hist_2d, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    # Convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
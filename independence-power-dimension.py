#!/usr/bin/env python
# coding: utf-8

# Power vs. Dimension for 20 Relationships


import os
import sys

import numpy as np
from joblib import Parallel, delayed

from hyppo.independence import MGC, Dcorr, Hsic, HHG, CCA, RV
from kmerf import KMERF
from simulations import make_independence_simulation, INDEPENDENCE_SIMS


sys.path.append(os.path.realpath(".."))


DIMENSIONS = range(3, 11)
SAMP_SIZE = 100
REPS = range(1000)


SAVE_PATH = "independence-n-{}_p-3_1000".format(int(SAMP_SIZE))


TESTS = {
    "KMERF": KMERF(forest="regressor"),
    "MGC": MGC(),
    "Dcorr": Dcorr(),
    "Hsic": Hsic(),
    "HHG": HHG(),
    "CCA": CCA(),
    "RV": RV(),
}


def _find_dim_range(sim):
    dim = 20
    if sim in ["linear", "exponential", "cubic"]:
        dim = 1000
    elif sim in [
        "joint_normal",
        "sin_four_pi",
        "sin_sixteen_pi",
        "multiplicative_noise",
    ]:
        dim = 10
    elif sim in [
        "uncorrelated_bernoulli",
        "logarithmic",
        "multimodal_independence",
    ]:
        dim = 100
    elif sim in ["square", "diamond"]:
        dim = 40
    return np.linspace(3, dim, 10, dtype=int) if dim > 10 else range(3, 11)


def _sim_slice(X, p):
    """
    Generate x, y from each sim
    """
    X_t = X[:, :p]
    y_t = np.concatenate((np.zeros(SAMP_SIZE // 2), np.ones(SAMP_SIZE // 2)))
    return X_t, y_t


def _perm_stat(est, X, p):
    """
    Generates null and alternate distributions
    """
    X, y = _sim_slice(X, p)
    y = y.reshape(-1, 1)
    obs_stat = est.statistic(X, y)
    permy = np.random.permutation(y)
    perm_stat = est.statistic(X, permy)

    return obs_stat, perm_stat


def _nonperm_pval(est, X, p):
    """
    Generates fast  permutation pvalues
    """
    X, y = _sim_slice(X, p)
    pvalue = est.test(X, y)[1]
    return pvalue


def compute_null(rep, est, est_name, sim, p=1):
    """
    Calculates empirical null and alternate distribution for each test.
    """
    dim_range = _find_dim_range(sim)
    X, _ = make_independence_simulation(
        n_samples=SAMP_SIZE,
        n_dim=dim_range[-1],
        simulation=sim,
        noise=False,
        seed=rep,
    )
    if est_name in ["Dcorr", "Hsic"]:
        pval = _nonperm_pval(est, X, p)
        save_kwargs = {"X": [pval]}
    else:
        alt_dist, null_dist = _perm_stat(est, X, p)
        save_kwargs = {"X": [alt_dist, null_dist], "delimiter": ","}
    np.savetxt(
        "{}/{}_{}_{}_{}.txt".format(SAVE_PATH, sim, est_name, p, rep), **save_kwargs
    )


# Run this block to regenerate power curves. Note that this takes a very long time!
_ = Parallel(n_jobs=-1, verbose=100)(
    [
        delayed(compute_null)(rep, est, est_name, sim, p=dim)
        for rep in REPS
        for est_name, est in TESTS.items()
        for sim in INDEPENDENCE_SIMS
        for dim in _find_dim_range(sim)
    ]
)

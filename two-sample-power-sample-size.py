#!/usr/bin/env python
# coding: utf-8

# Power vs. Sample Size for 15 Relationships


import os
import sys

import numpy as np
from joblib import Parallel, delayed

from hyppo.independence import MGC, Dcorr, Hsic, HHG, CCA, RV
from kmerf import KMERF
from simulations import make_marron_wand_classification, MARRON_WAND_SIMS


sys.path.append(os.path.realpath(".."))


SAMP_SIZES = range(10, 110, 10)
DIMENSION = 10
REPS = range(1000)


SAVE_PATH = "two-sample-p-{}_n-{}_{}".format(
    int(DIMENSION), int(SAMP_SIZES[0]), int(SAMP_SIZES[-1])
)


TESTS = {
    #"KMERF": KMERF(forest="classifier"),
    #"MGC": MGC(),
    "Dcorr": Dcorr(),
    "Hsic": Hsic(),
    #"HHG": HHG(),
    #"CCA": CCA(),
    #"RV": RV(),
}


def _sim_slice(X, n):
    """
    Generate x, y from each sim
    """
    X_t = np.concatenate(
        (X[: n // 2], X[SAMP_SIZES[-1] // 2 : SAMP_SIZES[-1] // 2 + n // 2])
    )
    y_t = np.concatenate((np.zeros(n // 2), np.ones(n // 2)))
    return X_t, y_t


def _perm_stat(est, X, n):
    """
    Generates null and alternate distributions
    """
    X, y = _sim_slice(X, n)
    y = y.reshape(-1, 1)
    obs_stat = est.statistic(X, y)
    permy = np.random.permutation(y)
    perm_stat = est.statistic(X, permy)

    return obs_stat, perm_stat


def _nonperm_pval(est, X, n):
    """
    Generates fast  permutation pvalues
    """
    X, y = _sim_slice(X, n)
    pvalue = est.test(X, y)[1]
    return pvalue


def compute_null(rep, est, est_name, sim, n=100):
    """
    Calculates empirical null and alternate distribution for each test.
    """
    X, _ = make_marron_wand_classification(
        n_samples=SAMP_SIZES[-1],
        n_dim=DIMENSION,
        n_informative=1,
        simulation=sim,
        seed=rep,
    )
    if est_name in ["Dcorr", "Hsic"]:
        try:
            pval = _nonperm_pval(est, X, n)
        except ValueError:
            pval = 1
        save_kwargs = {"X": [pval]}
    else:
        alt_dist, null_dist = _perm_stat(est, X, n)
        save_kwargs = {"X": [alt_dist, null_dist], "delimiter": ","}
    np.savetxt(
        "{}/{}_{}_{}_{}.txt".format(SAVE_PATH, sim, est_name, n, rep), **save_kwargs
    )


# Run this block to regenerate power curves. Note that this takes a very long time!
_ = Parallel(n_jobs=-1, verbose=100)(
    [
        delayed(compute_null)(rep, est, est_name, sim, n=samp_size)
        for rep in REPS
        for est_name, est in TESTS.items()
        for sim in MARRON_WAND_SIMS.keys()
        for samp_size in SAMP_SIZES
    ]
)

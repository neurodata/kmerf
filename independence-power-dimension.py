#!/usr/bin/env python
# coding: utf-8

# Power vs. Dimension for 20 Relationships


import os
import sys

import numpy as np
from joblib import Parallel, delayed

from hyppo.independence import MGC, Dcorr, Hsic, HHG, CCA, RV
from kmerf import KMERF
from simulations import indep_sim, INDEPENDENCE_SIMS, _find_dim_range


sys.path.append(os.path.realpath(".."))


DIMENSIONS = range(3, 11)
SAMP_SIZE = 100
REPS = range(1000)


SAVE_PATH = "independence-n-{}_p-3_1000".format(int(SAMP_SIZE))


TESTS = {
    "KMERF": KMERF(forest="regressor"),
    #"MGC": MGC(),
    "Dcorr": Dcorr(),
    "Hsic": Hsic(),
    #"HHG": HHG(),
    #"CCA": CCA(),
    #"RV": RV(),
}


def _indep_sim_gen(sim, n, p, noise=False):
    """
    Generate x, y from each sim
    """
    if sim in ["multiplicative_noise", "multimodal_independence"]:
        x, y = indep_sim(sim, n, p)
    else:
        x, y = indep_sim(sim, n, p, noise=noise)

    return x, y


def _perm_stat(est, sim, n=100, p=1, noise=False):
    """
    Generates null and alternate distributions
    """
    X, y = _indep_sim_gen(sim, n, p, noise=noise)
    obs_stat = est.statistic(X, y)
    permy = np.random.permutation(y)
    perm_stat = est.statistic(X, permy)

    return obs_stat, perm_stat


def _nonperm_pval(est, sim, n=100, p=1, noise=False):
    """
    Generates fast  permutation pvalues
    """
    X, y = _indep_sim_gen(sim, n, p, noise=noise)
    pvalue = est.test(X, y)[1]
    return pvalue


def compute_null(rep, est, est_name, sim, p=1):
    """
    Calculates empirical null and alternate distribution for each test.
    """
    if est_name in ["Dcorr", "Hsic"]:
        pval = _nonperm_pval(est, sim, p=p)
        save_kwargs = {"X": [pval]}
    else:
        alt_dist, null_dist = _perm_stat(est, sim, p=p)
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
        for sim in INDEPENDENCE_SIMS.keys()
        for dim in _find_dim_range(sim)
    ]
)

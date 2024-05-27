import math

import numpy as np
from joblib import Parallel, delayed

from simulations import MARRON_WAND_SIMS


TESTS = [
    "KMERF",
    "MGC",
    #"Dcorr",
    #"Hsic",
    "HHG",
    "CCA",
    "RV",
]


def refactor_data_power(
    alg="Dcorr",
    sim="linear",
    alpha=0.05,
    max_reps=1000,
    fig_name="two-sample-power-vs-d",
):
    type = fig_name[-1]
    if type == "d":
        file_path = "n-100_p-3_10"
        sample_dimensions = range(3, 11)
    elif type == "n":
        file_path = "p-3_n-10_100"
        sample_dimensions = range(10, 110, 10)
    else:
        raise ValueError("Invalid type")
    power = np.empty(len(sample_dimensions))

    for i, dim in enumerate(sample_dimensions):
        if alg in ["Dcorr", "Hsic"]:
            pvalues = []
            for rep in range(max_reps):
                try:
                    pvalue = np.genfromtxt(f"{file_path}/{sim}_{alg}_{dim}_{rep}.txt")
                except FileNotFoundError:
                    break
                pvalues.append(pvalue)
            empirical_power = (1 + (np.array(pvalues) <= alpha).sum()) / (
                1 + len(pvalues)
            )
        else:
            alt_dist, null_dist = [], []
            for rep in range(max_reps):
                try:
                    alt_data, null_data = np.genfromtxt(
                        f"{file_path}/{sim}_{alg}_{dim}_{rep}.txt"
                    )
                except FileNotFoundError:
                    break
                alt_dist.append(alt_data)
                null_dist.append(null_data)
            try:
                cutoff = np.sort(null_dist)[math.ceil(len(null_dist) * (1 - alpha))]
            except:
                print(alg, sim, fig_name)
            empirical_power = (1 + (np.array(alt_dist) >= cutoff).sum()) / (
                1 + len(alt_dist)
            )
        power[i] = empirical_power
    np.savetxt(f"{fig_name}/{sim}-{alg}-{fig_name}.csv", power, delimiter=",")


_ = Parallel(n_jobs=-1, verbose=100)(
    [
        delayed(refactor_data_power)(
            alg=alg, fig_name=fig_name, sim=sim, alpha=0.05, max_reps=10000
        )
        for alg in TESTS
        for fig_name in ["two-sample-power-vs-d", "two-sample-power-vs-n"]
        for sim in MARRON_WAND_SIMS.keys()
    ]
)

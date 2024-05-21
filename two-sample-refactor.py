import math

import numpy as np
from joblib import Parallel, delayed

from simulations import MARRON_WAND_SIMS


TESTS = [
    "KMERF",
    "MGC",
    "Dcorr",
    "Hsic",
    "HHG",
    "CCA",
    "RV",
]


def refactor_data_power(
    alg="Dcorr", sim="linear", type="d", alpha=0.05, max_reps=10000
):
    if type == "d":
        file_path = "n-100_p-4_1024"
        sample_dimensions = [2**i for i in range(2, 11)]
        power = np.empty(len(sample_dimensions))
    else:
        raise ValueError("Invalid type")

    for i, dim in enumerate(sample_dimensions):
        if alg in ["KMERF", "Dcorr", "Hsic"]:
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
            cutoff = np.sort(null_dist)[math.ceil(len(null_dist) * (1 - alpha))]
            empirical_power = (1 + (np.array(alt_dist) >= cutoff).sum()) / (
                1 + len(alt_dist)
            )
        power[i] = empirical_power
    np.savetxt(f"results/{sim}-{alg}-power-vs-{type}.csv", power, delimiter=",")


_ = Parallel(n_jobs=-1, verbose=100)(
    [
        delayed(refactor_data_power)(
            alg=alg, type=type, sim=sim, alpha=0.05, max_reps=10000
        )
        for alg in TESTS
        for type in ["d"]
        for sim in MARRON_WAND_SIMS.keys()
    ]
)

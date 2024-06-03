import math

import numpy as np
from joblib import Parallel, delayed

from simulations import MARRON_WAND_SIMS, INDEPENDENCE_SIMS, _find_dim_range


SIMULATIONS = {
    "two-sample-power": {"sim": MARRON_WAND_SIMS.keys(), "max_reps": 1000},
    "independence-power": {"sim": INDEPENDENCE_SIMS, "max_reps": 10000},
}

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
    alg="Dcorr",
    sim="linear",
    alpha=0.05,
    max_reps=1000,
    fig_name="two-sample-power-vs-d",
):
    FAST_ALGS = []
    type = fig_name[-1]
    if "two-sample" in fig_name and type == "d":
        FAST_ALGS = ["Dcorr", "Hsic"]
        file_path = "two-sample-n-100_p-3_10"
        sample_dimensions = range(3, 11)
    elif "independence" in fig_name and type == "d":
        FAST_ALGS = ["Dcorr", "Hsic"]
        file_path = "independence-n-100_p-3_1000"
        sample_dimensions = _find_dim_range(sim)
    elif type == "n":
        if "two-sample" in fig_name:
            file_path = "two-sample"
        elif "independence" in fig_name:
            file_path = "independence"
        file_path += "-p-10_n-10_100"
        sample_dimensions = range(10, 110, 10)
    else:
        raise ValueError(
            f"fig_name is {fig_name}; must contain two-sample or independence and"
            "end in d or n"
        )

    power = np.empty(len(sample_dimensions))
    for i, samp_dim in enumerate(sample_dimensions):
        if alg in FAST_ALGS:
            pvalues = []
            for rep in range(max_reps):
                try:
                    pvalue = np.genfromtxt(
                        f"{file_path}/{sim}_{alg}_{samp_dim}_{rep}.txt"
                    )
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
                        f"{file_path}/{sim}_{alg}_{samp_dim}_{rep}.txt"
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
    np.savetxt(f"{fig_name}/{sim}-{alg}-{fig_name}.csv", power, delimiter=",")


_ = Parallel(n_jobs=-1, verbose=100)(
    [
        delayed(refactor_data_power)(
            alg=alg,
            fig_name=fig_name,
            sim=sim,
            alpha=0.05,
            max_reps=SIMULATIONS[fig_name[:-5]]["max_reps"],
        )
        for alg in TESTS
        for fig_name in [
            "two-sample-power-vs-d",
            "two-sample-power-vs-n",
            "independence-power-vs-n",
            "independence-power-vs-d",
        ]
        for sim in SIMULATIONS[fig_name[:-5]]["sim"]
    ]
)

import numpy as np

# Dictionary of simulations from Marron and Wand 1992
# keys: names of each simulation corresponding to the class MarronWandSims
# values: probabilities associated with the mixture of Gaussians
MARRON_WAND_SIMS = {
    "skewed_unimodal": [1 / 5, 1 / 5, 3 / 5],
    "strongly_skewed": [1 / 8] * 8,
    "kurtotic_unimodal": [2 / 3, 1 / 3],
    "outlier": [1 / 10, 9 / 10],
    "bimodal": [1 / 2] * 2,
    "separated_bimodal": [1 / 2] * 2,
    "skewed_bimodal": [3 / 4, 1 / 4],
    "trimodal": [9 / 20, 9 / 20, 1 / 10],
    "claw": [1 / 2, *[1 / 10] * 5],
    "double_claw": [49 / 100, 49 / 100, *[1 / 350] * 7],
    "asymmetric_claw": [1 / 2, *[2 ** (1 - i) / 31 for i in range(-2, 3)]],
    "asymmetric_double_claw": [*[46 / 100] * 2, *[1 / 300] * 3, *[7 / 300] * 3],
    "smooth_comb": [2 ** (5 - i) / 63 for i in range(6)],
    "discrete_comb": [*[2 / 7] * 3, *[1 / 21] * 3],
    "independent": [],
}


def _moving_avg_cov(n_dim, rho):
    # Create a meshgrid of indices
    i, j = np.meshgrid(np.arange(1, n_dim + 1), np.arange(1, n_dim + 1), indexing="ij")

    # Calculate the covariance matrix using the corrected formula
    cov_matrix = rho ** np.abs(i - j)

    # Apply the banding condition
    cov_matrix[abs(i - j) > 1] = 0
    return cov_matrix


def _autoregressive_cov(n_dim, rho):
    # Create a meshgrid of indices
    i, j = np.meshgrid(np.arange(1, n_dim + 1), np.arange(1, n_dim + 1), indexing="ij")

    # Calculate the covariance matrix using the corrected formula
    cov_matrix = rho ** np.abs(i - j)

    return cov_matrix


def make_marron_wand_classification(
    n_samples,
    n_dim=4096,
    n_informative=256,
    simulation: str = "independent",
    rho: int = 0,
    band_type: str = "ma",
    return_params: bool = False,
    scaling_factor: float = 1.0,
    seed=None,
):
    """Generate Marron-Wand binary classification dataset.

    The simulation is similar to that of
    :func:`sktree.datasets.make_trunk_classification`
    where the first class is generated from a multivariate-Gaussians with mean vector of
    0's. The second class is generated from a mixture of Gaussians with mean vectors
    specified by the Marron-Wand simulations, but as the dimensionality increases,
    the second class distribution approaches the first class distribution by a factor
    of :math:`1 / sqrt(d)`.

    Full details for the Marron-Wand simulations can be found in
    :footcite:`marron1992exact`.

    Instead of the identity covariance matrix, one can implement a banded covariance
    matrix that follows :footcite:`Bickel_2008`.

    Parameters
    ----------
    n_samples : int
        Number of sample to generate. Must be an even number, else the total number of
        samples generated will be ``n_samples - 1``.
    n_dim : int, optional
        The dimensionality of the dataset and the number of
        unique labels, by default 4096.
    n_informative : int, optional
        The informative dimensions. All others for ``n_dim - n_informative``
        are Gaussian noise. Default is 256.
    simulation : str, optional
        Which simulation to run. Must be one of the
        following Marron-Wand simulations: 'independent', 'skewed_unimodal',
        'strongly_skewed', 'kurtotic_unimodal', 'outlier', 'bimodal',
        'separated_bimodal', 'skewed_bimodal', 'trimodal', 'claw', 'double_claw',
        'asymmetric_claw', 'asymmetric_double_claw', 'smooth_comb', 'discrete_comb'.
        When calling the Marron-Wand simulations, only the covariance parameters are
        considered (`rho` and `band_type`). Means are taken from
        :footcite:`marron1992exact`. By default 'independent'.
    rho : float, optional
        The covariance value of the bands. By default 0 indicating, an identity matrix
        is used.
    band_type : str
        The band type to use. For details, see Example 1 and 2 in
        :footcite:`Bickel_2008`. Either 'ma', or 'ar'.
    return_params : bool, optional
        Whether or not to return the distribution parameters of the classes normal
        distributions.
    scaling_factor : float, optional
        The scaling factor for the covariance matrix. By default 1.
    seed : int, optional
        Random seed, by default None.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_dim), dtype=np.float64
        Trunk dataset as a dense array.
    y : np.ndarray of shape (n_samples,), dtype=np.intp
        Labels of the dataset.
    G : np.ndarray of shape (n_samples, n_dim), dtype=np.float64
        The mixture of Gaussians for the Marron-Wand simulations.
        Returned if ``return_params`` is True.
    w : np.ndarray of shape (n_dim,), dtype=np.float64
        The weight vector for the Marron-Wand simulations.
        Returned if ``return_params`` is True.

    Notes
    -----
    **Marron-Wand Simulations**: The Marron-Wand simulations generate two classes of
    data with the setup specified in the paper.

    Covariance: The covariance matrix among different dimensions is controlled by the
    ``rho`` parameter and the ``band_type`` parameter. The ``band_type`` parameter
    controls the type of band to use, while the ``rho`` parameter controls the specific
    scaling factor for the covariance matrix while going from one dimension to the next.

    For each dimension in the first distribution, there is a mean of :math:`1 / d`,
    where ``d`` is the dimensionality. The covariance is the identity matrix.

    The second distribution has a mean vector that is the negative of the first.
    As ``d`` increases, the two distributions become closer and closer.
    Full details for the trunk simulation can be found in :footcite:`trunk1982`.

    References
    ----------
    .. footbibliography::
    """
    if simulation not in MARRON_WAND_SIMS.keys():
        raise ValueError(f"Simulation must be: {MARRON_WAND_SIMS.keys()}")

    rng = np.random.default_rng(seed=seed)

    # speed up computations for large multivariate normal matrix with SVD approximation
    if n_informative > 1000:
        mvg_sampling_method = "cholesky"
    else:
        mvg_sampling_method = "svd"

    if simulation == "independent":
        X = np.vstack(
            (
                rng.multivariate_normal(
                    np.zeros(n_dim),
                    np.identity(n_dim),
                    n_samples // 2,
                    method=mvg_sampling_method,
                ),
                rng.multivariate_normal(
                    np.zeros(n_dim),
                    np.identity(n_dim),
                    n_samples // 2,
                    method=mvg_sampling_method,
                ),
            )
        )
    else:
        if n_dim < n_informative:
            raise ValueError(
                f"Number of informative dimensions {n_informative} must be less than"
                f" number of dimensions, {n_dim}"
            )
        if rho != 0:
            if band_type == "ma":
                cov = _moving_avg_cov(n_informative, rho)
            elif band_type == "ar":
                cov = _autoregressive_cov(n_informative, rho)
            else:
                raise ValueError(f'Band type {band_type} must be one of "ma", or "ar".')
        else:
            cov = np.identity(n_informative)

        # allow arbitrary uniform scaling of the covariance matrix
        cov = scaling_factor * cov

        mixture_idx = rng.choice(
            len(MARRON_WAND_SIMS[simulation]),  # type: ignore
            size=n_samples // 2,
            replace=True,
            p=MARRON_WAND_SIMS[simulation],
        )
        # the parameters used for each Gaussian in the mixture for each Marron Wand
        # simulation
        norm_params = MarronWandSims(n_dim=n_informative, cov=cov)(simulation)
        G = np.fromiter(
            (
                rng_children.multivariate_normal(
                    *(norm_params[i]), size=1, method=mvg_sampling_method
                )
                for i, rng_children in zip(mixture_idx, rng.spawn(n_samples // 2))
            ),
            dtype=np.dtype((float, n_informative)),
        )

        # as the dimensionality of the simulations increasing, we are adding more and
        # more noise to the data using the w parameter
        w_vec = np.array([1.0 / np.sqrt(i) for i in range(1, n_informative + 1)])

        # create new generator instance to ensure reproducibility with multiple runs
        # with the same seed
        rng_F = np.random.default_rng(seed=seed).spawn(2)

        X = np.vstack(
            (
                rng_F[0].multivariate_normal(
                    np.zeros(n_informative),
                    cov,
                    n_samples // 2,
                    method=mvg_sampling_method,
                ),
                (1 - w_vec)
                * rng_F[1].multivariate_normal(
                    np.zeros(n_informative),
                    cov,
                    n_samples // 2,
                    method=mvg_sampling_method,
                )
                + w_vec * G.reshape(n_samples // 2, n_informative),
            )
        )

        if n_dim > n_informative:
            # create new generator instance to ensure reproducibility with multiple
            # runs with the same seed
            rng_noise = np.random.default_rng(seed=seed)
            X = np.hstack(
                (
                    X,
                    np.hstack(
                        [
                            rng_children.normal(loc=0, scale=1, size=(X.shape[0], 1))
                            for rng_children in rng_noise.spawn(n_dim - n_informative)
                        ]
                    ),
                )
            )

    y = np.concatenate((np.zeros(n_samples // 2), np.ones(n_samples // 2)))

    if return_params and not simulation == "independent":
        return [X, y, *list(zip(*norm_params)), G, w_vec]
    return X, y


class MarronWandSims:
    def __init__(self, n_dim=1, cov=1):
        self.n_dim = n_dim
        self.cov = cov

    def __call__(self, simulation):
        sims = self._my_method_generator()
        if simulation in sims.keys():
            return sims[simulation]()
        else:
            raise ValueError(f"simulation is not one of these: {sims.keys()}")

    def _my_method_generator(self):
        return {
            method: getattr(self, method)
            for method in dir(self)
            if not method.startswith("__")
        }

    def skewed_unimodal(self):
        return [
            [np.zeros(self.n_dim), self.cov],
            [np.full(self.n_dim, 1 / 2), self.cov * (2 / 3) ** 2],
            [np.full(self.n_dim, 13 / 12), self.cov * (5 / 9) ** 2],
        ]

    def strongly_skewed(self):
        return [
            [
                np.full(self.n_dim, 3 * ((2 / 3) ** l_mix - 1)),
                self.cov * (2 / 3) ** (2 * l_mix),
            ]
            for l_mix in range(8)
        ]

    def kurtotic_unimodal(self):
        return [
            [np.zeros(self.n_dim), self.cov],
            [np.zeros(self.n_dim), self.cov * (1 / 10) ** 2],
        ]

    def outlier(self):
        return [
            [np.zeros(self.n_dim), self.cov],
            [np.zeros(self.n_dim), self.cov * (1 / 10) ** 2],
        ]

    def bimodal(self):
        return [
            [-np.ones(self.n_dim), self.cov * (2 / 3) ** 2],
            [np.ones(self.n_dim), self.cov * (2 / 3) ** 2],
        ]

    def separated_bimodal(self):
        return [
            [-np.full(self.n_dim, 3 / 2), self.cov * (1 / 2) ** 2],
            [np.full(self.n_dim, 3 / 2), self.cov * (1 / 2) ** 2],
        ]

    def skewed_bimodal(self):
        return [
            [np.zeros(self.n_dim), self.cov],
            [np.full(self.n_dim, 3 / 2), self.cov * (1 / 3) ** 2],
        ]

    def trimodal(self):
        return [
            [np.full(self.n_dim, -6 / 5), self.cov * (3 / 5) ** 2],
            [np.full(self.n_dim, 6 / 5), self.cov * (3 / 5) ** 2],
            [np.zeros(self.n_dim), self.cov * (1 / 4) ** 2],
        ]

    def claw(self):
        return [
            [np.zeros(self.n_dim), self.cov],
            *[
                [np.full(self.n_dim, (l_mix / 2) - 1), self.cov * (1 / 10) ** 2]
                for l_mix in range(5)
            ],
        ]

    def double_claw(self):
        return [
            [-np.ones(self.n_dim), self.cov * (2 / 3) ** 2],
            [np.ones(self.n_dim), self.cov * (2 / 3) ** 2],
            *[
                [np.full(self.n_dim, (l_mix - 3) / 2), self.cov * (1 / 100) ** 2]
                for l_mix in range(7)
            ],
        ]

    def asymmetric_claw(self):
        return [
            [np.zeros(self.n_dim), self.cov],
            *[
                [
                    np.full(self.n_dim, l_mix + 1 / 2),
                    self.cov * (1 / ((2**l_mix) * 10)) ** 2,
                ]
                for l_mix in range(-2, 3)
            ],
        ]

    def asymmetric_double_claw(self):
        return [
            *[
                [np.full(self.n_dim, 2 * l_mix - 1), self.cov * (2 / 3) ** 2]
                for l_mix in range(2)
            ],
            *[
                [-np.full(self.n_dim, l_mix / 2), self.cov * (1 / 100) ** 2]
                for l_mix in range(1, 4)
            ],
            *[
                [np.full(self.n_dim, l_mix / 2), self.cov * (7 / 100) ** 2]
                for l_mix in range(1, 4)
            ],
        ]

    def smooth_comb(self):
        return [
            [
                np.full(self.n_dim, (65 - 96 * ((1 / 2) ** l_mix)) / 21),
                self.cov * (32 / 63) ** 2 / (2 ** (2 * l_mix)),
            ]
            for l_mix in range(6)
        ]

    def discrete_comb(self):
        return [
            *[
                [np.full(self.n_dim, (12 * l_mix - 15) / 7), self.cov * (2 / 7) ** 2]
                for l_mix in range(3)
            ],
            *[
                [np.full(self.n_dim, (2 * l_mix) / 7), self.cov * (1 / 21) ** 2]
                for l_mix in range(8, 11)
            ],
        ]

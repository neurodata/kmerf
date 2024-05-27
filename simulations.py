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

INDEPENDENCE_SIMS = [
    "linear",
    "exponential",
    "cubic",
    "joint_normal",
    "step",
    "quadratic",
    "w_shaped",
    "spiral",
    "uncorrelated_bernoulli",
    "logarithmic",
    "fourth_root",
    "sin_four_pi",
    "sin_sixteen_pi",
    "square",
    "two_parabolas",
    "circle",
    "ellipse",
    "diamond",
    "multiplicative_noise",
    "multimodal_independence",
]


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


def make_independence_simulation(
    n_samples,
    n_dim=1,
    simulation: str = "linear",
    noise: bool = False,
    seed=None,
    **kwargs,
):
    return IndependenceSims(samp_size=n_samples, n_dim=n_dim, noise=noise, seed=seed)(
        simulation=simulation, **kwargs
    )


class IndependenceSims:

    def __init__(self, samp_size=100, n_dim=1, noise=True, seed=None):
        self.samp_size = samp_size
        self.n_dim = n_dim
        self.noise = noise
        self.rng = np.random.default_rng(seed=seed)

    def __call__(self, simulation, **kwargs):
        sims = self._my_method_generator()
        if simulation in sims.keys():
            return sims[simulation](**kwargs)
        else:
            raise ValueError(f"simulation is not one of these: {sims.keys()}")

    def _my_method_generator(self):
        return {
            method: getattr(self, method)
            for method in dir(self)
            if not method.startswith("__")
        }

    def _gen_coeffs(self):
        """Calculates coefficients polynomials."""
        return np.array([1 / (i + 1) for i in range(self.n_dim)]).reshape(-1, 1)

    def _random_uniform(self, low=-1, high=1):
        """Generate random uniform data."""
        return np.array(self.rng.uniform(low, high, size=(self.samp_size, self.n_dim)))

    def _calc_eps(self):
        """Calculate noise."""
        return np.random.normal(0, 1, size=(self.samp_size, 1))

    def linear(self, low=-1, high=1):
        x = self._random_uniform(low, high)
        coeffs = self._gen_coeffs()
        eps = self._calc_eps()
        y = x @ coeffs + 1 * self.noise * eps
        return x, y

    def exponential(self, low=0, high=3):
        x = self._random_uniform(low, high)
        coeffs = self._gen_coeffs()
        eps = self._calc_eps()
        y = np.exp(x @ coeffs) + 10 * self.noise * eps
        return x, y

    def cubic(self, low=-1, high=1, cubs=[-12, 48, 128], scale=1 / 3):
        x = self._random_uniform(low, high)
        coeffs = self._gen_coeffs()
        eps = self._calc_eps()
        x_coeffs = x @ coeffs - scale
        y = (
            cubs[2] * x_coeffs**3
            + cubs[1] * x_coeffs**2
            + cubs[0] * x_coeffs**3
            + 80 * self.noise * eps
        )
        return x, y

    def joint_normal(self):
        coeffs = self._gen_coeffs()
        rho = 1 / (2 * coeffs)
        cov1 = np.concatenate(
            (np.identity(self.n_dim), rho * np.ones((self.n_dim, self.n_dim))), axis=1
        )
        cov2 = np.concatenate(
            (rho * np.ones((self.n_dim, self.n_dim)), np.identity(self.n_dim)), axis=1
        )
        covT = np.concatenate((cov1.T, cov2.T), axis=1)
        eps = self._calc_eps()
        x = self.rng.multivariate_normal(np.zeros(2 * self.n_dim), covT, self.samp_size)
        y = x[:, self.n_dim].reshape(-1, 1) + 0.5 * self.noise * eps
        x = x[:, : self.n_dim]
        return x, y

    def step(self, low=-1, high=1):
        if self.n_dim > 1:
            self.noise = True
        x = self._random_uniform(low, high)
        coeffs = self._gen_coeffs()
        eps = self._calc_eps()
        x_coeff = ((x @ coeffs) > 0) * 1
        y = x_coeff + self.noise * eps
        return x, y

    def quadratic(self, low=-1, high=1):
        x = self._random_uniform(low, high)
        coeffs = self._gen_coeffs()
        eps = self._calc_eps()
        x_coeffs = x @ coeffs
        y = x_coeffs**2 + 0.5 * self.noise * eps
        return x, y

    def w_shaped(self, low=-1, high=1):
        x = self._random_uniform(low, high)
        u = np.array(
            self.rng.spawn(1)[0].uniform(0, 1, size=(self.samp_size, self.n_dim))
        )
        coeffs = self._gen_coeffs()
        eps = self._calc_eps()
        x_coeffs = x @ coeffs
        u_coeffs = u @ coeffs
        y = 4 * ((x_coeffs**2 - 0.5) ** 2 + u_coeffs / 500) + 0.5 * self.noise * eps
        return x, y

    def spiral(self, low=0, high=5):
        coeffs = self._gen_coeffs()
        if self.n_dim > 1:
            self.noise = True
        n_dim = self.n_dim
        self.n_dim = 1
        rx = self._random_uniform(low=low, high=high)
        ry = rx
        rx = np.repeat(rx, n_dim, axis=1)
        z = rx
        x = np.zeros((self.samp_size, n_dim))
        x[:, 0] = np.cos(z[:, 0] * np.pi)
        for i in range(n_dim - 1):
            x[:, i + 1] = np.multiply(x[:, i], np.cos(z[:, i + 1] * np.pi))
            x[:, i] = np.multiply(x[:, i], np.sin(z[:, i + 1] * np.pi))
        x = np.multiply(rx, x) * coeffs.ravel()
        y = np.multiply(ry, np.sin(z[:, 0].reshape(-1, 1) * np.pi))
        eps = self._calc_eps()
        y = y + 0.4 * n_dim * self.noise * eps
        return x, y

    def uncorrelated_bernoulli(self, prob=0.5):
        rngs = self.rng.spawn(2)
        binom = self.rng.binomial(1, prob, size=(self.samp_size, 1))
        sig = np.identity(self.n_dim)
        gauss_noise = rngs[0].multivariate_normal(
            np.zeros(self.n_dim), sig, size=self.samp_size
        )
        x = (
            rngs[1].binomial(1, prob, size=(self.samp_size, self.n_dim))
            + 0.5 * self.noise * gauss_noise
        )
        coeffs = self._gen_coeffs()
        eps = self._calc_eps()
        x_coeffs = x @ coeffs
        y = binom * 2 - 1
        y = np.multiply(x_coeffs, y) + 0.5 * self.noise * eps
        return x, y

    def logarithmic(self):
        coeffs = self._gen_coeffs()
        sig = np.identity(self.n_dim)
        x = self.rng.multivariate_normal(np.zeros(self.n_dim), sig, size=self.samp_size)
        eps = self._calc_eps()
        x_coeffs = x @ coeffs
        y = np.log(x_coeffs**2) + 3 * self.noise * eps
        return x, y

    def fourth_root(self, low=-1, high=1):
        x = self._random_uniform(low, high)
        eps = self._calc_eps()
        coeffs = self._gen_coeffs()
        x_coeffs = x @ coeffs
        y = np.abs(x_coeffs) ** 0.25 + 0.25 * self.noise * eps
        return x, y

    def _sin(self, low=-1, high=1, period=4 * np.pi):
        """Helper function to calculate sine simulation"""
        coeffs = self._gen_coeffs()
        x = self._random_uniform(low, high)
        if self.n_dim > 1 or self.noise:
            sig = np.identity(self.n_dim)
            v = self.rng.spawn(1)[0].multivariate_normal(
                np.zeros(self.n_dim), sig, size=self.samp_size
            )
            x = x + 0.02 * self.n_dim * v
        eps = self._calc_eps()
        if period == 4 * np.pi:
            cc = 1
        else:
            cc = 0.5
        y = np.sin(x @ coeffs * period) + cc * self.noise * eps
        return x, y

    def sin_four_pi(self, low=-1, high=1):
        return self._sin(low=low, high=high, period=4 * np.pi)

    def sin_sixteen_pi(self, low=-1, high=1):
        return self._sin(low=low, high=high, period=16 * np.pi)

    def _square_diamond(self, low=-1, high=1, period=-np.pi / 2):
        """Helper function to calculate square/diamond simulation"""
        rngs = self.rng.spawn(2)
        coeffs = self._gen_coeffs()
        u = self._random_uniform(low, high)
        v = np.array(rngs[0].uniform(low, high, size=(self.samp_size, self.n_dim)))
        sig = np.identity(self.n_dim)
        gauss_noise = rngs[1].multivariate_normal(
            np.zeros(self.n_dim), sig, size=self.samp_size
        )
        x = (
            u * np.cos(period)
            + v * np.sin(period)
            + 0.05 * self.n_dim * gauss_noise * self.noise
        )
        y = -u @ coeffs * np.sin(period) + v @ coeffs * np.cos(period)
        return x, y

    def square(self, low=-1, high=1):
        return self._square_diamond(low=low, high=high, period=-np.pi / 8)

    def two_parabolas(self, low=-1, high=1, prob=0.5):
        rngs = self.rng.spawn(2)
        x = self._random_uniform(low, high)
        coeffs = self._gen_coeffs()
        u = rngs[0].binomial(1, prob, size=(self.samp_size, 1))
        rand_noise = np.array(rngs[1].uniform(0, 1, size=(self.samp_size, self.n_dim)))
        x_coeffs = x @ coeffs
        y = (x_coeffs**2 + 2 * self.noise * rand_noise) * (u - 0.5)
        return x, y

    def _circle_ellipse(self, low=-1, high=1, radius=1):
        """Helper function to calculate circle/ellipse simulation"""
        rngs = self.rng.spawn(2)
        coeffs = self._gen_coeffs()
        if self.n_dim > 1:
            self.noise = True
        x = self._random_uniform(low, high)
        rx = radius * np.ones((self.samp_size, self.n_dim))
        unif = np.array(rngs[0].uniform(low, high, size=(self.samp_size, self.n_dim)))
        sig = np.identity(self.n_dim)
        gauss_noise = rngs[1].multivariate_normal(
            np.zeros(self.n_dim), sig, size=self.samp_size
        )
        ry = np.ones((self.samp_size, 1))
        x[:, 0] = np.cos(unif[:, 0] * np.pi)
        for i in range(self.n_dim - 1):
            x[:, i + 1] = x[:, i] * np.cos(unif[:, i + 1] * np.pi)
            x[:, i] = x[:, i] * np.sin(unif[:, i + 1] * np.pi)
        x = rx * x + 0.4 * self.noise * rx * gauss_noise
        y = ry * np.sin(unif @ coeffs * np.pi)
        return x, y

    def circle(self, low=-1, high=1):
        return self._circle_ellipse(low=low, high=high, radius=1)

    def ellipse(self, low=-1, high=1):
        return self._circle_ellipse(low=low, high=high, radius=5)

    def diamond(self, low=-1, high=1):
        return self._square_diamond(low=low, high=high, period=-np.pi / 4)

    def multiplicative_noise(self):
        rngs = self.rng.spawn(2)
        coeffs = self._gen_coeffs()
        sig = np.identity(self.n_dim)
        x = rngs[0].multivariate_normal(np.zeros(self.n_dim), sig, size=self.samp_size)
        y = rngs[1].multivariate_normal(np.zeros(self.n_dim), sig, size=self.samp_size)
        y = np.multiply(x, y) @ coeffs
        return x, y

    def multimodal_independence(self, prob=0.5, sep1=3, sep2=2):
        rngs = self.rng.spawn(4)
        sig = np.identity(self.n_dim)
        u = rngs[0].multivariate_normal(np.zeros(self.n_dim), sig, size=self.samp_size)
        v = rngs[1].multivariate_normal(np.zeros(self.n_dim), sig, size=self.samp_size)
        u_2 = rngs[2].binomial(1, prob, size=(self.samp_size, self.n_dim))
        v_2 = rngs[3].binomial(1, prob, size=(self.samp_size, self.n_dim))
        x = u / sep1 + sep2 * u_2 - 1
        y = v[:, 0] / sep1 + sep2 * v_2[:, 0] - 1
        return x, y

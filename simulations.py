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


class _CheckInputs:
    """Check if additional arguments are correct"""

    def __init__(self, n, p):
        self.n = n
        self.p = p

    def __call__(self, *args):
        if type(self.n) is not int or type(self.p) is not int:
            raise ValueError("n and p must be ints")

        if self.n < 5 or self.p < 1:
            raise ValueError(
                "n must be greater than or equal to 5 and p "
                "must be greater than or equal to than 1"
            )

        for arg in args:
            if arg[1] is float and type(arg[0]) is int:
                continue
            if type(arg[0]) is not arg[1]:
                raise ValueError("Incorrect input variable type")


def _gen_coeffs(p):
    """Calculates coefficients polynomials"""
    return np.array([1 / (i + 1) for i in range(p)]).reshape(-1, 1)


def _random_uniform(n, p, low=-1, high=1):
    """Generate random uniform data"""
    return np.random.uniform(low, high, size=(n, p))


def _calc_eps(n):
    """Calculate noise"""
    return np.random.normal(0, 1, size=(n, 1))


def linear(n, p, noise=False, low=-1, high=1):
    r"""
    Simulates univariate or multivariate linear data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.
    low : float, (default: -1)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: -1)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Linear :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= w^T X + \kappa \epsilon

    Examples
    --------
    >>> from hyppo.sims import linear
    >>> x, y = linear(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)
    """
    extra_args = [(noise, bool), (low, float), (high, float)]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)
    y = x @ coeffs + 1 * noise * eps

    return x, y


def exponential(n, p, noise=False, low=0, high=3):
    r"""
    Simulates univariate or multivariate exponential data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.
    low : float, (default: 0)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: 3)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Exponential :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(0, 3)^p \\
        Y &= \exp (w^T X) + 10 \kappa \epsilon

    Examples
    --------
    >>> from hyppo.sims import exponential
    >>> x, y = exponential(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)
    """
    extra_args = [(noise, bool), (low, float), (high, float)]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)
    y = np.exp(x @ coeffs) + 10 * noise * eps

    return x, y


def cubic(n, p, noise=False, low=-1, high=1, cubs=[-12, 48, 128], scale=1 / 3):
    r"""
    Simulates univariate or multivariate cubic data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.
    low : float, (default: -1)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: -1)
        The upper limit of the uniform distribution simulated from.
    cubs : list of ints (default: [-12, 48, 128])
        Coefficients of the cubic function where each value corresponds to the
        order of the cubic polynomial.
    scale : float (default: 1/3)
        Scaling center of the cubic.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Cubic :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= 128 \left( w^T X - \frac{1}{3} \right)^3
             + 48 \left( w^T X - \frac{1}{3} \right)^2
             - 12 \left( w^T X - \frac{1}{3} \right)
             + 80 \kappa \epsilon

    Examples
    --------
    >>> from hyppo.sims import cubic
    >>> x, y = cubic(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)
    """
    extra_args = [
        (noise, bool),
        (low, float),
        (high, float),
        (cubs, list),
        (scale, float),
    ]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)

    x_coeffs = x @ coeffs - scale
    y = (
        cubs[2] * x_coeffs**3
        + cubs[1] * x_coeffs**2
        + cubs[0] * x_coeffs**3
        + 80 * noise * eps
    )

    return x, y


def joint_normal(n, p, noise=False):
    r"""
    Simulates univariate or multivariate joint-normal data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, p)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Joint Normal :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`: Let
    :math:`\rho = \frac{1}{2} p`, :math:`I_p` be the identity matrix of size
    :math:`p \times p`, :math:`J_p` be the matrix of ones of size
    :math:`p \times p` and
    :math:`\Sigma = \begin{bmatrix} I_p & \rho J_p \\ \rho J_p & (1 + 0.5\kappa) I_p \end{bmatrix}`. Then,

    .. math::

        (X, Y) \sim \mathcal{N}(0, \Sigma)

    Examples
    --------
    >>> from hyppo.sims import joint_normal
    >>> x, y = joint_normal(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)
    """
    if p > 10:
        raise ValueError("Covariance matrix for p>10 is not positive" "semi-definite")

    extra_args = [(noise, bool)]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    coeffs = _gen_coeffs(p)
    rho = 1 / (2 * coeffs)
    cov1 = np.concatenate((np.identity(p), rho * np.ones((p, p))), axis=1)
    cov2 = np.concatenate((rho * np.ones((p, p)), np.identity(p)), axis=1)
    covT = np.concatenate((cov1.T, cov2.T), axis=1)

    eps = _calc_eps(n)
    x = np.random.multivariate_normal(np.zeros(2 * p), covT, n)
    y = x[:, 0].reshape(-1, 1) + 0.5 * noise * eps
    x = x[:, :p]

    return x, y


def step(n, p, noise=False, low=-1, high=1):
    r"""
    Simulates univariate or multivariate step data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.
    low : float, (default: -1)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: -1)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Step :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= \mathbb{1}_{w^T X > 0} + \epsilon

    where :math:`\mathbb{1}` is the indicator function.

    Examples
    --------
    >>> from hyppo.sims import step
    >>> x, y = step(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)
    """
    extra_args = [(noise, bool), (low, float), (high, float)]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    if p > 1:
        noise = True
    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)

    x_coeff = ((x @ coeffs) > 0) * 1
    y = x_coeff + noise * eps

    return x, y


def quadratic(n, p, noise=False, low=-1, high=1):
    r"""
    Simulates univariate or multivariate quadratic data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.
    low : float, (default: -1)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: -1)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Quadratic :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= (w^T X)^2 + 0.5 \kappa \epsilon

    Examples
    --------
    >>> from hyppo.sims import quadratic
    >>> x, y = quadratic(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)
    """
    extra_args = [(noise, bool), (low, float), (high, float)]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)

    x_coeffs = x @ coeffs
    y = x_coeffs**2 + 0.5 * noise * eps

    return x, y


def w_shaped(n, p, noise=False, low=-1, high=1):
    r"""
    Simulates univariate or multivariate quadratic data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.
    low : float, (default: -1)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: -1)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    W-Shaped :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:
    :math:`\mathcal{U}(-1, 1)^p`,

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= \left[ \left( (w^T X)^2 - \frac{1}{2} \right)^2
                            + \frac{w^T U}{500} \right] + 0.5 \kappa \epsilon

    Examples
    --------
    >>> from hyppo.sims import w_shaped
    >>> x, y = w_shaped(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)
    """
    extra_args = [(noise, bool), (low, float), (high, float)]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    x = _random_uniform(n, p, low, high)
    u = _random_uniform(n, p, 0, 1)
    coeffs = _gen_coeffs(p)
    eps = _calc_eps(n)

    x_coeffs = x @ coeffs
    u_coeffs = u @ coeffs
    y = 4 * ((x_coeffs**2 - 0.5) ** 2 + u_coeffs / 500) + 0.5 * noise * eps

    return x, y


def spiral(n, p, noise=False, low=0, high=5):
    r"""
    Simulates univariate or multivariate spiral data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.
    low : float, (default: 0)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: 5)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Spiral :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:
    :math:`U \sim \mathcal{U}(0, 5)`, :math:`\epsilon \sim \mathcal{N}(0, 1)`

    .. math::

        X_{|d|} &= U \sin(\pi U) \cos^d(\pi U)\ \mathrm{for}\ d = 1,...,p-1 \\
        X_{|p|} &= U \cos^p(\pi U) \\
        Y &= U \sin(\pi U) + 0.4 p \epsilon

    Examples
    --------
    >>> from hyppo.sims import spiral
    >>> x, y = spiral(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)
    """
    extra_args = [(noise, bool), (low, float), (high, float)]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    coeffs = _gen_coeffs(p)
    if p > 1:
        noise = True
    rx = _random_uniform(n, p=1, low=low, high=high)
    ry = rx
    rx = np.repeat(rx, p, axis=1)
    z = rx
    x = np.zeros((n, p))
    x[:, 0] = np.cos(z[:, 0] * np.pi)
    for i in range(p - 1):
        x[:, i + 1] = x[:, i] * np.cos(z[:, i + 1] * np.pi)
        x[:, i] = x[:, i] * np.sin(z[:, i + 1] * np.pi)
    x = rx * x
    y = ry * np.sin(z @ coeffs * np.pi)

    eps = _calc_eps(n)
    y = y + 0.4 * p * noise * eps

    return x, y


def uncorrelated_bernoulli(n, p, noise=False, prob=0.5):
    r"""
    Simulates univariate or multivariate uncorrelated Bernoulli data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.
    prob : float, (default: 0.5)
        The probability of the bernoulli distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Uncorrelated Bernoulli :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:
    :math:`U \sim \mathcal{B}(0.5)`, :math:`\epsilon_1 \sim \mathcal{N}(0, I_p)`,
    :math:`\epsilon_2 \sim \mathcal{N}(0, 1)`,

    .. math::

        X &= \mathcal{B}(0.5)^p + 0.5 \epsilon_1 \\
        Y &= (2U - 1) w^T X + 0.5 \epsilon_2

    Examples
    --------
    >>> from hyppo.sims import uncorrelated_bernoulli
    >>> x, y = uncorrelated_bernoulli(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)
    """
    extra_args = [(noise, bool), (prob, float)]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    binom = np.random.binomial(1, prob, size=(n, 1))
    sig = np.identity(p)
    gauss_noise = np.random.multivariate_normal(np.zeros(p), sig, size=n)

    x = np.random.binomial(1, prob, size=(n, p)) + 0.5 * noise * gauss_noise
    coeffs = _gen_coeffs(p)

    eps = _calc_eps(n)
    x_coeffs = x @ coeffs
    y = binom * 2 - 1
    y = np.multiply(x_coeffs, y) + 0.5 * noise * eps

    return x, y


def logarithmic(n, p, noise=False):
    r"""
    Simulates univariate or multivariate logarithmic data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, p)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Logarithmic :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`\epsilon \sim \mathcal{N}(0, I_p)`,

    .. math::

        X &\sim \mathcal{N}(0, I_p) \\
        Y_{|d|} &= 2 \log_2 (|X_{|d|}|) + 3 \kappa \epsilon_{|d|}
                   \ \mathrm{for}\ d = 1, ..., p

    Examples
    --------
    >>> from hyppo.sims import logarithmic
    >>> x, y = logarithmic(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 2)
    """
    extra_args = [(noise, bool)]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    coeffs = _gen_coeffs(p)
    sig = np.identity(p)
    x = np.random.multivariate_normal(np.zeros(p), sig, size=n)
    eps = _calc_eps(n)

    y = np.log(x**2) + 3 * noise * eps
    y = y @ coeffs  # y[:, 0].reshape(-1, 1)

    return x, y


def fourth_root(n, p, noise=False, low=-1, high=1):
    r"""
    Simulates univariate or multivariate fourth root data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.
    low : float, (default: -1)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: -1)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Fourth Root :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}`:

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= |w^T X|^\frac{1}{4} + \frac{\kappa}{4} \epsilon

    Examples
    --------
    >>> from hyppo.sims import fourth_root
    >>> x, y = fourth_root(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 1)
    """
    extra_args = [(noise, bool), (low, float), (high, float)]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    x = _random_uniform(n, p, low, high)
    eps = _calc_eps(n)
    coeffs = _gen_coeffs(p)

    x_coeffs = x @ coeffs
    y = np.abs(x_coeffs) ** 0.25 + 0.25 * noise * eps

    return x, y


def _sin(n, p, noise=False, low=-1, high=1, period=4 * np.pi):
    """Helper function to calculate sine simulation"""
    extra_args = [(noise, bool), (low, float), (high, float), (period, float)]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    coeffs = _gen_coeffs(p).reshape(-1)
    x = _random_uniform(n, p, low, high) * np.flip(coeffs)
    if p > 1 or noise:
        sig = np.identity(p)
        v = np.random.multivariate_normal(np.zeros(p), sig, size=n)
        x = x + 0.02 * p * v
    eps = _calc_eps(n)

    if period == 4 * np.pi:
        cc = 1
    else:
        cc = 0.5

    y = np.sin(x * period) + cc * noise * eps  # @ coeffs * period) + cc * noise * eps

    return x, y


def sin_four_pi(n, p, noise=False, low=-1, high=1):
    r"""
    Simulates univariate or multivariate sine 4 :math:`\pi` data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.
    low : float, (default: -1)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: -1)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, p)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Sine 4:math:`\pi` :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`U \sim \mathcal{U}(-1, 1)`, :math:`V \sim \mathcal{N}(0, 1)^p`,
    :math:`\theta = 4 \pi`,

    .. math::

        X_{|d|} &= U + 0.02 p V_{|d|}\ \mathrm{for}\ d = 1, ..., p \\
        Y &= \sin (\theta X) + \kappa \epsilon

    Examples
    --------
    >>> from hyppo.sims import sin_four_pi
    >>> x, y = sin_four_pi(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 2)
    """
    return _sin(n, p, noise=noise, low=low, high=high, period=4 * np.pi)


def sin_sixteen_pi(n, p, noise=False, low=-1, high=1):
    r"""
    Simulates univariate or multivariate sine 16 :math:`\pi` data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.
    low : float, (default: -1)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: -1)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, p)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Sine 16:math:`\pi` :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`U \sim \mathcal{U}(-1, 1)`, :math:`V \sim \mathcal{N}(0, 1)^p`,
    :math:`\theta = 16 \pi`,

    .. math::

        X_{|d|} &= U + 0.02 p V_{|d|}\ \mathrm{for}\ d = 1, ..., p \\
        Y &= \sin (\theta X) + \kappa \epsilon

    Examples
    --------
    >>> from hyppo.sims import sin_sixteen_pi
    >>> x, y = sin_sixteen_pi(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 2)
    """
    return _sin(n, p, noise=noise, low=low, high=high, period=16 * np.pi)


def _square_diamond(n, p, noise=False, low=-1, high=1, period=-np.pi / 2):
    """Helper function to calculate square/diamond simulation"""
    extra_args = [(noise, bool), (low, float), (high, float), (period, float)]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    coeffs = _gen_coeffs(p).reshape(-1)
    u = _random_uniform(n, p, low, high)
    v = _random_uniform(n, p, low, high)
    sig = np.identity(p)
    gauss_noise = np.random.multivariate_normal(np.zeros(p), sig, size=n)

    x = u * np.cos(period) + v * np.sin(period) + 0.05 * np.flip(coeffs) * gauss_noise
    y = -u * coeffs * np.sin(period) + v * coeffs * np.cos(period)

    return x, y


def square(n, p, noise=False, low=-1, high=1):
    r"""
    Simulates univariate or multivariate square data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.
    low : float, (default: -1)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: -1)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, p)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Square :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`U \sim \mathcal{U}(-1, 1)`, :math:`V \sim \mathcal{N}(0, 1)^p`,
    :math:`\theta = -\frac{\pi}{8}`,

    .. math::

        X_{|d|} &= U \cos(\theta) + V \sin(\theta) + 0.05 p \epsilon_{|d|}\ \mathrm{for}\ d = 1, ..., p \\
        Y_{|d|} &= -U \sin(\theta) + V \cos(\theta)

    Examples
    --------
    >>> from hyppo.sims import square
    >>> x, y = square(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 2)
    """
    return _square_diamond(n, p, noise=noise, low=low, high=high, period=-np.pi / 8)


def two_parabolas(n, p, noise=False, low=-1, high=1, prob=0.5):
    r"""
    Simulates univariate or multivariate two parabolas data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.
    low : float, (default: -1)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: -1)
        The upper limit of the uniform distribution simulated from.
    prob : float, (default: 0.5)
        The probability of the bernoulli distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, 1)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Two Parabolas :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:

    .. math::

        X &\sim \mathcal{U}(-1, 1)^p \\
        Y &= ((w^T X)^2 + 2 \kappa \epsilon) \times \left( U = \frac{1}{2} \right)

    Examples
    --------
    >>> from hyppo.sims import two_parabolas
    >>> x, y = two_parabolas(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 2)
    """
    extra_args = [(noise, bool), (low, float), (high, float), (prob, float)]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    x = _random_uniform(n, p, low, high)
    coeffs = _gen_coeffs(p)
    u = np.random.binomial(1, prob, size=(n, 1))
    rand_noise = _random_uniform(n, 1, low=0, high=1)

    x_coeffs = x @ coeffs
    y = (x_coeffs**2 + 2 * noise * rand_noise) * (u - 0.5)

    return x, y


def _circle_ellipse(n, p, noise=False, low=-1, high=1, radius=1):
    """Helper function to calculate circle/ellipse simulation"""
    extra_args = [(noise, bool), (low, float), (high, float), (radius, float)]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    coeffs = _gen_coeffs(p)
    if p > 1:
        noise = True
    x = _random_uniform(n, p, low, high)
    rx = radius * np.ones((n, p))
    unif = _random_uniform(n, p, low, high)
    sig = np.identity(p)
    gauss_noise = np.random.multivariate_normal(np.zeros(p), sig, size=n)

    ry = np.ones((n, 1))
    x[:, 0] = np.cos(unif[:, 0] * np.pi)
    for i in range(p - 1):
        x[:, i + 1] = x[:, i] * np.cos(unif[:, i + 1] * np.pi)
        x[:, i] = x[:, i] * np.sin(unif[:, i + 1] * np.pi)

    x = rx * x + 0.4 * noise * rx * gauss_noise
    y = ry * np.sin(unif @ coeffs * np.pi)

    return x, y


def circle(n, p, noise=False, low=-1, high=1):
    r"""
    Simulates univariate or multivariate circle data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.
    low : float, (default: -1)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: -1)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, p)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Circle :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`U \sim \mathcal{U}(-1, 1)^p`, :math:`\epsilon \sim \mathcal{N}(0, I_p)`,
    :math:`r = 1`,

    .. math::

        X_{|d|} &= r \left( \sin(\pi U_{|d+1|}) \prod_{j=1}^d \cos(\pi U_{|j|}) + 0.4 \epsilon_{|d|} \right)\ \mathrm{for}\ d = 1, ..., p-1 \\
        X_{|d|} &= r \left( \prod_{j=1}^p \cos(\pi U_{|j|}) + 0.4 \epsilon_{|p|} \right) \\
        Y_{|d|} &= \sin(\pi U_{|1|})

    Examples
    --------
    >>> from hyppo.sims import circle
    >>> x, y = circle(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 2)
    """
    return _circle_ellipse(n, p, noise=noise, low=low, high=high, radius=1)


def ellipse(n, p, noise=False, low=-1, high=1):
    r"""
    Simulates univariate or multivariate ellipse data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.
    low : float, (default: -1)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: -1)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, p)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Ellipse :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`U \sim \mathcal{U}(-1, 1)^p`, :math:`\epsilon \sim \mathcal{N}(0, I_p)`,
    :math:`r = 5`,

    .. math::

        X_{|d|} &= r \left( \sin(\pi U_{|d+1|}) \prod_{j=1}^d \cos(\pi U_{|j|}) + 0.4 \epsilon_{|d|} \right)\ \mathrm{for}\ d = 1, ..., p-1 \\
        X_{|d|} &= r \left( \prod_{j=1}^p \cos(\pi U_{|j|}) + 0.4 \epsilon_{|p|} \right) \\
        Y_{|d|} &= \sin(\pi U_{|1|})

    Examples
    --------
    >>> from hyppo.sims import ellipse
    >>> x, y = ellipse(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 2)
    """
    return _circle_ellipse(n, p, noise=noise, low=low, high=high, radius=5)


def diamond(n, p, noise=False, low=-1, high=1):
    r"""
    Simulates univariate or multivariate diamond data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    noise : bool, (default: False)
        Whether or not to include noise in the simulation.
    low : float, (default: -1)
        The lower limit of the uniform distribution simulated from.
    high : float, (default: -1)
        The upper limit of the uniform distribution simulated from.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, p)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Diamond :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`U \sim \mathcal{U}(-1, 1)`, :math:`V \sim \mathcal{N}(0, 1)^p`,
    :math:`\theta = -\frac{\pi}{4}`,

    .. math::

        X_{|d|} &= U \cos(\theta) + V \sin(\theta) + 0.05 p \epsilon_{|d|}\ \mathrm{for}\ d = 1, ..., p \\
        Y_{|d|} &= -U \sin(\theta) + V \cos(\theta)

    Examples
    --------
    >>> from hyppo.sims import diamond
    >>> x, y = diamond(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 2)
    """
    return _square_diamond(n, p, noise=noise, low=low, high=high, period=-np.pi / 4)


def multiplicative_noise(n, p):
    r"""
    Simulates univariate or multivariate multiplicative noise data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, p)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Multiplicative Noise :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`\U \sim \mathcal{N}(0, I_p)`,

    .. math::

        X &\sim \mathcal{N}(0, I_p) \\
        Y_{|d|} &= U_{|d|} X_{|d|}\ \mathrm{for}\ d = 1, ..., p

    Examples
    --------
    >>> from hyppo.sims import multiplicative_noise
    >>> x, y = multiplicative_noise(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 2)
    """
    extra_args = []
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    coeffs = _gen_coeffs(p)
    sig = np.identity(p)
    x = np.random.multivariate_normal(np.zeros(p), sig, size=n)
    y = np.random.multivariate_normal(np.zeros(p), sig, size=n)
    y = np.multiply(x, y)
    y = y @ coeffs  # y[:, 0].reshape(-1, 1)

    return x, y


def multimodal_independence(n, p, prob=0.5, sep1=3, sep2=2):
    r"""
    Simulates univariate or multimodal independence data.

    Parameters
    ----------
    n : int
        The number of samples desired by the simulation.
    p : int
        The number of dimensions desired by the simulation.
    prob : float, (default: 0.5)
        The probability of the bernoulli distribution simulated from.
    sep1, sep2: float, (default: 3, 2)
        The separation between clusters of normally distributed data.

    Returns
    -------
    x, y : ndarray
        Simulated data matrices. `x` and `y` have shapes `(n, p)` and `(n, p)`
        where `n` is the number of samples and `p` is the number of
        dimensions.

    Notes
    -----
    Multimodal Independence :math:`(X, Y) \in \mathbb{R}^p \times \mathbb{R}^p`:
    :math:`U \sim \mathcal{N}(0, I_p)`, :math:`V \sim \mathcal{N}(0, I_p)`,
    :math:`U^\prime \sim \mathcal{B}(0.5)^p`, :math:`V^\prime \sim \mathcal{B}(0.5)^p`,

    .. math::

        X &= \frac{U}{3} + 2 U^\prime - 1 \\
        Y &= \frac{V}{3} + 2 V^\prime - 1

    Examples
    --------
    >>> from hyppo.sims import multimodal_independence
    >>> x, y = multimodal_independence(100, 2)
    >>> print(x.shape, y.shape)
    (100, 2) (100, 2)
    """
    extra_args = [(prob, float), (sep1, float), (sep2, float)]
    check_in = _CheckInputs(n, p)
    check_in(*extra_args)

    sig = np.identity(p)
    u = np.random.multivariate_normal(np.zeros(p), sig, size=n)
    v = np.random.multivariate_normal(np.zeros(p), sig, size=n)
    u_2 = np.random.binomial(1, prob, size=(n, p))
    v_2 = np.random.binomial(1, prob, size=(n, p))

    x = u / sep1 + sep2 * u_2 - 1
    y = v / sep1 + sep2 * v_2 - 1
    y = y[:, 0].reshape(-1, 1)

    return x, y


INDEPENDENCE_SIMS = {
    "linear": linear,
    "exponential": exponential,
    "cubic": cubic,
    "joint_normal": joint_normal,
    "step": step,
    "quadratic": quadratic,
    "w_shaped": w_shaped,
    "spiral": spiral,
    "uncorrelated_bernoulli": uncorrelated_bernoulli,
    "logarithmic": logarithmic,
    "fourth_root": fourth_root,
    "sin_four_pi": sin_four_pi,
    "sin_sixteen_pi": sin_sixteen_pi,
    "square": square,
    "two_parabolas": two_parabolas,
    "circle": circle,
    "ellipse": ellipse,
    "diamond": diamond,
    "multiplicative_noise": multiplicative_noise,
    "multimodal_independence": multimodal_independence,
}


def indep_sim(sim, n, p, **kwargs):
    """
    Allows choice for which simulation the user
    """
    if sim in INDEPENDENCE_SIMS.keys():
        sim_function = INDEPENDENCE_SIMS[sim]
    else:
        raise ValueError(
            "sim_name must be one of the following: {}".format(
                list(INDEPENDENCE_SIMS.keys())
            )
        )

    return sim_function(n, p, **kwargs)


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

import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import pairwise_distances

from scipy.stats.distributions import chi2
from hyppo.independence.base import IndependenceTest
from hyppo.independence.dcorr import _dcorr
from hyppo.independence._utils import sim_matrix


FOREST_TYPES = {
    "classifier": RandomForestClassifier,
    "regressor": RandomForestRegressor,
}


class KMERF(IndependenceTest):
    r"""
    Class for calculating the random forest based Dcorr test statistic and p-value.
    """

    def __init__(self, forest="regressor", ntrees=500, **kwargs):
        if forest in FOREST_TYPES.keys():
            self.clf = FOREST_TYPES[forest](n_estimators=ntrees, **kwargs)
        else:
            raise ValueError("forest must be one of the following ")
        IndependenceTest.__init__(self)

    def statistic(self, x, y):
        r"""
        Helper function that calculates the random forest based Dcorr test statistic.
        """
        self.clf.fit(x, y)
        self.distx = 1 - sim_matrix(self.clf, x)
        #if x.shape[1] == 1:
        #    self.disty = 1 - sim_matrix(self.clf, y.reshape(-1, 1))
        #else:
        self.disty = pairwise_distances(y.reshape(-1, 1), metric="euclidean")
        stat = _dcorr(self.distx, self.disty, bias=False, is_fast=False)

        return stat

    def test(self, x, y):
        n = x.shape[0]
        y = y.reshape(-1, 1)
        stat = self.statistic(x, y)
        statx = _dcorr(self.distx, self.distx, bias=False, is_fast=False)
        staty = _dcorr(self.disty, self.disty, bias=False, is_fast=False)
        pvalue = chi2.sf(stat / np.sqrt(statx * staty) * n + 1, 1)
        return stat, pvalue

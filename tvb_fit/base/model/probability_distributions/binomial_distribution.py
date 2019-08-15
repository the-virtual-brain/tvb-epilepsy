from collections import OrderedDict
import numpy as np
import numpy.random as nr
import scipy.stats as ss
from tvb_fit.base.model.probability_distributions import ProbabilityDistributionTypes
from tvb_fit.base.model.probability_distributions.discrete_probability_distribution \
    import DiscreteProbabilityDistribution

from tvb_utils.data_structures_utils import make_float, make_int


class BinomialDistribution(DiscreteProbabilityDistribution):

    def __init__(self, **params):
        self.type = ProbabilityDistributionTypes.BINOMIAL
        self.scipy_name = "binom"
        self.numpy_name = ProbabilityDistributionTypes.BINOMIAL
        self.constraint_string = "n > 0 and 0 < p < 1"
        self.n = make_int(params.get("n", 1))
        self.p = make_float(params.get("p", 0.5))
        self.__update_params__(n=self.n, p=self.p)

    def pdf_params(self):
        p = OrderedDict()
        p.update(zip(["n", "p"], [self.n, self.p]))
        return p

    def scale_params(self, loc=0.0, scale=1.0):
        return self.n, self.p

    def update_params(self, loc=0.0, scale=1.0, use="scipy", **params):
        self.__update_params__(loc, scale, use,
                               n=make_int(params.get("n", self.n)), p=make_float(params.get("p", self.p)))

    def constraint(self):
        # By default expr >= 0
        p = np.array(self.p).flatten() - np.finfo(np.float64).eps
        return np.hstack([np.array(self.n).flatten() - np.finfo(np.float64).eps, p, 1.0 - p + np.finfo(np.float64).eps])

    def _scipy(self, loc=0.0, scale=1.0):
        return ss.binom(n=self.n, p=self.p, loc=loc)

    def _numpy(self, loc=0.0, scale=1.0, size=(1,)):
        return lambda: nr.binomial(n=self.n, p=self.p, size=size) + loc

    def calc_mean_manual(self, loc=0.0, scale=1.0):
        return self.n * self.p + loc

    def calc_median_manual(self, loc=0.0, scale=1.0):
        return make_int(np.round(self.calc_mean_manual(loc=loc)))

    def calc_mode_manual(self, loc=0.0, scale=1.0):
        return make_int(np.round((self.n + 1) * self.p + loc) - 1)

    def calc_var_manual(self, loc=0.0, scale=1.0):
        return self.n * self.p * (1 - self.p)

    def calc_std_manual(self, loc=0.0, scale=1.0):
        return np.sqrt(self.calc_var_manual())

    def calc_skew_manual(self, loc=0.0, scale=1.0):
        return (1.0 - 2.0 * self.p) / self.calc_std_manual()

    def calc_kurt_manual(self, loc=0.0, scale=1.0):
        return (1.0 - 6.0 * self.p * (1.0 - self.p)) / self.calc_var_manual()

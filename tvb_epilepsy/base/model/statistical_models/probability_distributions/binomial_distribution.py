
import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.utils.data_structures_utils import make_float, make_int
from tvb_epilepsy.base.model.statistical_models.probability_distributions.discrete_probability_distribution  \
                                                                                  import DiscreteProbabilityDistribution


class BinomialDistribution(DiscreteProbabilityDistribution):

    def __init__(self, n=1, p=0.5):
        self.name = "bionomial"
        self.scipy_name = "binom"
        self.numpy_name = "binomial"
        self.constraint_string = "n > 0 and 0 < p < 1"
        self.n = make_int(n)
        self.p = make_float(p)
        self.__update_params__(n=self.n, p=self.p)

    def params(self):
        return {"n": self.n, "p": self.p}

    def update_params(self, n=1, p=0.5):
        self.__update_params__(n=make_int(n), p=make_float(p))

    def constraint(self):
        # By default expr >= 0
        p = np.array(self.p).flatten() - np.finfo(np.float64).eps
        return np.hstack([np.array(self.n).flatten() - 1, p, 1.0 - p + np.finfo(np.float64).eps])

    def scipy(self, loc=0.0, scale=1.0):
        return ss.binom(n=self.n, p=self.p, loc=loc, scale=scale)

    def numpy(self, size=(1,)):
        return lambda: nr.binomial(n=self.n, p=self.p, size=size)

    def calc_mean_manual(self):
        return self.n * self.p

    def calc_median_manual(self):
        return make_int(np.round(self.calc_mean_manual()))

    def calc_mode_manual(self):
        return make_int(np.round((self.n + 1) * self.p) - 1)

    def calc_var_manual(self):
        return self.n * self.p * (1 - self.p)

    def calc_std_manual(self):
        return np.sqrt(self.calc_var_manual())

    def calc_skew_manual(self):
        return (1.0 - 2.0 * self.p) / self.calc_std_manual()

    def calc_kurt_manual(self):
        return (1.0 - 6.0 * self.p * (1.0-self.p)) / self.calc_var_manual()

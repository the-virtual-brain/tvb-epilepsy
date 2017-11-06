
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
        self.params = {"n": make_int(n), "p": make_float(p)}
        self.constraint_string = "n > 0 and 0 < p < 1"
        self.__update_params__(**self.params)

    def update_params(self, **params):
        self.__update_params__(**self.params)

    def constraint(self):
        return np.all(self.params["n"] > 0) and np.all(self.params["p"] > 0.0) and np.all(self.params["p"] < 1.0)

    def scipy(self, loc=0.0, scale=1.0):
        return ss.binom(self.params["n"], self.params["p"], loc=loc, scale=scale)

    def numpy(self, size=(1,)):
        return lambda: nr.binomial(self.params["n"], self.params["p"], size=size)

    def calc_mean_manual(self):
        return self.params["n"] * self.params["p"]

    def calc_median_manual(self):
        return make_int(np.round(self.calc_mean_manual()))

    def calc_mode_manual(self):
        return make_int(np.round((self.params["n"] + 1) * self.params["p"]) - 1)

    def calc_var_manual(self):
        return self.params["n"] * self.params["p"] * (1 - self.params["p"])

    def calc_std_manual(self):
        return np.sqrt(self.calc_var_manual())

    def calc_skew_manual(self):
        return (1.0 - 2.0 * self.params["p"]) / self.calc_std_manual()

    def calc_kurt_manual(self):
        return (1.0 - 6.0 * self.params["p"] * (1.0-self.params["p"])) / self.calc_var_manual()


import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, make_float
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class BetaDistribution(ContinuousProbabilityDistribution):

    def __init__(self, alpha=1.0, beta=1.0):
        self.name = "beta"
        self.scipy_name = "beta"
        self.numpy_name = "beta"
        self.params = {"alpha": make_float(alpha), "beta": make_float(beta)}
        self.constraint_string = "alpha > 0 and beta > 0"
        self.__update_params__(**self.params)

    def update_params(self, **params):
        self.__update_params__(**self.params)

    def constraint(self):
        return np.all(self.params["alpha"] > 0.0) and np.all(self.params["beta"] > 0.0)

    def scipy(self, loc=0.0, scale=1.0):
        return ss.beta(self.params["alpha"], self.params["beta"], loc=loc, scale=scale)

    def numpy(self, size=(1,)):
        return lambda: nr.beta(self.params["alpha"], self.params["beta"], size=size)

    def calc_mean_manual(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="m")
        else:
            return self.params["alpha"] / (self.params["alpha"] + self.params["beta"])

    def calc_median_manual(self, use="scipy"):
        id = self.params["alpha"] > 1.0 and self.params["beta"] > 1.0
        if np.any(id==False):
            warning("No closed form of median for beta distribution for alpha or beta <= 1.0!" + "\nReturning nan!")
            alpha = np.array(self.params["alpha"])
            beta = np.array(self.params["beta"])
            median = np.nan((alpha+beta).shape)
            id = np.where(id)[0]
            median[id] = (alpha[id] - 1.0/3) / (alpha[id] + beta[id] - 2.0/3)
        else:
            warning("Approximate calculation for median of beta distribution!")
            median = (self.params["alpha"] - 1.0/3) / (self.params["alpha"] + self.params["beta"] - 2.0/3)
        return median

    def calc_mode_manual(self):
        id = self.params["alpha"] > 1.0 and self.params["beta"] > 1.0
        if np.any(id==False):
            warning("No closed form of mode for beta distribution for alpha or beta <= 1.0!" + "\nReturning nan!")
            alpha = np.array(self.params["alpha"])
            beta = np.array(self.params["beta"])
            mode = np.nan((alpha + beta).shape)
            id = np.where(id)[0]
            mode[id] =  (alpha[id] - 1.0) / (alpha[id] + beta[id] - 2.0)
        else:
            mode = (self.params["alpha"] - 1.0) / (self.params["alpha"] + self.params["beta"] - 2.0)
        return mode

    def calc_var_manual(self):
        a_plus_b = self.params["alpha"] + self.params["beta"]
        return self.params["alpha"] * self.params["beta"] / a_plus_b ** 2 / (a_plus_b + 1.0)

    def calc_std_manual(self):
        return np.sqrt(self.calc_var_manual())

    def calc_skew_manual(self):
        a_plus_b = self.params["alpha"] + self.params["beta"]
        return 2.0 * (self.params["beta"] - self.params["alpha"]) * np.sqrt(a_plus_b + 1.0) / \
               (a_plus_b + 2) / np.sqrt(self.params["alpha"] * self.params["beta"])

    def calc_kurt_manual(self):
        a_plus_b = self.params["alpha"] + self.params["beta"]
        ab = self.params["alpha"] * self.params["beta"]
        return 6.0 * (((self.params["alpha"] - self.params["beta"]) ** 2) * (a_plus_b + 1.0) - ab * (a_plus_b + 2.0)) \
               / ab / (a_plus_b + 2.0) / (a_plus_b + 3.0)

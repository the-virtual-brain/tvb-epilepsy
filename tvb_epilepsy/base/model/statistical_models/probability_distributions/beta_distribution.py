
from collections import OrderedDict

import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, make_float
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class BetaDistribution(ContinuousProbabilityDistribution):

    def __init__(self, **params):
        self.name = "beta"
        self.scipy_name = "beta"
        self.numpy_name = "beta"
        self.constraint_string = "alpha > 0 and beta > 0"
        self.alpha = make_float(params.get("alpha", params.get("a", 1.0)))
        self.beta = make_float(params.get("beta", params.get("b", 1.0)))
        self.a = self.alpha
        self.b = self.beta
        self.__update_params__(alpha=self.alpha, beta=self.beta)

    def pdf_params(self, parametrization="alpha-beta"):
        p = OrderedDict()
        if isequal_string(parametrization, "a-b") or \
           isequal_string(parametrization, "scipy") or \
           isequal_string(parametrization, "numpy"):
            p.update(zip(["a", "b"], [self.a, self.b]))
            return p
        else:
            p.update(zip(["alpha", "beta"], [self.alpha, self.beta]))
            return p

    def update_params(self, **params):
        self.__update_params__(alpha=make_float(params.get("alpha", params.get("a", self.alpha))),
                               beta=make_float(params.get("beta", params.get("b", self.beta))))
        self.a = self.alpha
        self.b = self.beta

    def constraint(self):
        # By default expr >= 0
        return np.hstack([np.array(self.alpha).flatten() - np.finfo(np.float64).eps,
                         np.array(self.beta).flatten() - np.finfo(np.float64).eps])

    def scipy(self, loc=0.0, scale=1.0):
        return ss.beta(a=self.alpha, b=self.beta, loc=loc, scale=scale)

    def numpy(self, size=(1,)):
        return lambda: nr.beta(a=self.alpha, b=self.beta, size=size)

    def calc_mean_manual(self, use="scipy"):
        if isequal_string(use, "scipy"):
            return self.scipy().stats(moments="m")
        else:
            return self.alpha / (self.alpha + self.beta)

    def calc_median_manual(self, use="scipy"):
        id = self.alpha > 1.0 and self.beta > 1.0
        if np.any(id==False):
            warning("No closed form of median for beta distribution for alpha or beta <= 1.0!" + "\nReturning nan!")
            alpha = np.array(self.alpha)
            beta = np.array(self.beta)
            median = np.nan((alpha+beta).shape)
            id = np.where(id)[0]
            median[id] = (alpha[id] - 1.0/3) / (alpha[id] + beta[id] - 2.0/3)
        else:
            warning("Approximate calculation for median of beta distribution!")
            median = (self.alpha - 1.0/3) / (self.alpha + self.beta - 2.0/3)
        return median

    def calc_mode_manual(self):
        id = self.alpha > 1.0 and self.beta > 1.0
        if np.any(id==False):
            warning("No closed form of mode for beta distribution for alpha or beta <= 1.0!" + "\nReturning nan!")
            alpha = np.array(self.alpha)
            beta = np.array(self.beta)
            mode = np.nan((alpha + beta).shape)
            id = np.where(id)[0]
            mode[id] =  (alpha[id] - 1.0) / (alpha[id] + beta[id] - 2.0)
        else:
            mode = (self.alpha - 1.0) / (self.alpha + self.beta - 2.0)
        return mode

    def calc_var_manual(self):
        a_plus_b = self.alpha + self.beta
        return self.alpha * self.beta / a_plus_b ** 2 / (a_plus_b + 1.0)

    def calc_std_manual(self):
        return np.sqrt(self.calc_var_manual())

    def calc_skew_manual(self):
        a_plus_b = self.alpha + self.beta
        return 2.0 * (self.beta - self.alpha) * np.sqrt(a_plus_b + 1.0) / \
               (a_plus_b + 2) / np.sqrt(self.alpha * self.beta)

    def calc_kurt_manual(self):
        a_plus_b = self.alpha + self.beta
        ab = self.alpha * self.beta
        return 6.0 * (((self.alpha - self.beta) ** 2) * (a_plus_b + 1.0) - ab * (a_plus_b + 2.0)) \
               / ab / (a_plus_b + 2.0) / (a_plus_b + 3.0)

from collections import OrderedDict
import numpy as np
import numpy.random as nr
import scipy.stats as ss
from tvb_fit.base.model.probability_distributions import ProbabilityDistributionTypes
from tvb_fit.base.utils.data_structures_utils import isequal_string, make_float
from tvb_fit.base.model.probability_distributions.continuous_probability_distribution \
    import ContinuousProbabilityDistribution


class BetaDistribution(ContinuousProbabilityDistribution):

    def __init__(self, **params):
        self.type = ProbabilityDistributionTypes.BETA
        self.scipy_name = ProbabilityDistributionTypes.BETA
        self.numpy_name = ProbabilityDistributionTypes.BETA
        self.constraint_string = "alpha > 0 and beta > 0"
        self.alpha = make_float(params.get("alpha", params.get("a", 2.0)))
        self.beta = make_float(params.get("beta", params.get("b", 2.0)))
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

    def scale_params(self, loc=0.0, scale=1.0):
        return self.alpha, self.beta

    def update_params(self, loc=0.0, scale=1.0, use="scipy", **params):
        self.__update_params__(loc, scale, use,
                               alpha=make_float(params.get("alpha", params.get("a", self.alpha))),
                               beta=make_float(params.get("beta", params.get("b", self.beta))))
        self.a = self.alpha
        self.b = self.beta

    def constraint(self):
        # By default expr >= 0
        return np.hstack([np.array(self.alpha).flatten() - np.finfo(np.float64).eps,
                         np.array(self.beta).flatten() - np.finfo(np.float64).eps])

    def _scipy(self, loc=0.0, scale=1.0):
        return ss.beta(a=self.alpha, b=self.beta, loc=loc, scale=scale)

    def _numpy(self, loc=0.0, scale=1.0, size=(1,)):
        return lambda: nr.beta(a=self.alpha, b=self.beta, size=size) * scale + loc

    def calc_mean_manual(self, loc=0.0, scale=1.0):
        return self.alpha / (self.alpha + self.beta) + loc

    def calc_median_manual(self, loc=0.0, scale=1.0):
        shape = (self.a + self.b).shape
        i1 = np.ones((1,))
        alpha = self.alpha * i1
        beta = self.beta * i1
        id = np.logical_and(self.alpha > 1.0, self.beta > 1.0)
        if np.any(id==False):
            self.logger.warning("No closed form of median for beta distribution "
                                "for alpha or beta <= 1.0!" + "\nReturning nan!")
            median = np.nan((alpha+beta).shape)
            id = np.where(id)[0]
            median[id] = (alpha[id] - 1.0/3) / (alpha[id] + beta[id] - 2.0/3)
            return np.reshape(median, shape) + loc
        else:
            # TODO: find a way to mute this warning...
            # self.logger.warning("Approximate calculation for median of beta distribution!")
            return (self.alpha - 1.0/3) / (self.alpha + self.beta - 2.0/3) + loc

    def calc_mode_manual(self, loc=0.0, scale=1.0):
        shape = (self.a + self.b).shape
        i1 = np.ones((1,))
        alpha = self.alpha * i1
        beta = self.beta * i1
        id = np.logical_and(self.alpha > 1.0, self.beta > 1.0)
        if np.any(id==False):
            self.logger.warning("No closed form of mode for beta distribution for alpha or beta <= 1.0!" + "\nReturning nan!")
            mode = np.nan * np.ones((alpha + beta).shape)
            id = np.where(id)[0]
            mode[id] = (alpha[id] - 1.0) / (alpha[id] + beta[id] - 2.0)
            return np.reshape(mode, shape) + loc
        else:
            return(self.alpha - 1.0) / (self.alpha + self.beta - 2.0) + loc

    def calc_var_manual(self, loc=0.0, scale=1.0):
        a_plus_b = self.alpha + self.beta
        return (self.alpha * self.beta / a_plus_b ** 2 / (a_plus_b + 1.0)) * scale**2

    def calc_std_manual(self, loc=0.0, scale=1.0):
        return np.sqrt(self.calc_var_manual(loc, scale))

    def calc_skew_manual(self, loc=0.0, scale=1.0):
        a_plus_b = self.alpha + self.beta
        return 2.0 * (self.beta - self.alpha) * np.sqrt(a_plus_b + 1.0) / \
               (a_plus_b + 2) / np.sqrt(self.alpha * self.beta)

    def calc_kurt_manual(self, loc=0.0, scale=1.0):
        a_plus_b = self.alpha + self.beta
        ab = self.alpha * self.beta
        return 6.0 * (((self.alpha - self.beta) ** 2) * (a_plus_b + 1.0) - ab * (a_plus_b + 2.0)) \
               / ab / (a_plus_b + 2.0) / (a_plus_b + 3.0)

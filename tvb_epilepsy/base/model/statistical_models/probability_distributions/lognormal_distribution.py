
from collections import OrderedDict

import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.utils.data_structures_utils import make_float, isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class LognormalDistribution(ContinuousProbabilityDistribution):

    def __init__(self, **params):
        self.type = "lognormal"
        self.scipy_name = "lognorm"
        self.numpy_name = "lognormal"
        self.constraint_string = "sigma > 0"
        self.mu = make_float(params.get("mu", np.log(params.get("scale", 1.0))))
        self.sigma = make_float(params.get("sigma", params.get("shape", 1.0)))
        self.__update_params__(mu=self.mu, sigma=self.sigma)
        self.shape = self.sigma
        self.scale = np.exp(self.mu)

    def pdf_params(self, parametrization="mu-sigma"):
        p = OrderedDict()
        if isequal_string(parametrization, "scipy") or isequal_string(parametrization, "numpy"):
            p.update(zip(["shape", "scale"], [self.shape, self.scale]))
            return p
        else:
            p.update(zip(["mu", "sigma"], [self.mu, self.sigma]))
            return p
        
    def update_params(self, **params):
        self.__update_params__(mu=make_float(params.get("mu", np.log(params.get("scale", self.scale))),
                               sigma=make_float(params.get("sigma", params.get("shape", self.shape)))))
        self.shape = self.sigma
        self.scale = np.exp(self.mu)
        
    def constraint(self):
        # By default expr >= 0
        return np.array(self.sigma).flatten() - np.finfo(np.float64).eps

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(s=self.sigma, loc=loc, scale=np.exp(self.mu))

    def numpy(self, size=(1,)):
        return lambda: nr.lognormal(mean=self.mu, sigma=self.sigma, size=size)

    def calc_mean_manual(self):
        return np.exp(self.mu + self.sigma**2 / 2.0)

    def calc_median_manual(self):
        return np.exp(self.mu)

    def calc_mode_manual(self):
        return np.exp(self.mu - self.sigma**2)

    def calc_var_manual(self):
        sigma2 = self.sigma ** 2
        return (np.exp(sigma2) - 1.0) * np.exp(2.0 * self.mu + sigma2)

    def calc_std_manual(self):
        return np.sqrt(self.calc_var_manual())

    def calc_skew_manual(self):
        sigma2exp = np.exp(self.sigma**2)
        return (sigma2exp + 2.0) * np.sqrt(sigma2exp - 1.0)

    def calc_kurt_manual(self):
        sigma2 = self.sigma ** 2
        return np.exp(4.0 * sigma2) + 2.0 * np.exp(3.0 * sigma2) + 3.0 * np.exp(2.0 * sigma2) - 6.0

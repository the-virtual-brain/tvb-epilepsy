
from collections import OrderedDict

import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.utils.data_structures_utils import make_float, isequal_string, construct_import_path
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
        self.context_str = "from " + construct_import_path(__file__) + " import LognormalDistribution"
        self.create_str = "LognormalDistribution('" + self.type + "')"
        self.update_str = "obj.update_params()"

    def pdf_params(self, parametrization="mu-sigma"):
        p = OrderedDict()
        if isequal_string(parametrization, "scipy") or isequal_string(parametrization, "numpy"):
            p.update(zip(["shape", "scale"], [self.shape, np.exp(self.mu)]))
            return p
        else:
            p.update(zip(["mu", "sigma"], [self.mu, self.sigma]))
            return p

    def scale_params(self, loc=0.0, scale=1.0):
        scale = np.exp(self.mu) * scale
        mu = np.log(scale)
        return mu, self.sigma

    def update_params(self, loc=0.0, scale=1.0, use="scipy", **params):
        self.__update_params__(loc, scale, use,
                               mu=make_float(params.get("mu", np.log(params.get("scale", np.exp(self.mu))))),
                               sigma=make_float(params.get("sigma", params.get("shape", self.shape))))
        self.shape = self.sigma

    def constraint(self):
        # By default expr >= 0
        return np.array(self.sigma).flatten() - np.finfo(np.float64).eps

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(s=self.sigma, loc=loc, scale=np.exp(self.mu)*scale)

    def numpy(self, loc=0.0, scale=1.0, size=(1,)):
        mu = self.scale_params(loc, scale)[0]
        return lambda: nr.lognormal(mean=self.mu, sigma=self.sigma, size=size) + loc

    def calc_mean_manual(self, loc=0.0, scale=1.0):
        mu = self.scale_params(loc, scale)[0]
        return np.exp(mu + self.sigma**2 / 2.0) + loc

    def calc_median_manual(self, loc=0.0, scale=1.0):
        mu = self.scale_params(loc, scale)[0]
        return np.exp(mu) + loc

    def calc_mode_manual(self, loc=0.0, scale=1.0):
        mu = self.scale_params(loc, scale)[0]
        return np.exp(mu - self.sigma**2) + loc

    def calc_var_manual(self, loc=0.0, scale=1.0):
        mu = self.scale_params(loc, scale)[0]
        sigma2 = self.sigma ** 2
        return (np.exp(sigma2) - 1.0) * np.exp(2.0 * mu + sigma2)

    def calc_std_manual(self, loc=0.0, scale=1.0):
        return np.sqrt(self.calc_var_manual(loc, scale))

    def calc_skew_manual(self, loc=0.0, scale=1.0):
        sigma2exp = np.exp(self.sigma**2)
        return (sigma2exp + 2.0) * np.sqrt(sigma2exp - 1.0)

    def calc_kurt_manual(self, loc=0.0, scale=1.0):
        sigma2 = self.sigma ** 2
        return np.exp(4.0 * sigma2) + 2.0 * np.exp(3.0 * sigma2) + 3.0 * np.exp(2.0 * sigma2) - 6.0

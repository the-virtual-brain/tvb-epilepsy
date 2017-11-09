
import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.utils.data_structures_utils import make_float, isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class LognormalDistribution(ContinuousProbabilityDistribution):

    def __init__(self, **params):
        self.name = "lognormal"
        self.scipy_name = "lognorm"
        self.numpy_name = "lognormal"
        self.constraint_string = "sigma > 0"
        self.mean = make_float(params.get("mean", np.log(params.get("scale", 1.0))))
        self.sigma = make_float(params.get("sigma", params.get("shape", 1.0)))
        self.__update_params__(mean=self.mean, sigma=self.sigma)
        self.shape = self.sigma
        self.scale = np.exp(self.mean)

    def params(self, parametrization="mean-sigma"):
        if isequal_string(parametrization, "scipy"):
            return {"shape": self.shape, "scale": self.scale}
        else:
            return {"mean": self.mean, "sigma": self.sigma}
        
    def update_params(self, **params):
        self.__update_params__(mean=make_float(params.get("mean", np.log(params.get("scale", self.scale))),
                               sigma=make_float(params.get("sigma", params.get("shape", self.shape)))))
        self.shape = self.sigma
        self.scale = np.exp(self.mean)
        
    def constraint(self):
        # By default expr >= 0
        return np.array(self.sigma).flatten() - np.finfo(np.float64).eps

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(s=self.sigma, loc=loc, scale=np.exp(self.mean))

    def numpy(self, size=(1,)):
        return lambda: nr.lognormal(mean=self.mean, sigma=self.sigma, size=size)

    def calc_mean_manual(self):
        return np.exp(self.mean + self.sigma ** 2 / 2.0)

    def calc_median_manual(self):
        return np.exp(self.mean)

    def calc_mode_manual(self):
        return np.exp(self.mean - self.sigma ** 2)

    def calc_var_manual(self):
        sigma2 = self.sigma ** 2
        return (np.exp(sigma2) - 1.0) * np.exp(2.0 * self.mean + self.sigma ** 2)

    def calc_std_manual(self):
        return np.sqrt(self.calc_var_manual())

    def calc_skew_manual(self):
        sigma2exp = np.exp(self.sigma ** 2)
        return (sigma2exp + 2.0) * np.sqrt(sigma2exp - 1.0)

    def calc_kurt_manual(self):
        sigma2 = self.sigma ** 2
        return np.exp(4.0 * sigma2) + 2.0 * np.exp(3.0 * sigma2) + 3.0 * np.exp(2.0 * sigma2) - 6.0

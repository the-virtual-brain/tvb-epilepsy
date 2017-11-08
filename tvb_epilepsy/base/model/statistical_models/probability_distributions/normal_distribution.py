
import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.utils.data_structures_utils import make_float, isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class NormalDistribution(ContinuousProbabilityDistribution):

    def __init__(self, mean=0.0, sigma=1.0):
        self.name = "normal"
        self.scipy_name = "norm"
        self.numpy_name = "normal"
        self.mean = make_float(mean)
        self.sigma = make_float(sigma)
        self.constraint_string = "sigma > 0"
        self.__update_params__(mean=self.mean, sigma=self.sigma)

    def params(self, parametrization="mean-sigma"):
        if isequal_string(parametrization, "scipy") or isequal_string(parametrization, "numpy"):
            return {"loc": self.pdf_shape, "scale": self.scale}
        else:
            return {"mean": self.mean, "sigma": self.sigma}

    def update_params(self, **params):
        self.__update_params__(**params)

    def constraint(self):
        # By default expr >= 0
        return np.array(self.sigma).flatten() - np.finfo(np.float64).eps

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(loc=self.mean, scale=self.sigma)

    def numpy(self, size=(1,)):
        return lambda: nr.normal(self.mean, self.sigma, size=size)

    def calc_mean_manual(self):
        return self.mean

    def calc_median_manual(self, ):
        return self.mean

    def calc_mode_manual(self):
        return self.mean

    def calc_var_manual(self):
        return self.sigma ** 2

    def calc_std_manual(self, ):
        return self.sigma

    def calc_skew_manual(self):
        return 0.0

    def calc_kurt_manual(self):
        return 0.0

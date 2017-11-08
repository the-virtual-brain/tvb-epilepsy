
import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import make_float, isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class GammaDistribution(ContinuousProbabilityDistribution):

    def __init__(self, **params):
        self.name = "gamma"
        self.scipy_name = "gamma"
        self.numpy_name = "gamma"
        self.constraint_string = "shape > 0 and scale > 0"
        self.shape = make_float(params.get("shape", params.get("alpha", params.get("k", 1.0))))
        self.scale = make_float(params.get("scale", params.get("theta",
                                                               1.0 / params.get("beta", params.get("rate", 1.0)))))
        self.k = self.shape
        self.theta = self.scale
        self.alpha = self.shape
        self.beta = 1.0 / self.scale
        self.__update_params__(shape=self.shape, scale=self.scale)

    def __str__(self):
        this_str = super(GammaDistribution, self).__str__()
        this_str = this_str[0:-1]
        this_str += "\n" + "13. k" + " = " + str(self.k) + \
                    "\n" + "14. theta" + " = " + str(self.theta) + \
                    "\n" + "15. alpha" + " = " + str(self.alpha) + \
                    "\n" + "16. beta" + " = " + str(self.beta) + "}"
        return this_str

    def params(self, parametrization="shape-scale"):
        if isequal_string(parametrization, "alpha-beta"):
            return {"alpha": self.shape, "beta": self.beta}
        elif isequal_string(parametrization, "k-theta"):
            return {"k": self.k, "theta": self.theta}
        elif isequal_string(parametrization, "shape-rate"):
            return {"shape": self.shape, "rate": 1.0 / self.scale}
        elif isequal_string(parametrization, "scipy"):
            return {"a": self.shape, "scale": self.scale}
        else:
            return {"shape": self.shape, "scale": self.scale}

    def update_params(self, **params):
        self.__update_params__(shape=make_float(params.get("shape", params.get("alpha", params.get("k", 1.0)))),
                               scale=make_float(params.get("scale", params.get("theta",
                                                               1.0 / params.get("beta", params.get("rate", 1.0))))))
        self.k = self.shape
        self.theta = self.scale
        self.alpha = self.shape
        self.beta = 1.0 / self.scale

    def constraint(self):
        # By default expr >= 0
        return np.hstack([np.array(self.shape).flatten() - np.finfo(np.float64).eps,
                          np.array(self.scale).flatten() - np.finfo(np.float64).eps])

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(a=self.shape, loc=loc, scale=self.scale)

    def numpy(self, size=(1,)):
        return lambda: nr.gamma(shape=self.shape, scale=self.scale, size=size)

    def calc_mean_manual(self):
        return self.shape * self.scale

    def calc_median_manual(self):
        warning("Gamma distribution does not have a simple closed form median! Returning nan!")
        return np.nan

    def calc_mode_manual(self):
        if self.shape >= 1.0:
            return (self.shape - 1.0) * self.scale
        else:
            warning("Mode cannot be calculate for gamma distribution when the shape parameter is smaller than 1.0! "
                    "Returning nan!")
            return np.nan

    def calc_var_manual(self):
        return self.shape * self.scale ** 2

    def calc_std_manual(self):
       return np.sqrt(self.shape) * self.scale

    def calc_skew_manual(self):
        return 2.0 / np.sqrt(self.shape)

    def calc_kurt_manual(self):
        return 6.0 / self.shape

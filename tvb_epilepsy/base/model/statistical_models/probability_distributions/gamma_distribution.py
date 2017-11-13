
from collections import OrderedDict

import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import make_float, isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class GammaDistribution(ContinuousProbabilityDistribution):

    def __init__(self, **params):
        self.type = "gamma"
        self.scipy_name = "gamma"
        self.numpy_name = "gamma"
        self.constraint_string = "shape > 0 and scale > 0"
        self.shape = make_float(params.get("shape", params.get("alpha", params.get("k", 2.0))))
        self.scale = make_float(params.get("scale", params.get("theta",
                                                               1.0 / params.get("beta", params.get("rate", 0.5)))))
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

    def pdf_params(self, parametrization="shape-scale"):
        p = OrderedDict()
        if isequal_string(parametrization, "alpha-beta"):
            p.update(zip(["alpha", "beta"], [self.alpha, self.beta]))
            return p
        elif isequal_string(parametrization, "k-theta"):
            p.update(zip(["k", "theta"], [self.k, self.theta]))
            return p
        elif isequal_string(parametrization, "shape-rate"):
            p.update(zip(["shape", "rate"], [self.shape, 1.0 / self.scale]))
            return p
        elif isequal_string(parametrization, "scipy"):
            p.update(zip(["a", "scale"], [self.shape, self.scale]))
            return p
        else:
            p.update(zip(["shape", "scale"], [self.shape, self.scale]))
            return p

    def update_params(self, **params):
        self.__update_params__(shape=make_float(params.get("shape", params.get("alpha", params.get("k", self.shape)))),
                               scale=make_float(params.get("scale",
                                         params.get("theta", 1.0 / params.get("beta", params.get("rate", self.beta))))))
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
        orig_shape = np.array(self.shape + self.scale).shape
        i1 = np.ones((1,))
        shape = self.shape * i1
        scale = self.scale * i1
        id = shape >= 1.0
        if not(np.all(id)):
            warning("Mode cannot be calculated for gamma distribution when the shape parameter is smaller than 1.0! "
                    "Returning nan!")
            mode = np.nan * np.ones((shape + scale).shape)
            id = np.where(id)[0]
            mode[id] = (shape[id] - 1.0) * scale[id]
            return np.reshape(mode, orig_shape)
        else:
            return (self.shape - 1.0) * self.scale

    def calc_var_manual(self):
        return self.shape * self.scale ** 2

    def calc_std_manual(self):
       return np.sqrt(self.shape) * self.scale

    def calc_skew_manual(self):
        return 2.0 / np.sqrt(self.shape)

    def calc_kurt_manual(self):
        return 6.0 / self.shape

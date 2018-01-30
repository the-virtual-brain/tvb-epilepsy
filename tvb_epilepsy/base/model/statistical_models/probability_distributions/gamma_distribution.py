from collections import OrderedDict
import numpy as np
import numpy.random as nr
import scipy.stats as ss
from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import make_float, isequal_string, construct_import_path
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution \
    import ContinuousProbabilityDistribution


class GammaDistribution(ContinuousProbabilityDistribution):

    def __init__(self, **params):
        self.type = "gamma"
        self.scipy_name = "gamma"
        self.numpy_name = "gamma"
        self.constraint_string = "alpha > 0 and beta > 0"
        self.alpha = make_float(params.get("alpha", params.get("k", params.get("shape", 2.0))))
        self.beta = make_float(params.get("beta", params.get("rate",
                                                             1.0 / params.get("theta", params.get("scale", 0.5)))))
        self.k = self.alpha
        self.theta = 1.0 / self.beta
        self.__update_params__(alpha=self.alpha, theta=self.theta)

    def __str__(self):
        this_str = super(GammaDistribution, self).__str__()
        this_str = this_str[0:-1]
        this_str += "\n" + "13. k" + " = " + str(self.k) + \
                    "\n" + "14. theta" + " = " + str(self.theta) + \
                    "\n" + "15. alpha" + " = " + str(self.alpha) + \
                    "\n" + "16. beta" + " = " + str(self.beta) + "}"
        return this_str

    def pdf_params(self, parametrization="alpha-beta"):
        p = OrderedDict()
        if isequal_string(parametrization, "shape-scale"):
            p.update(zip(["shape", "scale"], [self.alpha, self.theta]))
            return p
        elif isequal_string(parametrization, "k-theta"):
            p.update(zip(["k", "theta"], [self.k, self.theta]))
            return p
        elif isequal_string(parametrization, "shape-rate"):
            p.update(zip(["shape", "rate"], [self.alpha, self.beta]))
            return p
        elif isequal_string(parametrization, "scipy"):
            p.update(zip(["a", "scale"], [self.alpha, self.theta]))
            return p
        else:
            p.update(zip(["alpha", "beta"], [self.alpha, self.beta]))
            return p

    def scale_params(self, loc=0.0, scale=1.0):
        return self.alpha, self.beta / scale

    def update_params(self, loc=0.0, scale=1.0, use="scipy", **params):
        self.__update_params__(loc, scale, use,
                               alpha=make_float(params.get("alpha", params.get("k", params.get("shape", self.alpha)))),
                               beta=make_float(params.get("beta", 1.0 / params.get("theta",
                                                                                   params.get("scale",
                                                                                              params.get("rate",
                                                                                                   1.0 / self.beta))))))
        self.k = self.alpha
        self.theta = 1.0 / self.beta

    def constraint(self):
        # By default expr >= 0
        return np.hstack([np.array(self.alpha).flatten() - np.finfo(np.float64).eps,
                          np.array(self.beta).flatten() - np.finfo(np.float64).eps])

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(a=self.alpha, loc=loc, scale=self.theta * scale)

    def numpy(self, loc=0.0, scale=1.0, size=(1,)):
        return lambda: nr.gamma(shape=self.alpha, scale=self.theta * scale, size=size) + loc

    def calc_mean_manual(self, loc=0.0, scale=1.0):
        return self.alpha * self.theta * scale + loc

    def calc_median_manual(self, loc=0.0, scale=1.0):
        warning("Gamma distribution does not have a simple closed form median! Returning nan!")
        return np.nan

    def calc_mode_manual(self, loc=0.0, scale=1.0):
        orig_shape = np.array(self.alpha + self.theta).shape
        i1 = np.ones((1,))
        shape = self.alpha * i1
        np_scale = scale * self.theta * i1
        id = shape >= 1.0
        if not (np.all(id)):
            warning("Mode cannot be calculated for gamma distribution when the shape parameter is smaller than 1.0! "
                    "Returning nan!")
            mode = np.nan * np.ones((shape + np_scale).shape)
            id = np.where(id)[0]
            mode[id] = (shape[id] - 1.0) * np_scale[id]
            return np.reshape(mode, orig_shape) + loc
        else:
            return (self.alpha - 1.0) * self.theta * scale + loc

    def calc_var_manual(self, loc=0.0, scale=1.0):
        return self.alpha * (self.theta * scale) ** 2

    def calc_std_manual(self, loc=0.0, scale=1.0):
        return np.sqrt(self.alpha) * (self.theta * scale)

    def calc_skew_manual(self, loc=0.0, scale=1.0):
        return 2.0 / np.sqrt(self.alpha)

    def calc_kurt_manual(self, loc=0.0, scale=1.0):
        return 6.0 / self.alpha

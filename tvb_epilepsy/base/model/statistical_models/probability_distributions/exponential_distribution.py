
import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import make_float, isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution  \
                                                                                import ContinuousProbabilityDistribution


class ExponentialDistribution(ContinuousProbabilityDistribution):

    def __init__(self, **params):
        self.name = "exponential"
        self.scipy_name = "expon"
        self.numpy_name = "exponential"
        self.constraint_string = "scale > 0"
        self.scale = make_float(params.get("scale", 1.0/params.get("lamda", params.get("rate", 1.0))))
        self.lamda = 1.0 / self.scale
        self.rate = self.lamda
        self.__update_params__(scale=self.scale)

    def __str__(self):
        this_str = super(ExponentialDistribution, self).__str__()
        this_str = this_str[0:-1]
        this_str += "\n" + "13. rate or lamda" + " = " + str(self.rate) + "}"
        return this_str

    def params(self, parametrization="lamda"):
        if isequal_string(parametrization, "lamda"):
            return {"lamda": self.lamda}
        elif isequal_string(parametrization, "rate"):
            return {"rate": self.rate}
        else:
            return {"scale": self.scale}

    def update_params(self, **params):
        self.__update_params__(scale=make_float(params.get("scale", 1.0/params.get("lamda", params.get("rate", 1.0)))))
        self.lamda = 1.0 / self.scale
        self.rate = self.lamda

    def constraint(self):
        # By default expr >= 0
        return np.array(self.scale).flatten() - np.finfo(np.float64).eps

    def scipy(self, loc=0.0, scale=1.0):
        return ss.expon(loc=loc, scale=self.scale)

    def numpy(self, size=(1,)):
        return lambda: nr.exponential(scale=self.scale, size=size)

    def calc_mean_manual(self):
        return self.scale

    def calc_median_manual(self):
        warning("Approximate calculation for median of chisquare distribution!")
        return self.scale * np.log(2)

    def calc_mode_manual(self):
        return 0.0

    def calc_var_manual(self):
        return self.scale ** 2

    def calc_std_manual(self):
        return self.scale

    def calc_skew_manual(self):
        return 2.0

    def calc_kurt_manual(self):
        return 6.0

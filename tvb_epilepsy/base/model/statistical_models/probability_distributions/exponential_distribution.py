import numpy as np
import numpy.random as nr
import scipy.stats as ss
from tvb_epilepsy.base.model.statistical_models.probability_distributions import ProbabilityDistributionTypes
from tvb_epilepsy.base.utils.data_structures_utils import make_float, isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution \
    import ContinuousProbabilityDistribution


class ExponentialDistribution(ContinuousProbabilityDistribution):

    def __init__(self, **params):
        self.type = ProbabilityDistributionTypes.EXPONENTIAL
        self.scipy_name = "expon"
        self.numpy_name = ProbabilityDistributionTypes.EXPONENTIAL
        self.constraint_string = "scale > 0"
        self.lamda = make_float(params.get("lamda", params.get("rate", 1.0 / params.get("scale", 1.0))))
        self.rate = self.lamda
        self.__update_params__(lamda=self.lamda)

    def __str__(self):
        this_str = super(ExponentialDistribution, self).__str__()
        this_str = this_str[0:-1]
        this_str += "\n" + "13. rate or lamda" + " = " + str(self.lamda) + "}"
        return this_str

    def pdf_params(self, parametrization="lamda"):
        if isequal_string(parametrization, "scale"):
            return {"scale": 1.0 / self.lamda}
        elif isequal_string(parametrization, "rate"):
            return {"rate": self.rate}
        else:
            return {"lamda": self.lamda}

    def scale_params(self, loc=0.0, scale=1.0):
        return self.lamda / scale

    def update_params(self, loc=0.0, scale=1.0, use="scipy", **params):
        self.__update_params__(loc, scale, use,
                               lamda=make_float(params.get("lamda", params.get("rate", 1.0 / params.get("scale",
                                                                                                        1.0 / self.lamda)))))
        self.rate = self.lamda

    def constraint(self):
        # By default expr >= 0
        return np.array(1.0 / self.lamda).flatten() - np.finfo(np.float64).eps

    def scipy(self, loc=0.0, scale=1.0):
        return ss.expon(loc=loc, scale=scale / self.lamda)

    def numpy(self, loc=0.0, scale=1.0, size=(1,)):
        return lambda: nr.exponential(scale=scale / self.lamda, size=size) + loc

    def calc_mean_manual(self, loc=0.0, scale=1.0):
        return scale / self.lamda + loc

    def calc_median_manual(self, loc=0.0, scale=1.0):
        return scale / self.lamda * np.log(2) + loc

    def calc_mode_manual(self, loc=0.0, scale=1.0):
        return 0.0 + loc

    def calc_var_manual(self, loc=0.0, scale=1.0):
        return scale ** 2 / self.lamda ** 2

    def calc_std_manual(self, loc=0.0, scale=1.0):
        return scale / self.lamda

    def calc_skew_manual(self, loc=0.0, scale=1.0):
        return 2.0

    def calc_kurt_manual(self, loc=0.0, scale=1.0):
        return 6.0

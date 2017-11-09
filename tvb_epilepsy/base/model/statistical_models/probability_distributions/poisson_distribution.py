
import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import make_float, make_int, isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.discrete_probability_distribution  \
                                                                                  import DiscreteProbabilityDistribution


class PoissoniDistribution(DiscreteProbabilityDistribution):

    def __init__(self, **params):
        self.name = "poisson"
        self.scipy_name = "poisson"
        self.numpy_name = "poisson"
        self.lamda = make_float(params.get("lamda", params.get("lam", params.get("mu", 0.5))))
        self.constraint_string = "0 < lamda < 1"
        self.__update_params__(lamda=self.lamda)
        self.mu = self.lamda

    def params(self, parametrization="lamda"):
        if isequal_string(parametrization, "scipy"):
            return {"mu": self.mu}
        elif isequal_string(parametrization, "numpy"):
            return {"lam": self.mu}
        else:
            return {"lamda": self.lamda}

    def update_params(self, **params):
        self.__update_params__(lamda=make_float(params.get("lamda", params.get("lam", params.get("mu", self.lamda)))))
        self.mu = self.lamda

    def constraint(self):
        # By default expr >= 0
        lamda = np.array(self.lamda).flatten()
        return np.hstack([lamda - np.finfo(np.float64).eps, 1.0 - lamda + np.finfo(np.float64).eps])

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(self.lamda, loc=loc, scale=scale)

    def numpy(self, size=(1,)):
        return lambda: nr.poisson(self.lamda, size=size)

    def calc_mean_manual(self):
        return self.lamda

    def calc_median_manual(self):
        warning("Approximate calculation for median of poisson distribution!")
        return np.int(np.round(self.lamda + 1.0/3 - 0.02 / self.lamda))

    def calc_mode_manual(self):
        mode = np.ones(np.array(self.params["p"]).shape)
        mode[np.where(self.params["p"] < 0.5)[0]] = 0.0
        p05 = self.params["p"] == 0.5
        if np.any(p05):
            warning("The mode of poisson distribution for p=0.5 consists of two values (lamda-1 and lamda)!")
            mode = mode.astype('O')
            lamda = make_int(np.round(self.lamda))
            mode[np.where(p05)[0]] = (lamda - 1, lamda)

    def calc_var_manual(self):
        return self.lamda

    def calc_std_manual(self):
        return np.sqrt(self.calc_var_manual())

    def calc_skew_manual(self):
        return 1.0 / self.calc_std_manual()

    def calc_kurt_manual(self, use="scipy"):
        return 1.0 / self.lamda

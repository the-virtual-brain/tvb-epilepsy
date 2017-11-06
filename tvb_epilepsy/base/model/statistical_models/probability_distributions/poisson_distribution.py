
import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import make_float, make_int
from tvb_epilepsy.base.model.statistical_models.probability_distributions.discrete_probability_distribution  \
                                                                                  import DiscreteProbabilityDistribution


class PoissoniDistribution(DiscreteProbabilityDistribution):

    def __init__(self, lamda=0.5):
        self.name = "poisson"
        self.scipy_name = "poisson"
        self.numpy_name = "poisson"
        self.params = {"lamda": make_float(lamda)}
        self.constraint_string = "0 < lamda < 1"
        self.__update_params__(**self.params)

    def update_params(self, **params):
        self.__update_params__(**self.params)

    def constraint(self):
        return np.all(self.params["lamda"] > 0.0)

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(self.params["lamda"], loc=loc, scale=scale)

    def numpy(self, size=(1,)):
        return lambda: nr.exponential(self.params["lamda"], size=size)

    def calc_mean_manual(self):
        return self.params["lamda"]

    def calc_median_manual(self):
        warning("Approximate calculation for median of poisson distribution!")
        return np.int(np.round(self.params["lamda"] + 1.0/3 - 0.02 / self.params["lamda"]))

    def calc_mode_manual(self):
        mode = np.ones(np.array(self.params["p"]).shape)
        mode[np.where(self.params["p"] < 0.5)[0]] = 0.0
        p05 = self.params["p"] == 0.5
        if np.any(p05):
            warning("The mode of poisson distribution for p=0.5 consists of two values (lamda-1 and lamda)!")
            mode = mode.astype('O')
            lamda = make_int(np.round(self.params["lamda"]))
            mode[np.where(p05)[0]] = (lamda - 1, lamda)

    def calc_var_manual(self):
        return self.params["lamda"]

    def calc_std_manual(self):
        return np.sqrt(self.calc_var_manual())

    def calc_skew_manual(self):
        return 1.0 / self.calc_std_manual()

    def calc_exkurt_manual(self, use="scipy"):
        return 1.0 / self.params["lamda"]

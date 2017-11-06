
import numpy as np
import scipy.stats as ss

from tvb_epilepsy.base.utils.log_error_utils import warning, raise_not_implemented_error
from tvb_epilepsy.base.utils.data_structures_utils import make_float
from tvb_epilepsy.base.model.statistical_models.probability_distributions.discrete_probability_distribution  \
                                                                                  import DiscreteProbabilityDistribution


class BernoulliDistribution(DiscreteProbabilityDistribution):

    def __init__(self, p=0.5):
        self.name = "bernoulli"
        self.scipy_name = "bernoulli"
        self.numpy_name = ""
        self.params = {"p": make_float(p)}
        self.constraint_string = "0 < p < 1"
        self.__update_params__(**self.params)

    def update_params(self, **params):
        self.__update_params__(**self.params)

    def constraint(self):
        return np.all(self.params["p"] > 0.0) and np.all(self.params["p"] < 1.0)

    def scipy(self, loc=0.0, scale=1.0):
        return ss.bernoulli(self.params["p"], loc=loc, scale=scale)

    def numpy(self, size=(1,)):
        raise_not_implemented_error("No implementation of bernoulli distribution in numpy.random module!")

    def calc_mean_manual(self):
        return self.params["p"]

    def calc_median_manual(self):
        median = 0.5 * np.ones(np.array(self.params["p"]).shape)
        median[np.where(self.params["p"] < 0.5)[0]] = 0.0
        median[np.where(self.params["p"] > 0.5)[0]] = 1.0
        return median

    def calc_mode_manual(self):
        mode = np.ones(np.array(self.params["p"]).shape)
        mode[np.where(self.params["p"] < 0.5)[0]] = 0.0
        p05 = self.params["p"] == 0.5
        if np.any(p05):
            warning("The mode of bernoulli distribution for p=0.5 consists of two values (0.0 and 1.0)!")
            mode = mode.astype('O')
            mode[np.where(p05)[0]] = (0.0, 1.0)
        return mode


    def calc_var_manual(self):
        return self.params["p"] * (1 - self.params["p"])

    def calc_std_manual(self):
        return np.sqrt(self.calc_var_manual())

    def calc_skew_manual(self):
        return (1.0 - 2.0 * self.params["p"]) / self.calc_std_manual()

    def calc_kurt_manual(self):
        var = self.calc_var_manual()
        return (1.0 - 6.0 * var) / var

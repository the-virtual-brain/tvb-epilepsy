import numpy as np
import scipy.stats as ss
from tvb_fit.base.model.probability_distributions import ProbabilityDistributionTypes
from tvb_fit.base.model.probability_distributions.discrete_probability_distribution \
    import DiscreteProbabilityDistribution

from tvb_scripts.utils.log_error_utils import raise_not_implemented_error
from tvb_scripts.utils.data_structures_utils import make_float, make_int


class BernoulliDistribution(DiscreteProbabilityDistribution):

    def __init__(self, **params):
        self.type = ProbabilityDistributionTypes.BERNOULLI
        self.scipy_name = ProbabilityDistributionTypes.BERNOULLI
        self.numpy_name = ""
        self.constraint_string = "0 < p < 1"
        self.p = make_float(params.get("p", 0.5))
        self.__update_params__(p=self.p)

    def pdf_params(self):
        return {"p": self.p}

    def update_params(self, loc=0.0, scale=1.0, use="scipy", **params):
        self.__update_params__(loc, scale, use,
                               p=make_float(params.get("p", self.p)))

    def scale_params(self, loc=0.0, scale=1.0):
        return self.p

    def constraint(self):
        # By default expr >= 0
        p = np.array(self.p).flatten()
        return np.hstack([p - np.finfo(np.float64).eps, 1.0 - p + np.finfo(np.float64).eps])

    def _scipy(self, loc=0.0, scale=1.0):
        return ss.bernoulli(p=self.p, loc=loc)

    def _numpy(self, loc=0.0, scale=1.0, size=(1,)):
        raise_not_implemented_error("No implementation of bernoulli distribution in numpy.random module!")

    def calc_mean_manual(self, loc=0.0, scale=1.0):
        return self.p + loc

    def calc_median_manual(self, loc=0.0, scale=1.0):
        median = 0.5 * np.ones(np.array(self.p * np.ones((1,))).shape)
        median[np.where(self.p < 0.5)[0]] = 0.0
        median[np.where(self.p > 0.5)[0]] = 1.0
        return np.reshape(median, self.p_shape) + loc

    def calc_mode_manual(self, loc=0.0, scale=1.0):
        loc = make_int(np.round(loc))
        p = np.array(self.p)
        shape = p.shape
        mode = np.ones(np.array(p * np.ones((1,))).shape, dtype="i") + loc
        mode[np.where(p < 0.5)[0]] = 1 + loc
        p05 = p == 0.5
        if np.any(p05):
            self.logger.warning("The mode of bernoulli distribution for p=0.5 consists of two values (0.0 + loc and 1.0 + loc)!")
            mode = mode.astype('O')
            mode[np.where(p05)[0]] = [[0.0 + loc, 1.0 + loc]]
        return mode

    def calc_var_manual(self, loc=0.0, scale=1.0):
        return self.p * (1 - self.p) + loc

    def calc_std_manual(self, loc=0.0, scale=1.0):
        return np.sqrt(self.calc_var_manual())

    def calc_skew_manual(self, loc=0.0, scale=1.0):
        return (1.0 - 2.0 * self.p) / self.calc_std_manual()

    def calc_kurt_manual(self, loc=0.0, scale=1.0):
        var = self.calc_var_manual()
        return (1.0 - 6.0 * var) / var

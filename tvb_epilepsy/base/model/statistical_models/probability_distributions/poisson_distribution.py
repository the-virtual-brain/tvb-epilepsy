import numpy as np
import numpy.random as nr
import scipy.stats as ss
from tvb_epilepsy.base.utils.data_structures_utils import make_float, make_int, isequal_string
from tvb_epilepsy.base.model.statistical_models.probability_distributions.discrete_probability_distribution \
    import DiscreteProbabilityDistribution


class PoissonDistribution(DiscreteProbabilityDistribution):

    def __init__(self, **params):
        self.type = "poisson"
        self.scipy_name = "poisson"
        self.numpy_name = "poisson"
        self.lamda = make_float(params.get("lamda", params.get("lam", params.get("mu", 0.5))))
        self.constraint_string = "0 < lamda < 1"
        self.__update_params__(lamda=self.lamda)
        self.mu = self.lamda

    def pdf_params(self, parametrization="lamda"):
        if isequal_string(parametrization, "scipy"):
            return {"mu": self.mu}
        elif isequal_string(parametrization, "numpy"):
            return {"lam": self.mu}
        else:
            return {"lamda": self.lamda}

    def scale_params(self, loc=0.0, scale=1.0):
        return self.lamda

    def update_params(self, loc=0.0, scale=1.0, use="scipy", **params):
        self.__update_params__(loc, scale, use,
                               lamda=make_float(params.get("lamda", params.get("lam", params.get("mu", self.lamda)))))
        self.mu = self.lamda

    def constraint(self):
        # By default expr >= 0
        lamda = np.array(self.lamda).flatten()
        return np.hstack([lamda - np.finfo(np.float64).eps, 1.0 - lamda + np.finfo(np.float64).eps])

    def scipy(self, loc=0.0, scale=1.0):
        return getattr(ss, self.scipy_name)(self.lamda, loc=loc, scale=scale)

    def numpy(self, loc=0.0, scale=1.0, size=(1,)):
        return lambda: nr.poisson(self.lamda, size=size) + loc

    def calc_mean_manual(self, loc=0.0, scale=1.0):
        return self.lamda + loc

    def calc_median_manual(self, loc=0.0, scale=1.0):
        self.logger.warning("Approximate calculation for median of poisson distribution!")
        return np.int(np.round(self.lamda + 1.0 / 3 - 0.02 / self.lamda + loc))

    def calc_mode_manual(self, loc=0.0, scale=1.0):
        return [make_int(np.round(self.lamda + loc)) - 1, make_int(np.round(self.lamda + loc))]

    def calc_var_manual(self):
        return self.lamda

    def calc_std_manual(self):
        return np.sqrt(self.calc_var_manual())

    def calc_skew_manual(self):
        return 1.0 / self.calc_std_manual()

    def calc_kurt_manual(self, use="scipy"):
        return 1.0 / self.lamda

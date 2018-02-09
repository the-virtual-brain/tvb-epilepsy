import numpy as np
import numpy.random as nr
import scipy.stats as ss
from tvb_epilepsy.base.model.statistical_models.probability_distributions import ProbabilityDistributionTypes
from tvb_epilepsy.base.utils.data_structures_utils import make_int
from tvb_epilepsy.base.model.statistical_models.probability_distributions.continuous_probability_distribution \
    import ContinuousProbabilityDistribution


class ChisquareDistribution(ContinuousProbabilityDistribution):

    def __init__(self, **params):
        self.type = ProbabilityDistributionTypes.CHISQUARE
        self.scipy_name = "chi"
        self.numpy_name = ProbabilityDistributionTypes.CHISQUARE
        self.constraint_string = "int(df) > 0"
        self.df = make_int(params.get("df", 1))
        self.__update_params__(df=self.df)

    def __str__(self):
        this_str = super(ChisquareDistribution, self).__str__()
        this_str = this_str[0:-1]
        this_str += "\n" + "13. degrees of freedom" + " = " + str(self.df) + "}"
        return this_str

    def pdf_params(self, parametrization="df"):
        return {"df": self.df}

    def scale_params(self, loc=0.0, scale=1.0):
        return self.df

    def update_params(self, loc=0.0, scale=1.0, use="scipy", **params):
        self.__update_params__(loc, scale, use,
                               df=make_int(params.get("df", self.df)))

    def constraint(self):
        # By default expr >= 0
        return np.array(self.df).flatten() - np.finfo(np.float64).eps

    def scipy(self, loc=0.0, scale=1.0):
        return ss.chi(df=self.df, loc=loc, scale=scale)

    def numpy(self, loc=0.0, scale=1.0, size=(1,)):
        return lambda: nr.chisquare(df=self.df, size=size) * scale + loc

    def calc_mean_manual(self, loc=0.0, scale=1.0):
        return self.df + make_int(np.around(loc))

    def calc_median_manual(self, loc=0.0, scale=1.0):
        self.logger.warning("Approximate calculation for median of chisquare distribution!")
        return self.df * (1 - 2.0 / (9 * self.df)) ** 3 + loc

    def calc_mode_manual(self, loc=0.0, scale=1.0):
        shape = np.array(self.df).shape
        dfmax = np.array(self.df * np.ones((1,)), dtype='i')
        dfmax = (dfmax.flatten() - 2).tolist()
        for id in range(len(dfmax)):
            dfmax[id] = np.max([dfmax[id], 0])
        return np.reshape(dfmax, shape) + make_int(np.around(loc))

    def calc_var_manual(self, loc=0.0, scale=1.0):
        return 2 * self.df * scale ** 2

    def calc_std_manual(self, loc=0.0, scale=1.0):
        return np.sqrt(self.calc_var_manual(loc, scale))

    def calc_skew_manual(self, loc=0.0, scale=1.0):
        return np.sqrt(8.0 / self.df)

    def calc_kurt_manual(self, loc=0.0, scale=1.0):
        return 12.0 / self.df

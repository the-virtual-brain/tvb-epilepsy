
from abc import ABCMeta, abstractmethod

import numpy as np

from tvb_epilepsy.base.model.statistical_models.probability_distributions.probability_distribution \
                                                                                          import ProbabilityDistribution


class ContinuousProbabilityDistribution(ProbabilityDistribution):

    __metaclass__ = ABCMeta

    def scipy_pdf(self, x=None, q=[0.01, 0.99], loc=0.0, scale=1.0):
        if x is None:
            x = np.linspace(self.scipy(loc, scale).ppf(q[0]), self.scipy(loc, scale).ppf(q[0]), 101)
        return self.scipy(loc, scale).pdf(x), x

    @abstractmethod
    def constraint(self):
        pass

    @abstractmethod
    def scipy(self, loc=0.0, scale=1.0):
        pass

    @abstractmethod
    def calc_mu(self, use="scipy"):
        pass

    @abstractmethod
    def calc_median(self, use="scipy"):
        pass

    @abstractmethod
    def calc_mode(self):
        pass

    @abstractmethod
    def calc_var(self, use="scipy"):
        pass

    @abstractmethod
    def calc_std(self, use="scipy"):
        pass

    @abstractmethod
    def calc_skew(self, use="scipy"):
        pass

    @abstractmethod
    def calc_exkurt(self, use="scipy"):
        pass
import numpy as np

from tvb_epilepsy.base.model.statistical_models.probability_distributions.probability_distribution \
                                                                                          import ProbabilityDistribution


class DiscreteProbabilityDistribution(ProbabilityDistribution):

    def scipy_pdf(self, x=None, q=[0.01, 0.99], loc=0.0, scale=1.0):
        if x is None:
            x = np.linspace(self.scipy(loc, scale).ppf(q[0]), self.scipy(loc, scale).ppf(q[0]), 101)
        return self.scipy(loc, scale).pmf(x), x
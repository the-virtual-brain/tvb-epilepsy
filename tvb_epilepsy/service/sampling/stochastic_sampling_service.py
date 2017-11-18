
import numpy as np
import numpy.random as nr
import scipy.stats as ss

from tvb_epilepsy.base.constants.module_constants import MAX_SINGLE_VALUE
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning
from tvb_epilepsy.base.utils.data_structures_utils import dict_str, formal_repr, isequal_string
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.probability_distributions.probability_distribution \
                                                                                          import ProbabilityDistribution
from tvb_epilepsy.service.sampling.sampling_service import SamplingService


logger = initialize_logger(__name__)


class StochasticSamplingService(SamplingService):

    def __init__(self, n_samples=10, sampling_module="scipy", random_seed=None):
        super(StochasticSamplingService, self).__init__(n_samples)
        self.random_seed = random_seed
        self.sampling_module = sampling_module.lower()

    def __repr__(self):

        d = {"01. Sampling module": self.sampling_module,
             "02. Sampler": self.sampler,
             "03. Number of samples": self.n_samples,
             "04. Samples' p_shape": self.shape,
             "05. Random seed": self.random_seed,
             }
        return formal_repr(self, d) + "\n06. Resulting statistics: " + dict_str(self.stats)

    def __str__(self):
        return self.__repr__()

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model({"sampling_module": self.sampling_module, "sampler": self.sampler,
                                        "n_samples": self.n_samples, "p_shape": self.shape,
                                        "random_seed": self.random_seed, "stats": self.stats})
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        return h5_model

    def _truncated_distribution_sampling(self, trunc_limits, size):
        # Following: https://stackoverflow.com/questions/25141250/
        # how-to-truncate-a-numpy-scipy-exponential-distribution-in-an-efficient-way
        # TODO: to have distributions parameters valid for the truncated distributions instead for the original one
        # pystan might be needed for that...
        rnd_cdf = nr.uniform(self.sampler.cdf(x=trunc_limits.get("low", -np.inf)),
                             self.sampler.cdf(x=trunc_limits.get("high", np.inf)),
                             size=size)
        return self.sampler.ppf(q=rnd_cdf)

    def sample(self, parameter=(), **kwargs):
        nr.seed(self.random_seed)
        if isinstance(parameter, Parameter):
            parameter_shape = parameter.p_shape
            low = parameter.low
            high = parameter.high
            prob_distr = parameter
        else:
            parameter_shape = kwargs.pop("shape", (1,))
            low = kwargs.pop("low", -MAX_SINGLE_VALUE)
            high = kwargs.pop("high", MAX_SINGLE_VALUE)
            prob_distr = kwargs.pop("probability_distribution", "uniform")
        low, high = self.check_for_infinite_bounds(low, high)
        low, high, n_outputs, parameter_shape = self.check_size(low, high, parameter_shape)
        self.adjust_shape(parameter_shape)
        out_shape = tuple([self.n_samples] + list(self.shape)[:-1])
        if np.any(low > -MAX_SINGLE_VALUE) or np.any(high < MAX_SINGLE_VALUE):
            if not(isequal_string(self.sampling_module, "scipy")):
                warning("Switching to scipy for truncated distributions' sampling!")
            self.sampling_module = "scipy"
            if isinstance(prob_distr, basestring):
                self.sampler = getattr(ss, prob_distr)(*parameter, **kwargs)
            elif isinstance(prob_distr, ProbabilityDistribution):
                self.sampler = prob_distr.scipy()
            samples = self._truncated_distribution_sampling({"low": low, "high": high}, out_shape)
        elif self.sampling_module.find("scipy") >= 0:
            if isinstance(prob_distr, basestring):
                self.sampler = getattr(ss, prob_distr)(*parameter, **kwargs)
            elif isinstance(prob_distr, ProbabilityDistribution):
                self.sampler = prob_distr.scipy(**kwargs)
            samples = self.sampler.rvs(size=out_shape)
        elif self.sampling_module.find("numpy") >= 0:
            if isinstance(prob_distr, basestring):
                self.sampler = lambda size: getattr(nr, prob_distr)(*parameter, size=size, **kwargs)
            elif isinstance(prob_distr, ProbabilityDistribution):
                self.sampler = lambda size: prob_distr.numpy(size=size)
            samples = self.sampler(out_shape)
        return samples.T

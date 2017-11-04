import sys

import importlib
from collections import OrderedDict
from copy import deepcopy

import numpy as np
import numpy.random as nr
import scipy as scp
import scipy.stats as ss
from SALib.sample import saltelli, fast_sampler, morris, ff

from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning, raise_value_error, \
                                                                                            raise_not_implemented_error
from tvb_epilepsy.base.utils.data_structures_utils import dict_str, formal_repr, isequal_string, shape_to_size
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.probability_distributions.probability_distribution \
                                                                                          import ProbabilityDistribution


logger = initialize_logger(__name__)


class SamplingService(object):

    def __init__(self, n_samples=10):
        self.sampler = None
        self.sampling_module = ""
        self.n_samples = n_samples
        self.shape = (1, n_samples)
        self.stats = {}

    def __repr__(self):
        d = {"01. Sampling module": self.sampling_module,
             "02. Sampler": self.sampler,
             "03. Number of samples": self.n_samples,
             "04. Samples' shape": self.shape,
             }
        return formal_repr(self, d) + "\n05. Resulting statistics: " + dict_str(self.stats)

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model({"sampling_module": self.sampling_module, "sampler": self.sampler,
                                        "n_samples": self.n_samples, "shape": self.shape, "stats": self.stats})
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

    def adjust_shape(self, parameter_shape):
        shape = []
        for p in parameter_shape:
            shape.append(p)
        shape.append(self.n_samples)
        self.shape = shape

    def compute_stats(self, samples):
        return OrderedDict([("mu", samples.mean(axis=-1)), ("m", scp.median(samples, axis=-1)),
                            ("std", samples.std(axis=-1)), ("var", samples.var(axis=-1)),
                            ("k", ss.kurtosis(samples, axis=-1)), ("skew", ss.skew(samples, axis=-1)),
                            ("min", samples.min(axis=-1)), ("max", samples.max(axis=-1)),
                            ("1%", np.percentile(samples, 1, axis=-1)), ("5%", np.percentile(samples, 5, axis=-1)),
                            ("10%", np.percentile(samples, 10, axis=-1)), ("p25", np.percentile(samples, 25, axis=-1)),
                            ("p50", np.percentile(samples, 50, axis=-1)), ("p75", np.percentile(samples, 75, axis=-1)),
                            ("p90", np.percentile(samples, 90, axis=-1)), ("p95", np.percentile(samples, 95, axis=-1)),
                            ("p99", np.percentile(samples, 99, axis=-1))])

    def generate_samples(self, parameter, stats=False, **kwargs):
        samples = self.sample(parameter, **kwargs)
        self.stats = self.compute_stats(samples)
        if stats:
            return samples, self.stats
        else:
            return samples


class DeterministicSamplingService(SamplingService):

    def __init__(self, n_samples=10, grid_mode=True):
        super(DeterministicSamplingService, self).__init__(n_samples)
        self.sampling_module = "numpy.linspace"
        self.sampler = np.linspace
        self.grid_mode = grid_mode
        self.shape = (1, self.n_samples)

    def sample(self, parameter, **kwargs):
        if isinstance(parameter, Parameter):
            parameter_shape = parameter.shape
            low = parameter.low
            high = parameter.high
        else:
            parameter_shape = parameter["shape"]
            low = parameter["low"]
            high = parameter["high"]
        self.adjust_shape(parameter_shape)
        i1 = np.ones(parameter_shape)
        low = np.array(low) * i1
        high = np.array(high) * i1
        samples = []
        for (lo, hi) in zip(low.flatten(), high.flatten()):
            samples.append(self.sampler(lo, hi, self.n_samples))
        if self.grid_mode:
            samples_grids = np.meshgrid(*(samples), sparse=False, indexing="ij")
            samples = []
            for sb in samples_grids:
                samples.append(sb.flatten())
                samples = np.array(samples)
                self.shape = samples.shape
        else:
            samples = np.array(samples)
            transpose_shape = tuple([self.n_samples] + list(self.shape)[0:-1])
            samples = np.reshape(samples, transpose_shape).T
        return samples


# TODO: Add pystan as a stochastic sampling module, when/if needed.

class StochasticSamplingService(SamplingService):

    def __init__(self, n_samples=10, sampling_module="scipy", sampler=None, random_seed=None):
        super(StochasticSamplingService, self).__init__(n_samples)
        self.random_seed = random_seed
        self.sampling_module = sampling_module.lower()
        self.sampler = sampler

    def __repr__(self):

        d = {"01. Sampling module": self.sampling_module,
             "02. Sampler": self.sampler,
             "03. Number of samples": self.n_samples,
             "04. Samples' shape": self.shape,
             "05. Random seed": self.random_seed,
             }
        return formal_repr(self, d) + "\n06. Resulting statistics: " + dict_str(self.stats)

    def __str__(self):
        return self.__repr__()

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model({"sampling_module": self.sampling_module, "sampler": self.sampler,
                                        "n_samples": self.n_samples, "shape": self.shape,
                                        "random_seed": self.random_seed, "stats": self.stats})
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        return h5_model

    def _salib_sample(self, parameter, **kwargs):
        if isinstance(parameter, Parameter):
            parameter_shape = parameter.shape
            low = np.array(parameter.low)
            high = np.array(parameter.high)
            d = (low == -np.inf)
            if np.any(id):
                warning("SALib sampling is not possible with infinite bounds! Setting lowest system value for low!")
                low[id] = sys.floatinfo["MIN"]
            id = (high == np.inf)
            if np.any(id):
                warning("SALib sampling is not possible with infinite bounds! Setting highest system value for high!")
                high[id] = sys.floatinfo["MAX"]
            i1 = np.ones(parameter_shape)
            low = (np.array(low) * i1).tolist()
            high = (np.array(high) * i1).tolist()
            bounds = [list(b) for b in zip(low, high)]
            n_outputs = len(bounds)
        else:
            bounds = parameter["bounds"]
            parameter_shape = parameter.get("shape", len(bounds))
            n_outputs = shape_to_size(parameter_shape)
            if len(bounds) < n_outputs:
                if len(bounds) == 1:
                    bounds = bounds * n_outputs
                else:
                    raise_value_error("Parameters shape (" + str(parameter_shape) +
                                      ") and bounds length (" + str(len(bounds)) + ") do not match!")
        self.adjust_shape(parameter_shape)
        self.sampler = importlib.import_module("SALib.sample." + self.sampler).sample
        size = self.n_samples
        problem = {'num_vars': n_outputs, 'bounds': bounds}
        if self.sampler is ff.sample:
            samples = (self.sampler(problem)).T
        else:
            other_params = {}
            if self.sampler is saltelli.sample:
                size = int(np.round(1.0 * size / (2 * n_outputs + 2)))
            elif self.sampler is fast_sampler.sample:
                other_params = {"M": kwargs.get("M", 4)}
            elif self.sampler is morris.sample:
                # I don't understand this method and its inputs. I don't think we will ever use it.
                raise_not_implemented_error()
            samples = self.sampler(problem, size, **other_params)
        # Adjust samples number:
        self.n_samples = samples[0]
        self.shape = list(self.shape)
        self.shape[-1] = self.n_samples
        self.shape = tuple(self.shape)
        transpose_shape = tuple([self.n_samples] + list(self.shape)[0:-1])
        return np.reshape(samples.T, transpose_shape).T

    def _truncated_distribution_sampling(self, trunc_limits, size):
        # Following: https://stackoverflow.com/questions/25141250/
        # how-to-truncate-a-numpy-scipy-exponential-distribution-in-an-efficient-way
        # TODO: to have distributions parameters valid for the truncated distributions instead for the original one
        # pystan might be needed for that...
        rnd_cdf = nr.uniform(self.sampler.cdf(x=trunc_limits.get("low", -np.inf)),
                             self.sampler.cdf(x=trunc_limits.get("high", np.inf)),
                             size=size)
        return self.sampler.ppf(q=rnd_cdf)


    def sample(self, parameter, **kwargs):
        nr.seed(self.random_seed)
        if self.sampling_module.find("SALib") >= 0:
            self._salib_sample(parameter, **kwargs)
        else:
            if isinstance(parameter, Parameter):
                parameter_shape = parameter.shape
                low = parameter.low
                high = parameter.high
                prob_distr = parameter.probability_distribution
            else:
                parameter_shape = parameter["shape"]
                low = parameter.get("low", -np.inf)
                high = parameter.get("high", -np.inf)
                prob_distr = parameter["probability_distribution", "uniform"]
            self.adjust_shape(self, parameter_shape)
            i1 = np.ones(parameter_shape)
            low = np.array(low) * i1
            high = np.array(high) * i1
            out_shape = tuple([self.n_samples] + list(self.shape)[:-1])
            if np.any(low > -np.inf) or np.any(high < np.inf):
                if not(isequal_string(self.sampling_module, "scipy")):
                    warning("Switching to scipy for truncated distributions' sampling!")
                    self.sampling_module = "scipy"
                    if isinstance(prob_distr, basestring):
                        self.sampler = getattr(ss, prob_distr)(**kwargs)
                    elif isinstance(prob_distr, ProbabilityDistribution):
                        self.sampler = prob_distr.scipy
                    samples = self._truncated_distribution_sampling({"low": low, "high": high}, out_shape)
            elif self.sampling_module.find("scipy") >= 0:
                if isinstance(prob_distr, basestring):
                    self.sampler = getattr(ss, prob_distr)(**kwargs)
                elif isinstance(prob_distr, ProbabilityDistribution):
                    self.sampler = prob_distr.scipy
                samples = self.sampler.rvs(size=out_shape)
            elif self.sampling_module.find("numpy") >= 0:
                if isinstance(prob_distr, basestring):
                    pdf_name = prob_distr
                    pdf_params = kwargs
                elif isinstance(prob_distr, ProbabilityDistribution):
                    pdf_name = prob_distr.name
                    pdf_params = prob_distr.params
                self.sampler = getattr(nr, pdf_name.name)
                samples = self.sampler(*pdf_params.values(), size=out_shape)
            return samples.T


import importlib
from collections import OrderedDict

import numpy as np
import numpy.random as nr
import scipy as scp
import scipy.stats as ss
from SALib.sample import saltelli, fast_sampler, morris, ff

from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning, raise_value_error, \
    raise_not_implemented_error
from tvb_epilepsy.base.utils.data_structures_utils import dict_str, formal_repr, dicts_of_lists, \
    dicts_of_lists_to_lists_of_dicts, isequal_string
from tvb_epilepsy.base.h5_model import convert_to_h5_model


logger = initialize_logger(__name__)


class SamplingService(object):

    def __init__(self, n_samples=10, n_outputs=1):
        self.sampler = None
        self.sampling_module = ""
        self.n_samples = n_samples
        self.n_outputs = n_outputs
        self.shape = (n_outputs, n_samples)
        self.stats = {}

    def __repr__(self):
        d = {"01. Sampling module": self.sampling_module,
             "02. Sampler": self.sampler,
             "03. Number of samples": self.n_samples,
             "04. Number of output parameters": self.n_outputs,
             "05. Samples' shape": self.shape,
             }
        return formal_repr(self, d) + "\n06. Distribution parameters: " + dict_str(self.params) + \
                                      "\n07 Resulting statistics: " + dict_str(self.stats)

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model({"sampling_module": self.sampling_module, "sampler": self.sampler,
                                        "n_samples": self.n_samples, "n_outputs": self.n_outputs, "shape": self.shape,
                                        "stats": self.stats})
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

    def adjust_to_parameters_shape(self, parameter_shape):
        shape = [self.n_outputs]
        for s in parameter_shape:
            shape.append(s)
        shape += [self.samples]
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

    def generate_samples(self, parameter, stats=False, force_parameters_shape=False, **kwargs):
        if force_parameters_shape:
            self.adjust_to_parameters_shape(parameter.shape)
        samples = self.sample(parameter, **kwargs)
        self.stats = self.compute_stats(samples)
        if stats:
            return samples, self.stats
        else:
            return samples


class DeterministicSamplingService(SamplingService):

    def __init__(self, n_samples=10, n_outputs=1, grid_mode=True):
        super(DeterministicSamplingService, self).__init__(n_samples, n_outputs)
        self.sampling_module = "numpy.linspace"
        self.sampler = np.linspace
        self.grid_mode = grid_mode
        if self.grid_mode:
            self.shape = (self.n_outputs, np.power(self.n_samples, self.n_outputs))

    def sample(self, parameter, **kwargs):
        i1 = np.ones((self.n_outputs,))
        low = (np.array(parameter.low) * i1).tolist()
        high = (np.array(parameter.high) * i1).tolist()
        samples = []
        if len(self.shape) > 2:
            shape = tuple(list(self.shape)[1:-1] + [1])
        else:
            shape = 1
        for io in range(self.n_outputs):
            samples.append(np.tile(self.sampler(low[io], high[io], self.n_samples), shape))
        if self.grid_mode:
            samples_grids = np.meshgrid(*(samples.tolist()), sparse=False, indexing="ij")
            samples = []
            for sb in samples_grids:
                samples.append(sb.flatten())
        return np.array(samples)


# TODO: Add pystan as a stochastic sampling module, when/if needed.

class StochasticSamplingService(SamplingService):

    def __init__(self, n_samples=10, n_outputs=1, sampling_module="scipy", sampler=None, random_seed=None):
        super(StochasticSamplingService, self).__init__(n_samples, n_outputs)
        self.random_seed = random_seed
        self.sampling_module = sampling_module.lower()
        if isequal_string(sampling_module, "salib"):
            self.sampler = sampler
            self.sampling_module = "SALib.sample" + self.sampler + ".sampler"

    def __repr__(self):

        d = {"01. Sampling module": self.sampling_module,
             "02. Sampler": self.sampler,
             "03. Number of samples": self.n_samples,
             "04. Number of output parameters": self.n_outputs,
             "05. Samples' shape": self.shape,
             "06. Random seed": self.random_seed,
             }
        return formal_repr(self, d) + "\n07. Resulting statistics: " + dict_str(self.stats)

    def __str__(self):
        return self.__repr__()

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model({"sampling_module": self.sampling_module, "sampler": self.sampler,
                                        "n_samples": self.n_samples, "n_outputs": self.n_outputs, "shape": self.shape,
                                        "random_seed": self.random_seed, "stats": self.stats})
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        return h5_model

    def _salib_sample(self, bounds, **kwargs):
        sampler = importlib.import_module("SALib.sample." + self.sampler).sample
        size = self.n_samples
        problem = {'num_vars': self.n_outputs, 'bounds': bounds}
        if sampler is ff.sample:
            samples = sampler(problem)
        else:
            other_params = {}
            if sampler is saltelli.sample:
                size = int(np.round(1.0*size / (2*self.n_outputs + 2)))
            elif sampler is fast_sampler.sample:
                other_params = {"M": kwargs.get("M", 4)}
            elif sampler is morris.sample:
                # I don't understand this method and its inputs. I don't think we will ever use it.
                raise_not_implemented_error()
            samples = sampler(problem, size, **other_params).T
            # Adjust samples number:
            self.n_samples = samples[1]
            if len(self.shape) > 2:
                shape = tuple(list(self.shape)[1:-1] + [1])
                out_samples = []
                for s in samples:
                    out_samples.append(np.tile(s, shape))
                samples = np.array(out_samples)
        self.shape = samples.shape
        return samples

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
        i1 = np.ones((self.n_outputs))
        low = (np.array(parameter.low) * i1).tolist()
        high = (np.array(parameter.high) * i1).tolist()
        if self.sampling_module.find("SALib") >= 0:
            if np.any(low == -np.inf) or np.any(high == np.inf):
                raise_value_error("SALib sampling is not possible with infinite bounds!")
            return self._salib_sample(bounds=zip(low, high), **kwargs)
        else:
            if np.any(low > -np.inf) or np.any(high < np.inf):
                if not(isequal_string(self.sampling_module, "scipy")):
                    warning("Switching to scipy for truncated distributions' sampling!")
                self.sampler = parameter.prob_distr.scipy
                trunc_limits = dicts_of_lists_to_lists_of_dicts({"low": low, "high": high})
                samples = []
                output_shape = tuple(list(self.shape)[1:])
                for io in range(self.n_outputs):
                    samples.append(
                        self._truncated_distribution_sampling(self.sampler, trunc_limits[io], output_shape))
                return np.array(samples)
            elif self.sampling_module.find("scipy") >= 0:
                self.sampler = parameter.prob_distr.scipy
                return (self.sampler).rvs(size=self.shape)
            elif self.sampling_module.find("numpy") >= 0:
                self.sampler = getattr(nr, parameter.prob_distr.name)
                return self.sampler(*parameter.prob_distr.params.values(), size=self.shape)


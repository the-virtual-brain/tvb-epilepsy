
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
    dicts_of_lists_to_lists_of_dicts
from tvb_epilepsy.base.h5_model import convert_to_h5_model


logger = initialize_logger(__name__)


class SamplingService(object):

    def __init__(self, n_samples=10, n_outputs=1):

        self.sampler = None
        self.sampling_module = ""
        self.shape = (n_outputs, n_samples)
        self.n_samples = n_samples
        self.n_outputs = n_outputs
        self.stats = {}
        self.params = {}

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
                                   "params": self.params, "stats": self.stats})
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

    def _list_params(self):
        self.params = dicts_of_lists(self.params, self.n_outputs)

    def compute_stats(self, samples):
        return OrderedDict([("mu", samples.mean(axis=1)), ("m", scp.median(samples, axis=1)),
                            ("std", samples.std(axis=1)), ("var", samples.var(axis=1)),
                            ("k", ss.kurtosis(samples, axis=1)), ("skew", ss.skew(samples, axis=1)),
                            ("min", samples.min(axis=1)), ("max", samples.max(axis=1)),
                            ("1%", np.percentile(samples, 1, axis=1)), ("5%", np.percentile(samples, 5, axis=1)),
                            ("10%", np.percentile(samples, 10, axis=1)), ("p25", np.percentile(samples, 25, axis=1)),
                            ("p50", np.percentile(samples, 50, axis=1)), ("p75", np.percentile(samples, 75, axis=1)),
                            ("p90", np.percentile(samples, 90, axis=1)), ("p95", np.percentile(samples, 95, axis=1)),
                            ("p99", np.percentile(samples, 99, axis=1))])

    def generate_samples(self, stats=False, **kwargs):
        samples = self.sample(**kwargs)
        self.stats = self.compute_stats(samples)
        if stats:
            return samples, self.stats
        else:
            return samples


class DeterministicSamplingService(SamplingService):

    def __init__(self, n_samples=10, n_outputs=1, low=0.0, high=1.0, grid_mode=True):

        super(DeterministicSamplingService, self).__init__(n_samples, n_outputs)

        self.sampling_module = "numpy.linspace"
        self.sampler = np.linspace
        self.grid_mode = grid_mode
        if self.grid_mode:
            self.shape = (self.n_outputs, np.power(self.n_samples, self.n_outputs))

        if np.any(high <= low):
            raise_value_error("\nHigh limit of linear space " + str(high) +
                             " is not greater than the lower one " + str(low) + "!")
        else:
            self.params = {"low": low, "high": high}
            self._list_params()

    def sample(self, **kwargs):

        samples = []
        for io in range(self.n_outputs):
            samples.append(self.sampler(self.params["low"][io], self.params["high"][io], self.n_samples))

        if self.grid_mode:
            samples_grids = np.meshgrid(*samples, sparse=False, indexing="ij")
            samples = []
            for sb in samples_grids:
                samples.append(sb.flatten())

        return np.array(samples)


# TODO: Add pystan as a stochastic sampling module, when/if needed.

class StochasticSamplingService(SamplingService):

    def __init__(self, n_samples=10, n_outputs=1, sampler="uniform", trunc_limits={},
                 sampling_module="numpy", random_seed=None, **kwargs):

        super(StochasticSamplingService, self).__init__(n_samples, n_outputs)

        self.random_seed = random_seed
        self.params = kwargs
        self._list_params()
        self.trunc_limits = trunc_limits
        sampling_module = sampling_module.lower()

        self.sampler = sampler

        if len(self.trunc_limits) > 0:

            self.trunc_limits = dicts_of_lists(self.trunc_limits, self.n_outputs)

            # We use inverse transform sampling for truncated distributions...

            if sampling_module is not "scipy":
                warning("\nSelecting scipy module for truncated distributions")

            self.sampling_module = "scipy.stats." + sampler + " inverse transform sampling"

        elif sampling_module == "scipy":
            self.sampling_module = "scipy.stats." + self.sampler + ".rvs"

        elif sampling_module == "numpy":
            self.sampling_module = "numpy.random." + self.sampler

        elif sampling_module == "salib":
            self.sampling_module = "SALib.sample." + self.sampler + ".sample"

        else:
            raise_value_error("Sampler module " + str(sampling_module) + " is not recognized!")

    def __repr__(self):

        d = {"01. Sampling module": self.sampling_module,
             "02. Sampler": self.sampler,
             "03. Number of samples": self.n_samples,
             "04. Number of output parameters": self.n_outputs,
             "05. Samples' shape": self.shape,
             "06. Random seed": self.random_seed,
             }
        return formal_repr(self, d) + \
        "\n07. Distribution parameters: " + dict_str(self.params) + \
        "\n08. Truncation limits: " + str([dict_str(d) for d in dicts_of_lists_to_lists_of_dicts(self.trunc_limits)]) + \
        "\n08. Resulting statistics: " + dict_str(self.stats)

    def __str__(self):
        return self.__repr__()

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model({"sampling_module": self.sampling_module, "sampler": self.sampler,
                                   "n_samples": self.n_samples, "n_outputs": self.n_outputs, "shape": self.shape,
                                   "random_seed": self.random_seed,
                                   "trunc_limits": np.array([(d.get("low", -np.inf), d.get("high", np.inf))
                                                        for d in dicts_of_lists_to_lists_of_dicts(self.trunc_limits)]),
                                        "params": self.params, "stats": self.stats})
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        return h5_model

    def _numpy_sample(self, distribution, size, **params):
        return getattr(nr, distribution)(size=size, **params)

    def _scipy_sample(self, distribution, size, **params):
        return getattr(ss, distribution)(**params).rvs(size)

    def _truncated_distribution_sampling(self, distribution, trunc_limits, size, **kwargs):
        # Following: https://stackoverflow.com/questions/25141250/
        # how-to-truncate-a-numpy-scipy-exponential-distribution-in-an-efficient-way
        # TODO: to have distributions parameters valid for the truncated distributions instead for the original one
        # pystan might be needed for that...
        rnd_cdf = nr.uniform(getattr(ss, distribution)(**kwargs).cdf(x=trunc_limits.get("low", -np.inf)),
                             getattr(ss, distribution)(**kwargs).cdf(x=trunc_limits.get("high", np.inf)),
                             size=size)
        return getattr(ss, distribution)(**kwargs).ppf(q=rnd_cdf)

    def _salib_sample(self, **kwargs):

        sampler = importlib.import_module("SALib.sample." + self.sampler).sample

        size = self.n_samples

        problem = {'num_vars': self.n_outputs, 'bounds': kwargs.get("bounds", [0.0, 1.0] * self.n_outputs)}
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
                raise_not_implemented_error

            samples = sampler(problem, size, **other_params)

        # Adjust samples number:
        self.n_samples = samples.shape[0]
        self.shape = (self.n_outputs, self.n_samples)

        return samples.T

    def sample(self, **kwargs):

        nr.seed(self.random_seed)

        if self.sampling_module.find("SALib") >= 0:
            samples = self._salib_sample(**self.params)

        else:

            params = dicts_of_lists_to_lists_of_dicts(self.params)

            if self.sampling_module.find("inverse transform") >= 0:
                trunc_limits = dicts_of_lists_to_lists_of_dicts(self.trunc_limits)
                samples = []
                if len(params)== 0:
                    for io in range(self.n_outputs):
                        samples.append(self._truncated_distribution_sampling(self.sampler, trunc_limits[io],
                                                                             self.n_samples))
                elif len(params) == self.n_outputs:
                    for io in range(self.n_outputs):
                        samples.append(self._truncated_distribution_sampling(self.sampler, trunc_limits[io],
                                                                             self.n_samples, **(params[io])))
                else:
                    raise_value_error("\nParameters are neither an empty list nor a list of n_parameters = "
                                     + str(self.n_outputs) + " but one of length " + str(len(self.params)) + " !")

            elif self.sampling_module.find("scipy") >= 0:

                samples = []
                if len(params) == 0:
                    for io in range(self.n_outputs):
                        samples.append(self._scipy_sample(self.sampler, self.n_samples))
                elif len(params) == self.n_outputs:
                    for io in range(self.n_outputs):
                        samples.append(self._scipy_sample(self.sampler, self.n_samples, **(params[io])))
                else:
                    raise_value_error("\nParameters are neither an empty list nor a list of length n_parameters = "
                                     + str(self.n_outputs) + " but one of length " + str(len(self.params)) + " !")

            elif self.sampling_module.find("numpy") >= 0:
                samples = []
                if len(params) == 0:
                    for io in range(self.n_outputs):
                        samples.append(self._numpy_sample(self.sampler, self.n_samples))
                elif len(params) == self.n_outputs:
                    for io in range(self.n_outputs):
                        samples.append(self._numpy_sample(self.sampler, self.n_samples, **(params[io])))
                else:
                    raise_value_error("\nParameters are neither an empty list nor a list of length n_parameters = "
                                     + str(self.n_outputs) + " but one of length " + str(len(self.params)) + " !")

        return np.reshape(samples, self.shape)

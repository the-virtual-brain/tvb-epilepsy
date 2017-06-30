import warnings

import numpy as np
import numpy.random as nr
import scipy.stats as ss
import scipy as scp
import SALib.sample
from SALib.sample import sobol_sequence, saltelli

import importlib
from collections import OrderedDict

from tvb_epilepsy.base.utils import shape_to_size, formal_repr, dict_str, dicts_of_lists, \
                                    dicts_of_lists_to_lists_of_dicts, list_of_dicts_to_dicts_of_ndarrays


def mean_std_to_low_high(mu=0.0, std=1.0):

    std = std * np.sqrt(3.0)

    low = mu - std
    high = mu + std

    return low, high


def low_high_to_mean_std(low=0.0, high=1.0):

    mu = (low + high) / 2.0

    std = (high - low) / 2.0 / np.sqrt(3)

    return mu, std

# def loc_scale_to_mean_std(loc=0.0, scale=1.0, distribution=):


class SampleService(object):

    def __init__(self, n_samples=10, n_outputs=1):

        self.sampler = None
        self.sampling_module = ""
        self.shape = (n_outputs, n_samples)
        self.n_samples = n_samples
        self.n_outputs = n_outputs
        self.stats = {}
        self.params = {}

    def compute_stats(self, samples):
        return OrderedDict([("mu", samples.mean(axis=1)), ("m", scp.median(samples, axis=1)),
                            ("mode", ss.mode(samples, axis=1)),
                            ("std", samples.std()), ("var", samples.var(axis=1)),
                            ("k", ss.kurtosis(samples, axis=1)), ("skew", ss.skew(samples, axis=1)),
                            ("min", samples.min(axis=1)), ("max", samples.max(axis=1)),
                            ("1%", np.percentile(samples, 1, axis=1)), ("5%", np.percentile(samples, 5, axis=1)),
                            ("10%", np.percentile(samples, 10, axis=1)), ("p25", np.percentile(samples, 25, axis=1)),
                            ("p50", np.percentile(samples, 50, axis=1)), ("p75", np.percentile(samples, 75, axis=1)),
                            ("p90", np.percentile(samples, 90, axis=1)), ("p95", np.percentile(samples, 95, axis=1)),
                            ("p99", np.percentile(samples, 99, axis=1))])

    def generate_samples(self, stats=False):
        samples = self.sample()
        self.stats = self.compute_stats(samples)
        if stats:
            return samples, self.stats
        else:
            return samples

    def list_params(self):
        self.params = dicts_of_lists(self.params, self.n_outputs)

    def __repr__(self):
        d = OrderedDict([("shape", str(self.shape)), ("sampler", self.sampling_module)])

        return formal_repr(self, d) + \
               "\nparameters: " + dict_str(self.params) + \
               "\nstats: " + dict_str(self.stats)

class DeterministicSampleService(SampleService):

    def __init__(self, n_samples=10, n_outputs=1, low=0.0, high=1.0, grid_mode=True):

        super(DeterministicSampleService, self).__init__(n_samples, n_outputs)

        self.sampling_module = "numpy.linspace"
        self.sampler = np.linspace
        self.grid_mode = grid_mode
        if self.grid_mode:
            self.shape = (self.n_outputs, np.power(self.n_samples, self.n_outputs))

        if np.any(high <= low):
            raise ValueError("\nHigh limit of linear space " + str(high) +
                             " is not greater than the lower one " + str(low) + "!")
        else:
            self.params = {"low": low, "high": high}
            self.list_params()

    def sample(self):

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

class StochasticSampleService(SampleService):

    def __init__(self, n_samples=10, n_outputs=1, sampler="uniform", trunc_limits={},
                 sampling_module="numpy", random_seed=None, **kwargs):

        super(StochasticSampleService, self).__init__(n_samples, n_outputs)

        self.random_seed = random_seed
        self.params = kwargs
        self.list_params()
        self.trunc_limits = trunc_limits

        if sampling_module == "salib":

            self.sampling_module = "SALib.sample." + sampler + ".sample"
            sampler = importlib.import_module("SALib.sample." + sampler)
            self.sampler = lambda **some_kwargs: self.salib_sample(sampler.sample, **some_kwargs)

        else:

            if len(self.trunc_limits) > 0:

                self.trunc_limits = dicts_of_lists(self.trunc_limits, self.n_outputs)

                # We use inverse transform sampling for truncated distributions...

                if sampling_module is not "scipy":
                    warnings.warn("\nSelecting scipy module for truncated distributions")

                self.sampling_module = "scipy.stats." + sampler + " inverse transform sampling"
                self.sampler = lambda trunc_limits, **some_kwargs: \
                   self.truncated_distribution(getattr(ss, sampler), trunc_limits, **some_kwargs)

            else:

                if sampling_module == "numpy":
                    self.sampling_module = "numpy.random." + sampler
                    self.sampler = getattr(nr, sampler)

                elif sampling_module == "scipy":
                    self.sampling_module = "scipy.stats." + sampler + ".rvs"
                    self.sampler = getattr(ss, sampler).rvs

                else:
                    raise ValueError("Sampler module " + str(sampling_module) + " is not recognized!")

    def truncated_distribution(self, distribution, trunc_limits, size=1, **kwargs):
        # Following: https://stackoverflow.com/questions/25141250/
        # how-to-truncate-a-numpy-scipy-exponential-distribution-in-an-efficient-way
        # TODO: to have distributions parameters valid for the truncated distributions instead for the original one
        # pystan might be needed for that...
        rnd_cdf = nr.uniform(distribution.cdf(x=trunc_limits.get("low", distribution.a), **kwargs),
                             distribution.cdf(x=trunc_limits.get("high", distribution.b), **kwargs),
                             size=size)
        return distribution.ppf(q=rnd_cdf, **kwargs)

    def salib_sample(self, sampler, size=1, **kwargs):

        problem = {'num_vars': self.n_outputs, 'bounds': kwargs.get("bounds", [0.0, 1.0] * self.n_outputs)}
        if sampler is saltelli.sample:
            size = int(np.round(1.0*size / (2*self.outputs + 2)))

        samples = sampler(problem, size)
        #Adjust samples number:
        self.n_samples = samples.shape[0]
        self.shape = (self.n_outputs, self.n_samples)

        return samples.T

    def sample(self):

        nr.seed(self.random_seed)

        if self.sampling_module.find("SALib") >= 0:
            samples = self.sampler(size=self.n_samples, **self.params)

        else:

            params = dicts_of_lists_to_lists_of_dicts(self.params)

            if self.sampling_module.find("inverse transform") >= 0:
                trunc_limits = dicts_of_lists_to_lists_of_dicts(self.trunc_limits)
                samples = []
                for io in range(self.n_outputs):
                    samples.append(self.sampler(trunc_limits[io], size=self.n_samples, **(params[io])))

            else:
                samples = []
                for io in range(self.n_outputs):
                    samples.append(self.sampler(size=self.n_samples, **(params[io])))

        return np.reshape(samples, self.shape)

    def __repr__(self):
        d = OrderedDict([("sampler", str(self.sampling_module)), ("n_samples", str(self.shape)),
                         ("n_outputs", str(self.shape)), ("random_seed", str(self.random_seed))])

        return formal_repr(self, d) + "\n" + "\ntruncation limits: " + dict_str(self.trunc_limits) + "\n"


if __name__ == "__main__":

    print("\nDeterministic linspace sampling:")

    sampler = DeterministicSampleService(n_samples=10, n_outputs=2, low=1.0, high=2.0, grid_mode=True)

    samples, stats = sampler.generate_samples(stats=True)

    # for key, value in stats.iteritems():
    #
    #     print("\n" + key + ": " + str(value))

    print(sampler.__repr__())

    print("\nStochastic uniform sampling:")

    sampler = StochasticSampleService(n_samples=10, n_outputs=1, low=1.0, high=2.0)

    samples, stats = sampler.generate_samples(stats=True)

    # for key, value in stats.iteritems():
    #     print("\n" + key + ": " + str(value))

    print(sampler.__repr__())

    print("\nStochastic truncated normal sampling:")

    sampler = StochasticSampleService(n_samples=10, n_outputs=2, sampler="norm",
                                      trunc_limits={"low": 0.0, "high": +3.0}, loc=1.0,
                                      scale=2.0)

    samples, stats = sampler.generate_samples(stats=True)

    # for key, value in stats.iteritems():
    #     print("\n" + key + ": " + str(value))

    print(sampler.__repr__())

    print("\nSensitivity analysis sampling:")

    sampler = StochasticSampleService(n_samples=10, n_outputs=2, sampler="latin", sampling_module="salib",
                                      bounds=[0.0, 0.1]*2)

    samples, stats = sampler.generate_samples(stats=True)

    # for key, value in stats.iteritems():
    #     print("\n" + key + ": " + str(value))

    print(sampler.__repr__())
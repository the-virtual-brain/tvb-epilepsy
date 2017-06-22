import warnings

import numpy as np
import numpy.random as nr
import scipy.stats as ss
import scipy as scp

from collections import OrderedDict

from tvb_epilepsy.base.utils import shape_to_size, formal_repr, dict_str


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

    def __init__(self, shape=(1,)):

        self.sample_engine = None
        self.shape = shape
        self.size = shape_to_size(self.shape)
        self.stats = {}
        self.params = {}

    def compute_stats(self, samples):
        return OrderedDict([("mu", samples.mean()), ("m", scp.median(samples)), ("mode", ss.mode(samples)),
                            ("std", samples.std()), ("var", samples.var()),
                            ("k", ss.kurtosis(samples)), ("skew", ss.skew(samples)),
                            ("min", samples.min()), ("max", samples.max()),
                            ("1%", np.percentile(samples, 1)), ("5%", np.percentile(samples, 5)),
                            ("10%", np.percentile(samples, 10)), ("p25", np.percentile(samples, 25)),
                            ("p50", np.percentile(samples, 50)), ("p75", np.percentile(samples, 75)),
                            ("p90", np.percentile(samples, 90)), ("p95", np.percentile(samples, 95)),
                            ("p99", np.percentile(samples, 99))])

    def generate_samples(self, stats=False):
        samples = self.sample()
        self.stats = self.compute_stats(samples)
        if stats:
            return samples, self.stats
        else:
            return samples

    def __repr__(self):
        d = OrderedDict([("shape", str(self.shape)), ("sample_engine", str(self.sample_engine))])

        return formal_repr(self, d) + \
               "\nparameters: " + dict_str(self.params) + \
               "\nstats: " + dict_str(self.stats)

class DeterministicSampleService(SampleService):

    def __init__(self, shape=(1,), low=0.0, high=1.0):

        super(DeterministicSampleService, self).__init__(shape)

        self.sample_engine = np.linspace

        if high <= low:
            raise ValueError("\nHigh limit of linear space " + str(high) +
                             " is not greater than the lower one " + str(low) + "!")
        else:
            self.params = {"low": low, "high": high}

    def sample(self):
        return np.reshape(self.sample_engine(self.params["low"], self.params["high"], self.size), self.shape)


# TODO: Add pystan as a stochastic sampling module, when/if needed.

class StochasticSampleService(SampleService):

    def __init__(self, shape=(1,), distribution="uniform", trunc_limits=None, sampling_module="numpy", random_seed=None,
                 **kwargs):

        super(StochasticSampleService, self).__init__(shape)

        self.random_seed = random_seed
        self.distribution = distribution
        self.params = kwargs

        if isinstance(trunc_limits, dict):

            self.trunc_limits = trunc_limits

            # We use inverse transform sampling for truncated distributions...

            if sampling_module is not "scipy":
                warnings.warn("\nSelecting scipy module for truncated distributions")

            self.sampling_module = ss
            self.sampling_engine = lambda **some_kwargs: \
                self.truncate_distribution(getattr(self.sampling_module, self.distribution), **some_kwargs)

        else:

            self.trunc_limits = {}

            if sampling_module == "numpy":
                self.sampling_module = nr
                self.sampling_engine = getattr(self.sampling_module, distribution)

            elif sampling_module == "scipy":
                self.sampling_module = ss
                self.sampling_engine = getattr(self.sampling_module, distribution).rvs

            else:
                raise ValueError("Sampler module " + str(sampling_module) + " is not recognized!")

    def truncate_distribution(self, distribution, size=1, **kwargs):
        # Following: https://stackoverflow.com/questions/25141250/
        # how-to-truncate-a-numpy-scipy-exponential-distribution-in-an-efficient-way
        # TODO: to have distributions parameters valid for the truncated distributions instead for the original one
        # pystan might be needed for that...
        rnd_cdf = nr.uniform(distribution.cdf(x=self.trunc_limits.get("low", distribution.a), **kwargs),
                             distribution.cdf(x=self.trunc_limits.get("high", distribution.b), **kwargs),
                             size=size)
        return distribution.ppf(q=rnd_cdf, **kwargs)

    def sample(self):
        nr.seed(self.random_seed)
        return np.reshape(self.sampling_engine(size=self.size, **self.params), self.shape)

    def __repr__(self):
        d = OrderedDict([("distribution", self.distribution), ("shape", str(self.shape)),
                         ("random_seed", str(self.random_seed)), ("sampling_module", str(self.sampling_module)),
                         ("sample_engine", str(self.sample_engine))])

        return formal_repr(self, d) + "\n" + \
               "\ntruncation limits: " + dict_str(self.trunc_limits) + "\n"


if __name__ == "__main__":

    print("\nDeterministic linspace sampling:")

    sampler = DeterministicSampleService(shape=(10,), low=1.0, high=2.0)

    samples, stats = sampler.generate_samples(stats=True)

    # for key, value in stats.iteritems():
    #
    #     print("\n" + key + ": " + str(value))

    print(sampler.__repr__())

    print("\nStochastic uniform sampling:")

    sampler = StochasticSampleService(shape=(10,), low=1.0, high=2.0)

    samples, stats = sampler.generate_samples(stats=True)

    # for key, value in stats.iteritems():
    #     print("\n" + key + ": " + str(value))

    print(sampler.__repr__())

    print("\nStochastic truncated normal sampling:")

    sampler = StochasticSampleService(shape=(10,), distribution="norm", trunc_limits={"low": 0, "high": +3.0}, loc=1.0,
                                      scale=2.0)

    samples, stats = sampler.generate_samples(stats=True)

    # for key, value in stats.iteritems():
    #     print("\n" + key + ": " + str(value))

    print(sampler.__repr__())
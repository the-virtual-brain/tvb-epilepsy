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


# A helper function to match normal distributions with several others we might use...

distributions = {"uniform": {"constraint": lambda p: np.all(p["alpha"] < p["beta"]),
                             "constraint_str": "alpha < beta",
                             "to_mu_std": lambda p: (0.5 * (p["alpha"] + p["beta"]),
                                                     0.5 * (p["beta"] - p["alpha"]) / np.sqrt(3)),
                             "from_mu_std": lambda mu, std: {"alpha": mu - std, "beta": mu + std},

                "lognormal": {"constraint": lambda p: np.all(p["sigma"] > 0),
                              "constraint_str": "sigma > 0",
                             "to_mu_std": lambda p: lognormal_to_mu_std(p),
                             "from_mu_std": lambda mu, std: lognormal_from_mu_std(mu, std)},

                "chisquare": {"constraint": lambda p: np.all(p["k"] == np.round(p["k"])) and np.all(p["k"] > 0),
                              "constraint_str": " k == round(k) and k > 0",
                             "to_mu_std": lambda p: p["k"],
                             "from_mu_std": lambda mu, std: {"k": mu}},

                "gamma": {"constraint": lambda p: (np.all(p.get("alpha", -1.0) > 0.0) and
                                                   np.all(p.get("beta", -1.0) > 0.0))
                                                  or (np.all(p.get("k", -1.0) > 0.0) and
                                                      np.all(p.get("theta", -1.0) > 0.0)),
                          "constraint_str": "(alpha > 0 and beta > 0) or (k > 0 and theta > 0)",
                          "to_mu_std": lambda p: gamma_to_mu_std(p)},
                          "from_mu_std": lambda mu, std: gamma_from_mu_std(mu, std)},



                 }

def lognormal_from_mu_std(mu, std):
    mu2 = np.power(mu, 2)
    var = np.power(std, 2)
    return {"mu": np.log(mu2 / np.sqrt(var + mu2)), "sigma": np.sqrt(np.log(1.0 + var / mu2))}

def lognormal_to_mu_std(p):
    sigma2 = np.power(p["sigma"], 2)
    return np.exp(p["mu"] + 0.5 * sigma2), np.sqrt(np.exp(2.0 * p["mu"] + sigma2) * (np.exp(sigma2) - 1.0))

def gamma_from_mu_std(mu, std):
    k = np.power(mu / std, 2)
    theta = np.power(std) / mu
    return {"k": k, "theta": theta, "alpha": k, "beta": 1.0 / theta}

def gamma_to_mu_std(p):
    if p.get("a", False) and p.get("beta", False):

        return p["a"] / p["beta"], np.sqrt(p["a"]) / p["beta"]

    elif p.get("k", False) and p.get("theta", False):

        return p["k"] * p["theta"], np.sqrt(p["k"]) * p["theta"]
    else:
        raise ValueError("The input gamma distribution parameters are neither of the a, beta system, nor of the "
                         "k, theta one!")


def mean_std_to_distribution_params(distribution="uniform", mu=0.0, std=1.0):

    p = {}

    if distribution is "uniform":

        p["alpha"] = mu - std
        p["beta"] = mu + std

    elif distribution is "lognormal":

        mu2 = np.power(mu, 2)
        var = np.power(std, 2)

        p["mu"] = np.log(mu2 / np.sqrt(var + mu2))

        p["sigma"] = np.sqrt(np.log(1.0 + var / mu2))

    elif distribution is "chisquare":

        p["k"] = mu

        if std != np.sqrt(2.0*mu):
            warnings.warn("std is not equal to sqrt(2*mu) as it should be for chisquare distribution!")

    elif distribution is "gamma":

        p["k"] = np.power(mu / std, 2)

        p["theta"] = np.power(std) / mu

        p["a"] = p["k"]

        p["beta"] = 1.0 / p["theta"]

    elif distribution is "exponential":

        if std != mu:
            warnings.warn("std is not equal to mu as it should be for exponential distribution!")

        p["lambda"] = 1.0 / mu

        if not (np.all(p["lambda"] > 0)):
            raise ValueError("It should be lambda = " + str(p["lambda"]) + " > 0 !")

    elif distribution is "beta":

        var = np.power(std, 2)
        mu1 = (1.0 - mu)

        if var < mu * mu1:

            vmu = mu * mu1 / var - 1.0

            p["alpha"] = mu * vmu
            p["beta"] = mu1 * vmu

            if not(np.all(p["alpha"] > 0 and p["beta"] > 0)):
                raise ValueError("It should be alpha = " + str(p["alpha"]) + " > 0 and  "
                                 + "beta = " + str(p["beta"]) + " > 0 !")

        else:
            raise ValueError("Variance = " + str(var) + " has to be smaller than the quantity mu*(1-mu) = " + str(mu1)
                             + " !")

    elif distribution is "binomial":

        var = np.power(std, 2)
        vm = var / mu

        p["p"] = 1.0 + vm

        p["n"] = mu / p["p"]

        if np.all(p["n"] == np.round(p["n"]) >= 0 and 0.0 > p["p"] <= 1.0):
            p["n"] = np.int(p["n"])

        else:
            raise ValueError("n = " + str(p["n"]) + " has to be a positive integer number and p = " + str(p["p"]) +
                             " has to lie in the interval (0.0, 1.0]!")


    return p


def distribution_params_to_mean_std(distribution="uniform", p={"alpha":0.0, "beta":1.0}):

    if distribution is "uniform":

        mu = 0.5 * (p["alpha"] + p["beta"])

        std = 0.5 * (p["beta"] - p["alpha"]) / np.sqrt(3)

    elif distribution is "lognormal":

        sigma2 = np.power(p["sigma"], 2)
        mu = np.exp(p["mu"] + 0.5 * sigma2)

        std = np.sqrt(np.exp(2.0 * p["mu"] + sigma2) * (np.exp(sigma2) - 1.0))

    elif distribution is "chisquare":

        mu = p["k"]

        std = np.sqrt(2.0 * p["k"])

    elif distribution is "gamma":

        if p.get("a", False) and p.get("beta", False):

            mu = p["a"] / p["beta"]

            std = np.sqrt(p["a"]) / p["beta"]

        elif p.get("k", False) and p.get("theta", False):

            mu = p["k"] * p["theta"]

            std = np.sqrt(p["k"]) * p["theta"]

        else:
            raise ValueError("The input gamma distribution parameters are neither of the a, beta system, nor of the "
                             "k, theta one!")

    elif distribution is "exponential":

        if not (np.all(p["lambda"] > 0)):
            raise ValueError("It should be lambda = " + str(p["lambda"]) + " > 0 !")

        mu = 1.0 / p["lambda"]
        std = mu

    elif distribution is "beta":

        if np.all(p["alpha"] > 0 and p["beta"] > 0):

            ab = p["alpha"] + p["beta"]
            mu = p["alpha"] / ab
            std = np.sqrt(p["alpha"] * p["beta"] / (ab + 1.0)) / ab

        else:
            raise ValueError("It should be alpha = " + str(p["alpha"]) + " > 0 and  "
                             + "beta = " + str(p["beta"]) + " > 0 !")

    elif distribution is "binomial":

        if np.all(p["n"] == np.round(p["n"]) >= 0 and 0.0 > p["p"] <= 1.0):

            mu = p["n"] * p["p"]
            std = np.sqrt(mu * (1 - p["p"]))

        else:
            raise ValueError("n = " + str(p["n"]) + " has to be a positive integer number and p = " + str(p["p"]) +
                             " has to lie in the interval (0.0, 1.0]!")



    return mu, std


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
        d = OrderedDict([("n_parameters", str(self.n_outputs)), ("n_samples", str(self.n_samples)),
                         ("shape", str(self.shape)), ("sampler", self.sampling_module)])

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
                   self.truncated_distribution_sampling(getattr(ss, sampler), trunc_limits, **some_kwargs)

            else:

                if sampling_module == "numpy":
                    self.sampling_module = "numpy.random." + sampler
                    self.sampler = getattr(nr, sampler)

                elif sampling_module == "scipy":
                    self.sampling_module = "scipy.stats." + sampler + ".rvs"
                    self.sampler = getattr(ss, sampler).rvs

                else:
                    raise ValueError("Sampler module " + str(sampling_module) + " is not recognized!")

    def truncated_distribution_sampling(self, distribution, trunc_limits, size=1, **kwargs):
        # Following: https://stackoverflow.com/questions/25141250/
        # how-to-truncate-a-numpy-scipy-exponential-distribution-in-an-efficient-way
        # TODO: to have distributions parameters valid for the truncated distributions instead for the original one
        # pystan might be needed for that...
        rnd_cdf = nr.uniform(distribution.cdf(x=trunc_limits.get("alpha", -np.inf), **kwargs),
                             distribution.cdf(x=trunc_limits.get("beta", np.inf), **kwargs),
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
                if len(params) == 0:
                    for io in range(self.n_outputs):
                        samples.append(self.sampler(trunc_limits[io], size=self.n_samples))
                elif len(params) == self.n_outputs:
                    for io in range(self.n_outputs):
                        samples.append(self.sampler(trunc_limits[io], size=self.n_samples, **(params[io])))
                else:
                    raise ValueError("Parameters are neither an empty list nor a list of length n_parameters = "
                                     + str(self.n_outputs) + " but one of length " + str(len(self.params)) + " !")

            else:
                samples = []
                if len(params) == 0:
                    for io in range(self.n_outputs):
                        samples.append(self.sampler(size=self.n_samples))
                elif len(params) == self.n_outputs:
                    for io in range(self.n_outputs):
                        samples.append(self.sampler(size=self.n_samples, **(params[io])))
                else:
                    raise ValueError("Parameters are neither an empty list nor a list of length n_parameters = "
                                     + str(self.n_outputs) + " but one of length " + str(len(self.params)) + " !")

        return np.reshape(samples, self.shape)

    def __repr__(self):

        d = OrderedDict([("random_seed", str(self.random_seed))])

        return super(StochasticSampleService, self).__repr__() + "\n"+ \
               formal_repr(self, d) + "\n" + "\ntruncation limits: " + dict_str(self.trunc_limits) + "\n"


if __name__ == "__main__":

    print("\nDeterministic linspace sampling:")

    sampler = DeterministicSampleService(n_samples=10, n_outputs=2, low=1.0, high=2.0, grid_mode=True)

    samples, stats = sampler.generate_samples(stats=True)

    # for key, value in stats.iteritems():
    #
    #     print("\n" + key + ": " + str(value))

    print(sampler.__repr__())

    print("\nStochastic uniform sampling:")

    sampler = StochasticSampleService() #(n_samples=10, n_outputs=1, low=1.0, high=2.0)

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
                                      bounds=[[0.0, 0.1], [0.0, 0.1]])

    samples, stats = sampler.generate_samples(stats=True)

    # for key, value in stats.iteritems():
    #     print("\n" + key + ": " + str(value))

    print(sampler.__repr__())
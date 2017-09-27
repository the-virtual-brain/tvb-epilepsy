
import importlib
from collections import OrderedDict

import numpy as np
import numpy.random as nr
import scipy.stats as ss
import scipy as scp
from SALib.sample import saltelli, fast_sampler, morris, ff

from tvb_epilepsy.base.utils import initialize_logger, formal_repr, warning, raise_value_error, \
                                    raise_not_implemented_error, dict_str, dicts_of_lists, \
                                    dicts_of_lists_to_lists_of_dicts
from tvb_epilepsy.base.h5_model import convert_to_h5_model

from tvb.basic.logger.builder import get_logger

# Helper functions to match normal distributions with several others we might use...
# Input parameters should be scalars or ndarrays! Not lists!

logger = initialize_logger(__name__)


distrib_dict = {"uniform": {"constraint": lambda p: np.all(p["alpha"] < p["beta"]),
                             "constraint_str": "alpha < beta",
                             "to_mu_std": lambda p: (0.5 * (p["alpha"] + p["beta"]),
                                                     0.5 * (p["beta"] - p["alpha"]) / np.sqrt(3)),
                             "from_mu_std": lambda mu, std: {"alpha": mu - std * np.sqrt(3),
                                                             "beta": mu + std* np.sqrt(3)}
                            },
                 "lognormal": {"constraint": lambda p: np.all(p["sigma"] > 0.0),
                               "constraint_str": "sigma > 0",
                               "to_mu_std": lambda p: lognormal_to_mu_std(p),
                               "from_mu_std": lambda mu, std: lognormal_from_mu_std(mu, std)
                               },
                 "chisquare": {"constraint": lambda p: np.all(np.abs(p["k"] - np.round(p["k"]) < 10 ** -6))
                                                       and np.all(p["k"] > 0),
                               "constraint_str": "abs(k - round(k)) < 10 ** -6 and k > 0",
                               "to_mu_std": lambda p: (p["k"], np.sqrt(2*p["k"])),
                               "from_mu_std": lambda mu, std: {"k": mu} },

                 "gamma": {"constraint": lambda p: (np.all(p.get("alpha", -1.0) > 0.0) and
                                                   np.all(p.get("beta", -1.0) > 0.0))
                                                  or (np.all(p.get("k", -1.0) > 0.0) and
                                                      np.all(p.get("theta", -1.0) > 0.0)),
                           "constraint_str": "(alpha > 0 and beta > 0) or (k > 0 and theta > 0)",
                           "to_mu_std": lambda p: gamma_to_mu_std(p),
                           "from_mu_std": lambda mu, std: gamma_from_mu_std(mu, std)
                           },
                 "exponential": {"constraint": lambda p: np.all(p["lambda"] > 0.0),
                                 "constraint_str": "lambda > 0",
                                 "to_mu_std": lambda p: (1.0 / p["lambda"], 1.0 / p["lambda"]),
                                 "from_mu_std": lambda mu, std: {"lambda": 1.0 / mu}
                                 },
                 "beta": {"constraint": lambda p: np.all(p["alpha"] > 0.0) and np.all(p["beta"] > 0.0),
                          "constraint_str": "alpha > 0 and beta > 0",
                          "to_mu_std": lambda p: beta_to_mu_std(p),
                          "from_mu_std": lambda mu, std: beta_from_mu_std(mu, std)
                          },
                 "binomial": {"constraint": lambda p: np.all(np.abs(p["n"] - np.round(p["n"])) < 10 ** -6)
                                                      and np.all(p["n"] > 0)
                                                      and np.all(p["p"] > 0.0) and np.all(p["p"] < 1.0),
                              "constraint_str": "abs(n - round(n)) < 10 ** -6 and n > 0 and p > 0.0 and p < 1.0",
                              "to_mu_std": lambda p: binomial_to_mu_std(p),
                              "from_mu_std": lambda mu, std: binomial_from_mu_std(mu, std)
                              },
                 "poisson": {"constraint": lambda p: np.all(p["lambda"] > 0.0),
                             "constraint_str": "lambda > 0",
                             "to_mu_std": lambda p: (p["lambda"], np.sqrt(p["lambda"])),
                             "from_mu_std": lambda mu, std: {"lambda": mu}
                             },
                 "bernoulli": {"constraint": lambda p: np.all(p["p"] > 0.0) and np.all(p["p"] < 1.0),
                               "constraint_str": "p > 0.0 and p < 1.0",
                               "to_mu_std": lambda p: (p["p"], np.sqrt(p["p"] * (1 - p["p"]))),
                               "from_mu_std": lambda mu, std: {"p": mu}
                               },
                 }


def lognormal_from_mu_std(mu, std):
    mu2 = mu ** 2
    var = std ** 2
    return {"mu": np.log(mu2 / np.sqrt(var + mu2)), "sigma": np.sqrt(np.log(1.0 + var / mu2))}


def lognormal_to_mu_std(p):
    sigma2 = p["sigma"] ** 2
    return np.exp(p["mu"] + 0.5 * sigma2), np.sqrt(np.exp(2.0 * p["mu"] + sigma2) * (np.exp(sigma2) - 1.0))


def gamma_from_mu_std(mu, std):
    k = (mu / std) ** 2
    theta = std ** 2 / mu
    return {"k": k, "theta": theta, "alpha": k, "beta": 1.0 / theta}


def gamma_to_mu_std(p):
    if p.get("a", False) and p.get("beta", False):
        return p["a"] / p["beta"], np.sqrt(p["a"]) / p["beta"]

    elif p.get("k", False) and p.get("theta", False):
        return p["k"] * p["theta"], np.sqrt(p["k"]) * p["theta"]

    else:
        raise_value_error("The input gamma distribution parameters are neither of the a, beta system, nor of the "
                          "k, theta one!")


def beta_from_mu_std(mu, std):
    var = std ** 2
    mu1 = 1.0 - mu

    if var < mu * mu1:
        vmu = mu * mu1 / var - 1.0
        return {"alpha": mu * vmu, "beta": mu1 * vmu}

    else:
        raise_value_error("Variance = " + str(var) + " has to be smaller than the quantity mu*(1-mu) = " + str(mu1)
                           + " !")


def beta_to_mu_std(p):
    ab = p["alpha"] + p["beta"]
    return p["alpha"] / ab, np.sqrt(p["alpha"] * p["beta"] / (ab + 1.0)) / ab


def binomial_from_mu_std(mu, std):
    var = std ** 2
    vm = var / mu
    p = {"p": 1.0 - vm}
    p.update({"n": mu / p["p"]})
    return p


def binomial_to_mu_std(p):
    mu = p["n"] * p["p"]
    return mu, np.sqrt(mu * (1 - p["p"]))


def mean_std_to_distribution_params(distribution, mu, std=1.0):

    if np.any(std <= 0.0):
        raise_value_error("Standard deviation std = " + str(std) + " <= 0!")

    std_check = {"exponential": lambda mu: mu,
                 "poisson": lambda mu: np.sqrt(mu),
                 "chisquare": lambda mu: np.sqrt(2.0*mu),
                 "bernoulli": lambda mu: np.sqrt(mu * (1.0-mu))
                }
    if np.in1d(distribution, ["exponential", "poisson", "chisquare", "bernoulli"]):
        std_check = std_check[distribution](mu)
        if std != std_check:
            msg = "\nmu = " +  str(mu) + "\nstd = " + str(std) + "\nstd should be = " + str(std_check)
            warning(msg + "\nStandard deviation constraint not satisfied for distribution " + distribution + "!)")

    p = distrib_dict[distribution]["from_mu_std"](mu, std)

    if distrib_dict[distribution]["constraint"](p):
        return p

    else:
        for key, val in p.iteritems():
            logger.info("\n" + str(key) + ": " + str(val))
        raise_value_error("\nDistribution parameters'constraints " + distrib_dict[distribution]["constraint_str"]
                         + " is not met!")


def distribution_params_to_mean_std(distribution, **p):

    if distrib_dict[distribution]["constraint"](p):

        mu, std = distrib_dict[distribution]["to_mu_std"](p)

        if np.any(std <= 0.0):
            raise_value_error("\nStandard deviation std = " + str(std) + " <= 0!")

        return mu, std

    else:
        for key, val in p.iteritems():
            logger.info("\n" + str(key) + ": " + str(val))
        raise_value_error("\nDistribution parameters'constraints " + distrib_dict[distribution]["constraint_str"]
                         + " is not met!")


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

        #Adjust samples number:
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
                        samples.append(self._truncated_distribution_sampling(self.sampler,trunc_limits[io],
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

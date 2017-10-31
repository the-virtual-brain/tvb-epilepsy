import numpy as np
import scipy.stats as ss

from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error, warning
from tvb_epilepsy.service.sampling_service import logger


# Helper functions to match normal distributions with several others we might use...
# Input parameters should be scalars or ndarrays! Not lists!


distrib_dict = {"uniform": {"constraint": lambda a, b: np.all(a < b),
                             "constraint_str": "a < b",
                             "mu": lambda a, b: 0.5 * (a + b),
                             "med": lambda a, b: 0.5 * (a + b),
                             "mod": raise_value_error("No definite mode for uniform distribution!"),
                             "std": lambda a, b: 0.5 * (b - a) / np.sqrt(3),
                             # "from_mu_std": lambda mu, std: {"a": mu - std * np.sqrt(3),
                             #                                 "b": mu + std * np.sqrt(3)}
                            },
                 "normal": {"constraint": lambda mu, sigma: np.all(sigma > 0.0),
                             "constraint_str": "sigma > 0",
                             "mu": lambda mu, sigma: mu,
                             "med": lambda mu, sigma: mu,
                             "mod": lambda mu, sigma: mu,
                             "std": lambda mu, sigma: sigma,
                             # "from_mu_std": lambda mu, std: {"mu": mu,
                             #                                 "sigma": std}
                            },
                 "lognormal": {"constraint": lambda mu, sigma: np.all(sigma > 0.0),
                               "constraint_str": "sigma > 0",
                               "mu": lambda mu, sigma: lognormal_to_mu(mu, sigma),
                               "med": lambda mu, sigma: lognormal_to_med(mu, sigma),
                               "mod": lambda mu, sigma: lognormal_to_mod(mu, sigma),
                               "std": lambda mu, sigma: lognormal_to_std(mu, sigma),
                               # "from_mu_std": lambda mu, std: lognormal_from_mu_std(mu, std)
                               },

                 "gamma": {"constraint": lambda alpha=-1, beta=-1, k=-1, theta=-1:
                                                  (np.all(alpha > 0.0) and np.all(beta > 0.0))
                                                  or (np.all(k > 0.0) and np.all(theta > 0.0)),
                           "constraint_str": "(alpha > 0 and beta > 0) or (k > 0 and theta > 0)",
                           "mu": lambda alpha=-1, beta=-1, k=-1, theta=-1: gamma_to_mu(alpha, beta, k, theta),
                           "med": raise_value_error("No simple close form median for gamma distribution!"),
                           "mod": lambda alpha=-1, beta=-1, k=-1, theta=-1: gamma_to_mod(alpha, beta, k, theta),
                           "std": lambda alpha=-1, beta=-1, k=-1, theta=-1: gamma_to_std(alpha, beta, k, theta),
                           },
                "chisquare": {"constraint": lambda p: np.all(np.abs(p["k"] - np.round(p["k"]) < 10 ** -6))
                                                      and np.all(p["k"] > 0),
                              "constraint_str": "abs(k - round(k)) < 10 ** -6 and k > 0",
                              "to_mu_std": lambda p: (p["k"], np.sqrt(2 * p["k"])),
                              "from_mu_std": lambda mu, std: {"k": mu}
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


def lognormal_to_mu(mu, sigma):
    return np.exp(mu + 0.5 * sigma ** 2)


def lognormal_to_med(mu, sigma):
    return np.exp(mu)


def lognormal_to_mod(mu, sigma):
    return np.exp(mu - 0.5 * sigma ** 2)


def lognormal_to_std(mu, sigma):
    sigma2 = sigma ** 2
    return np.sqrt(np.exp(2.0 * mu + sigma2) * (np.exp(sigma2) - 1.0))


def gamma_from_mu_std(mu, std):
    k = (mu / std) ** 2
    theta = std ** 2 / mu
    return {"k": k, "theta": theta, "alpha": k, "beta": 1.0 / theta}


def gamma_to_mu(alpha=-1, beta=-1, k=-1, theta=-1):
    if (np.all(alpha > 0.0) and np.all(beta > 0.0)):
        return alpha / beta
    elif (np.all(k > 0.0) and np.all(theta > 0.0)):
        return k * theta
    else:
        raise_value_error("The input gamma distribution parameters are neither of the a, beta system, nor of the "
                          "k, theta one!")


def gamma_to_mod(alpha=-1, beta=-1, k=-1, theta=-1):
    if (np.all(alpha > 0.0) and np.all(beta > 0.0)):
        if alpha >= 1.0:
            return (alpha - 1.0) / beta
        else:
            raise_value_error("alpha = " + str(alpha) +
                              " is not >= 1.0, as it should be for location (mode) to be well defined!")
    elif (np.all(k > 0.0) and np.all(theta > 0.0)):
        if k  >= 1.0:
            return (k  - 1.0) * theta
        else:
            raise_value_error("k = " + str(k) +
                              " is not >= 1.0, as it should be for location (mode) to be well defined!")
    else:
        raise_value_error("The input gamma distribution parameters are neither of the a, beta system, nor of the "
                          "k, theta one!")


def gamma_to_std(alpha=-1, beta=-1, k=-1, theta=-1):
    if (np.all(alpha > 0.0) and np.all(beta > 0.0)):
        return alpha / beta, np.sqrt(alpha) / beta
    elif (np.all(k > 0.0) and np.all(theta > 0.0)):
        return np.sqrt(k) * theta
    else:
        raise_value_error("The input gamma distribution parameters are neither of the a, beta system, nor of the "
                          "k, theta one!")



def gamma_to_loc_sc(p):
    if p.get("alpha", False) and p.get("beta", False):
        if p["alpha"] >= 1.0:
            sc = 1.0 / p["beta"]
            return (p["alpha"] - 1.0) * sc, sc
        else:
            raise_value_error("alpha = " + str(p["alpha"]) +
                              " is not >= 1.0, as it should be for location (mode) to be well defined!")

    elif p.get("k", False) and p.get("theta", False):
        if p["k"] >= 1.0:
            return (p["k"] - 1.0) * p["theta"], p["theta"]
        else:
            raise_value_error("k = " + str(p["k"]) +
                              " is not >= 1.0, as it should be for location (mode) to be well defined!")
    else:
        raise_value_error("The input gamma distribution parameters are neither of the a, beta system, nor of the "
                          "k, theta one!")


def gamma_from_mu_std(loc, sc):
    k = loc / sc + 1.0
    return {"k": k, "theta": sc, "alpha": k, "beta": 1.0 / sc}


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


def mean_std_to_distribution_params(distribution, mu, std=1.0, output="dict", **kwargs):

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
        if isequal_string(output, "dict"):
            return p
        else:
            if isequal_string(distribution, "gamma"):
                gamma_mode = kwargs.get("gamma_mode")
                if isequal_string(str(gamma_mode), "alpha_beta"):
                    return (p["alpha"], p["beta"])
                elif isequal_string(str(gamma_mode), "k_theta"):
                    return (p["k"], p["theta"])
            else:
                p = tuple(p.values())
                if len(p) == 1:
                    return p[0]
                else:
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
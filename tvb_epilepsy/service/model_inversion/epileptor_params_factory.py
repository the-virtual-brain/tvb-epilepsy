
import numpy as np

from tvb_epilepsy.base.computations.probability_distributions import ProbabilityDistributionTypes
from tvb_epilepsy.base.model.probabilistic_models.probabilistic_parameter import NegativeLognormal
from tvb_epilepsy.service.probabilistic_parameter_builder import generate_probabilistic_parameter


def generate_lognormal_parameter(name, mean, low, high, sigma=None, sigma_scale=2, p_shape=(), use="scipy"):
    if sigma is None:
        sigma = np.abs(mean - low) / sigma_scale
    logsm21 = np.log((sigma / mean) ** 2 + 1)
    mu = np.log(mean) - 0.5 * logsm21
    sigma = np.sqrt(logsm21)
    return generate_probabilistic_parameter(name, low, high, loc=0.0, scale=1.0, p_shape=p_shape,
                                            probability_distribution=ProbabilityDistributionTypes.LOGNORMAL,
                                            optimize_pdf=False, use=use, **{"mu": mu, "sigma": sigma})


def generate_negative_lognormal_parameter(name, mean, low, high, sigma=None, sigma_scale=2, p_shape=(), use="scipy"):
    parameter = generate_lognormal_parameter(name.split("_star")[0]+"_star", high - mean, 0.0, high - low,
                                             sigma, sigma_scale, p_shape, use)

    return NegativeLognormal(parameter.name.split("_star")[0], "NegativeLognormal", parameter, high)

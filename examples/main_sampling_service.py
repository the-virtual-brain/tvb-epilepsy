from copy import deepcopy

import numpy as np

from tvb_epilepsy.base.constants.configurations import FOLDER_RES
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning
from tvb_epilepsy.base.model.statistical_models.stochastic_parameter import generate_stochastic_parameter
from tvb_epilepsy.io.writer_custom import CustomH5Writer
from tvb_epilepsy.service.probability_distribution_factory import AVAILABLE_DISTRIBUTIONS
from tvb_epilepsy.service.stochastic_parameter_factory import set_parameter, set_parameter_defaults
from tvb_epilepsy.service.sampling.deterministic_sampling_service import DeterministicSamplingService
from tvb_epilepsy.service.sampling.salib_sampling_service import SalibSamplingService
from tvb_epilepsy.service.sampling.stochastic_sampling_service import StochasticSamplingService

logger = initialize_logger(__name__)

if __name__ == "__main__":
    n_samples = 100
    logger.info("\nDeterministic numpy.linspace sampling:")
    sampler = DeterministicSamplingService(n_samples=n_samples, grid_mode=True)
    samples, stats = sampler.generate_samples(low=1.0, high=2.0, shape=(2,), stats=True)
    for key, value in stats.iteritems():
        print("\n" + key + ": " + str(value))
    logger.info(sampler.__repr__())
    writer = CustomH5Writer()
    writer.write_generic(sampler, FOLDER_RES, "test_Stochastic_Sampler.h5", subtype=sampler.__class__.__name__)

    logger.info("\nStochastic uniform sampling with numpy:")
    sampler = StochasticSamplingService(n_samples=n_samples, sampling_module="numpy")
    #                                      a (low), b (high)
    samples, stats = sampler.generate_samples(parameter=(1.0, 2.0), probability_distribution="uniform", shape=(2,),
                                              stats=True)
    for key, value in stats.iteritems():
        print("\n" + key + ": " + str(value))

    logger.info(sampler.__repr__())
    writer.write_generic(sampler, FOLDER_RES, "test1_Stochastic_Sampler.h5", subtype=sampler.__class__.__name__)

    logger.info("\nStochastic truncated normal sampling with scipy:")
    sampler = StochasticSamplingService(n_samples=n_samples)
    #                                   loc (mean), scale (sigma)
    samples, stats = sampler.generate_samples(parameter=(1.5, 1.0), probability_distribution="norm", low=1, high=2,
                                              shape=(2,), stats=True)
    for key, value in stats.iteritems():
        print("\n" + key + ": " + str(value))
    logger.info(sampler.__repr__())
    writer.write_generic(sampler, FOLDER_RES, "test2_Stochastic_Sampler.h5", subtype=sampler.__class__.__name__)

    logger.info("\nSensitivity analysis sampling:")
    sampler = SalibSamplingService(n_samples=n_samples, sampler="latin")
    samples, stats = sampler.generate_samples(low=1, high=2, shape=(2,), stats=True)
    for key, value in stats.iteritems():
        print("\n" + key + ": " + str(value))
    logger.info(sampler.__repr__())
    writer.write_generic(sampler, FOLDER_RES, "test3_Stochastic_Sampler.h5", subtype=sampler.__class__.__name__)

    logger.info("\nTesting distribution class and conversions...")
    sampler = StochasticSamplingService(n_samples=n_samples)
    for distrib_name in AVAILABLE_DISTRIBUTIONS:
        print("\n" + distrib_name)
        logger.info("\nmode/mean, std to distribution " + distrib_name + ":")
        if np.in1d(distrib_name, ["exponential", "chisquare"]):
            target_stats = {"mean": 1.0}
            stats_m = "mean"
        elif np.in1d(distrib_name, ["bernoulli", "poisson"]):
            target_stats = {"mean": np.ones((2,))}
            stats_m = "mean"
        elif isequal_string(distrib_name, "binomial"):
            target_stats = {"mean": 1.0, "std": 2.0}
            stats_m = "mean"
        else:
            if isequal_string(distrib_name, "uniform"):
                target_stats = {"mean": 1.0, "std": 2.0}
                stats_m = "mean"
            else:
                target_stats = {"mean": 1.0, "std": 2.0}
                stats_m = "mean"
        parameter1 = generate_stochastic_parameter(name="test1_" + distrib_name, low=0.0, high=2.0, p_shape=(2, 2),
                                                   probability_distribution=distrib_name, optimize_pdf=True,
                                                   use="manual", **target_stats)
        name2 = "test2_" + distrib_name
        defaults = set_parameter_defaults(name2, _pdf=distrib_name, _shape=(2, 2), _lo=0.0, _hi=2.0,
                                          **(deepcopy(target_stats)))
        parameter2 = set_parameter(name=name2, use="manual", **defaults)
        for parameter in (parameter1, parameter2):
            logger.info(str(parameter))
            samples = sampler.generate_samples(parameter=parameter, stats=True)
            for key, value in stats.iteritems():
                print("\n" + key + ": " + str(value))
            diff = target_stats[stats_m] - stats[stats_m]
            if np.any(np.abs(diff.flatten()) > 0.001):
                warning("Large difference between target and resulting samples' " + stats_m + "!: " + str(diff))
            del (parameter)

import numpy as np
from tvb_epilepsy.base.configurations import FOLDER_RES
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.probability_distributions.probability_distribution import \
                                                                          AVAILABLE_DISTRIBUTIONS, generate_distribution
from tvb_epilepsy.service.sampling.deterministic_sampling_service import  DeterministicSamplingService
from tvb_epilepsy.service.sampling.stochastic_sampling_service import StochasticSamplingService
from tvb_epilepsy.service.sampling.salib_sampling_service import SalibSamplingService


logger = initialize_logger(__name__)


if __name__ == "__main__":
    logger.info("\nDeterministic numpy.linspace sampling:")
    sampler = DeterministicSamplingService(n_samples=10, grid_mode=True)
    samples, stats = sampler.generate_samples(low=1.0, high=2.0, shape= (2,), stats=True)
    for key, value in stats.iteritems():
        print("\n" + key + ": " + str(value))
    logger.info(sampler.__repr__())
    sampler.write_to_h5(FOLDER_RES, "test_Stochastic_Sampler.h5")

    logger.info("\nStochastic uniform sampling with numpy:")
    sampler = StochasticSamplingService(n_samples=10, sampling_module="numpy")
    #                                      a (low), b (high)
    samples, stats = sampler.generate_samples(parameter=(1.0, 2.0), probability_distribution="uniform", shape=(2,),
                                              stats= True)
    for key, value in stats.iteritems():
        print("\n" + key + ": " + str(value))

    logger.info(sampler.__repr__())
    sampler.write_to_h5(FOLDER_RES, "test1_Stochastic_Sampler.h5")

    logger.info("\nStochastic truncated normal sampling with scipy:")
    sampler = StochasticSamplingService(n_samples=10)
    #                                   loc (mean), scale (sigma)
    samples, stats = sampler.generate_samples(parameter=(1.5, 1.0), probability_distribution="norm", low=1, high=2,
                                              shape=(2,), stats=True)
    for key, value in stats.iteritems():
        print("\n" + key + ": " + str(value))
    logger.info(sampler.__repr__())
    sampler.write_to_h5(FOLDER_RES, "test2_Stochastic_Sampler.h5")

    logger.info("\nSensitivity analysis sampling:")
    sampler = SalibSamplingService(n_samples=10, sampler="latin")
    samples, stats = sampler.generate_samples(low=1, high=2, shape=(2,), stats=True)
    for key, value in stats.iteritems():
        print("\n" + key + ": " + str(value))
    logger.info(sampler.__repr__())
    sampler.write_to_h5(FOLDER_RES, "test3_Stochastic_Sampler.h5")

    logger.info("\nTesting distribution class and conversions...")
    sampler = StochasticSamplingService(n_samples=100)
    for distrib_name in AVAILABLE_DISTRIBUTIONS:
        logger.info("\nmean, std to distribution " + distrib_name + ":")
        if np.in1d(distrib_name, ["exponential", "chisquare"]):
            target_stats = {"mean": 1.0}
        elif np.in1d(distrib_name, ["bernoulli", "poisson"]):
            target_stats = {"mean": 1.0}
        elif isequal_string(distrib_name, "binomial"):
            target_stats = {"mean": 1.0}
        else:
            target_stats = {"mean": 1.0, "std": 2.0}
        parameter = Parameter(probability_distribution=generate_distribution(distrib_name, target_stats=target_stats),
                              low=-10**6, high=10**6)
        logger.info(str(parameter))
        samples = sampler.generate_samples(parameter=parameter, stats=True)
        for key, value in stats.iteritems():
            print("\n" + key + ": " + str(value))
        diff = target_stats["mean"] - stats["mean"]
        if np.abs(diff) > 10 ** -2:
            warning("Large difference between target and resulting samples' mean!: " + str(diff))

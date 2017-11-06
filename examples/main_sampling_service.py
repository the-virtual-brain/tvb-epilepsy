import numpy as np
from tvb_epilepsy.base.configurations import FOLDER_RES
from tvb_epilepsy.base.utils.data_structures_utils import dict_str
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.probability_distributions.probability_distribution import \
                                                                                                 AVAILABLE_DISTRIBUTIONS
from tvb_epilepsy.base.model.statistical_models.probability_distributions.uniform_distribution import \
                                                                                                     UniformDistribution
from tvb_epilepsy.base.model.statistical_models.probability_distributions.normal_distribution import NormalDistribution
from tvb_epilepsy.service.sampling.deterministic_sampling_service import  DeterministicSamplingService
from tvb_epilepsy.service.sampling.stochastic_sampling_service import StochasticSamplingService
from tvb_epilepsy.service.sampling.salib_sampling_service import SalibSamplingService


logger = initialize_logger(__name__)


if __name__ == "__main__":
    logger.info("\nDeterministic numpy.linspace sampling:")
    sampler = DeterministicSamplingService(n_samples=10, grid_mode=True)
    samples, stats = sampler.generate_samples(low=1.0, high=2.0, shape= (2,), stats=True)
    # for key, value in stats.iteritems():
    #
    #     print("\n" + key + ": " + str(value))
    logger.info(sampler.__repr__())
    sampler.write_to_h5(FOLDER_RES, "test_Stochastic_Sampler.h5")

    logger.info("\nStochastic uniform sampling with numpy:")
    sampler = StochasticSamplingService(n_samples=10, sampling_module="numpy")
    #                                      a (low), b (high)
    samples, stats = sampler.generate_samples(parameter=(1.0, 2.0), probability_distribution="uniform", shape=(2,))
    # for key, value in stats.iteritems():
    #     print("\n" + key + ": " + str(value))

    logger.info(sampler.__repr__())
    sampler.write_to_h5(FOLDER_RES, "test1_Stochastic_Sampler.h5")

    logger.info("\nStochastic truncated normal sampling with scipy:")
    sampler = StochasticSamplingService(n_samples=10)
    #                                   loc (mean), scale (sigma)
    samples, stats = sampler.generate_samples(parameter=(1.5, 1.0), probability_distribution="norm", low=1, high=2,
                                              shape=(2,), stats=True)
    # for key, value in stats.iteritems():
    #     print("\n" + key + ": " + str(value))
    logger.info(sampler.__repr__())
    sampler.write_to_h5(FOLDER_RES, "test2_Stochastic_Sampler.h5")

    logger.info("\nSensitivity analysis sampling:")
    sampler = SalibSamplingService(n_samples=10, sampler="latin")
    samples, stats = sampler.generate_samples(low=1, high=2, shape=(2,), stats=True)
    # for key, value in stats.iteritems():
    #     print("\n" + key + ": " + str(value))
    logger.info(sampler.__repr__())
    sampler.write_to_h5(FOLDER_RES, "test3_Stochastic_Sampler.h5")

    # logger.info("\nTesting distribution class and conversions...")
    # for distribution in AVAILABLE_DISTRIBUTIONS:
    #     logger.info("\nmean, std to distribution " + distribution + ":")
    #     if distribution == "poisson":
    #         mu = 0.25
    #         std = 0.5
    #     elif distribution == "beta":
    #         mu = 0.5
    #         std = 0.25
    #     elif distribution == "binomial":
    #         mu = 1.0
    #         std = 1.0 / np.sqrt(2)
    #     elif distribution == "chisquare":
    #         mu = 1.0
    #         std = np.sqrt(2 * mu)
    #     else:
    #         mu = 0.5
    #         std = 0.5
    #     logger.info(dict_str({"mean": mu, "std": std}))
    #     p = mean_std_to_distribution_params(distribution, mu=mu, std=std)
    #
    #     logger.info(str(p))
    #
    #     logger.info("\nDistribution " + distribution + " to mean, std:")
    #
    #     mu1, std1 = distribution_params_to_mean_std(distribution, **p)
    #
    #     logger.info(dict_str({"mean": mu, "std": std}))
    #
    #     if np.abs(mu - mu1) > 10 ** -6 or np.abs(std - std1) > 10 ** -6:
    #         raise ValueError("mean - mu1 = " + str(mu - mu1) + "std - std1 = " + str(std - std1))
    #
    #     sampler = StochasticSamplingService(n_samples=10)
    #     #                                   loc (mean), scale (sigma)
    #     samples, stats = sampler.generate_samples(1.5, 1.0, probability_distribution="norm", low=1, high=2, shape=(2,),
    #                                               stats=True)

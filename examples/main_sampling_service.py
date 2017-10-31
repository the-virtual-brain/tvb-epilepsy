import numpy as np
from tvb_epilepsy.base.configurations import FOLDER_RES
from tvb_epilepsy.base.utils.data_structures_utils import dict_str
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.service.sampling_service import StochasticSamplingService, DeterministicSamplingService
from tvb_epilepsy.base.computations.statistics_utils import distrib_dict, mean_std_to_distribution_params, \
    distribution_params_to_mean_std

logger = initialize_logger(__name__)

if __name__ == "__main__":
    logger.info("\nDeterministic linspace sampling:")

    sampler = DeterministicSamplingService(n_samples=10, n_outputs=2, low=1.0, high=2.0, grid_mode=True)

    samples, stats = sampler.generate_samples(stats=True)

    # for key, value in stats.iteritems():
    #
    #     print("\n" + key + ": " + str(value))

    logger.info(sampler.__repr__())
    sampler.write_to_h5(FOLDER_RES, "test_Stochastic_Sampler.h5")

    logger.info("\nStochastic uniform sampling:")

    sampler = StochasticSamplingService()  # (n_samples=10, n_outputs=1, low=1.0, high=2.0)

    samples, stats = sampler.generate_samples(stats=True)

    # for key, value in stats.iteritems():
    #     print("\n" + key + ": " + str(value))

    logger.info(sampler.__repr__())
    sampler.write_to_h5(FOLDER_RES, "test1_Stochastic_Sampler.h5")

    logger.info("\nStochastic truncated normal sampling:")

    sampler = StochasticSamplingService(n_samples=10, n_outputs=2, sampler="norm",
                                        trunc_limits={"low": 0.0, "high": +3.0}, loc=1.0,
                                        scale=2.0)

    samples, stats = sampler.generate_samples(stats=True)

    # for key, value in stats.iteritems():
    #     print("\n" + key + ": " + str(value))

    logger.info(sampler.__repr__())
    sampler.write_to_h5(FOLDER_RES, "test2_Stochastic_Sampler.h5")

    logger.info("\nSensitivity analysis sampling:")

    sampler = StochasticSamplingService(n_samples=10, n_outputs=2, sampler="latin", sampling_module="salib",
                                        bounds=[[0.0, 0.1], [0.0, 0.1]])

    samples, stats = sampler.generate_samples(stats=True)

    # for key, value in stats.iteritems():
    #     print("\n" + key + ": " + str(value))

    logger.info(sampler.__repr__())
    sampler.write_to_h5(FOLDER_RES, "test3_Stochastic_Sampler.h5")

    logger.info("\nTesting distribution conversions...")

    for distribution in distrib_dict:

        logger.info("\nmu, std to distribution " + distribution + ":")

        if distribution == "poisson":
            mu = 0.25
            std = 0.5
        elif distribution == "beta":
            mu = 0.5
            std = 0.25
        elif distribution == "binomial":
            mu = 1.0
            std = 1.0 / np.sqrt(2)
        elif distribution == "chisquare":
            mu = 1.0
            std = np.sqrt(2 * mu)
        else:
            mu = 0.5
            std = 0.5

        logger.info(dict_str({"mu": mu, "std": std}))

        p = mean_std_to_distribution_params(distribution, mu=mu, std=std)

        logger.info(str(p))

        logger.info("\nDistribution " + distribution + " to mu, std:")

        mu1, std1 = distribution_params_to_mean_std(distribution, **p)

        logger.info(dict_str({"mu": mu, "std": std}))

        if np.abs(mu - mu1) > 10 ** -6 or np.abs(std - std1) > 10 ** -6:
            raise ValueError("mu - mu1 = " + str(mu - mu1) + "std - std1 = " + str(std - std1))

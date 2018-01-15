import os

import numpy as np

from tvb_epilepsy.base.constants.configurations import DATA_CUSTOM, FOLDER_RES
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.io.writer_custom import CustomH5Writer
from tvb_epilepsy.service.sampling.stochastic_sampling_service import StochasticSamplingService
from tvb_epilepsy.scripts.pse_scripts import pse_from_hypothesis
from tvb_epilepsy.io.reader_custom import CustomH5Reader as Reader

logger = initialize_logger(__name__)

if __name__ == "__main__":
    # -------------------------------Reading data-----------------------------------

    data_folder = os.path.join(DATA_CUSTOM, 'Head')

    reader = Reader()
    writer = CustomH5Writer()

    head = reader.read_head(data_folder)

    # --------------------------Hypothesis definition-----------------------------------

    n_samples = 100

    # Sampling of the global coupling parameter
    stoch_sampler = StochasticSamplingService(n_samples=n_samples, random_seed=1000, )
    K_samples, K_sample_stats = stoch_sampler.generate_samples(parameter=(10.0, 3.0), shape=(1,), low=0.0, stats=True)

    #
    # Manual definition of hypothesis...:
    x0_indices = [20]
    x0_values = [0.9]
    e_indices = [70]
    e_values = [0.9]
    disease_indices = x0_indices + e_indices
    n_disease = len(disease_indices)

    n_x0 = len(x0_indices)
    n_e = len(e_indices)
    all_regions_indices = np.array(range(head.number_of_regions))
    healthy_indices = np.delete(all_regions_indices, disease_indices).tolist()
    n_healthy = len(healthy_indices)
    # This is an example of x0_values mixed Excitability and Epileptogenicity Hypothesis:
    hyp_x0_E = DiseaseHypothesis(head.connectivity.number_of_regions,
                                 excitability_hypothesis={tuple(x0_indices): x0_values},
                                 epileptogenicity_hypothesis={tuple(e_indices): e_values},
                                 connectivity_hypothesis={})

    # Now running the parameter search analysis:
    logger.info("running PSE LSA...")
    model_configuration, lsa_service, lsa_hypothesis, pse_results = pse_from_hypothesis(hyp_x0_E,
                                                                                        head.connectivity.normalized_weights,
                                                                                        head.connectivity.region_labels,
                                                                                        n_samples, param_range=0.1,
                                                                                        global_coupling=[{
                                                                                            "indices": all_regions_indices}],
                                                                                        healthy_regions_parameters=[
                                                                                            {"name": "x0_values",
                                                                                             "indices": healthy_indices}],
                                                                                        logger=logger,
                                                                                        save_services=True)[:4]

    lsa_service.plot_lsa(lsa_hypothesis, model_configuration, region_labels=head.connectivity.region_labels,
                         pse_results=pse_results)

    writer.write_dictionary(pse_results, os.path.join(FOLDER_RES, lsa_hypothesis.name + "_PSE_LSA_results.h5"))

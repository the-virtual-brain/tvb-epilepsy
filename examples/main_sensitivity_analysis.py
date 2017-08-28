import warnings
import numpy as np
import os
from tvb_epilepsy.base.configurations import DATA_CUSTOM, FOLDER_RES
from tvb_epilepsy.base.constants import K_DEF
from tvb_epilepsy.custom.readers_custom import CustomReader as Reader
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb.basic.logger.builder import get_logger
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.scripts.sensitivity_analysis_sripts import sensitivity_analysis_pse_from_hypothesis
from tvb_epilepsy.service.sensitivity_analysis_service import METHODS

logger = get_logger(__name__)

if __name__ == "__main__":
    # -------------------------------Reading data-----------------------------------

    data_folder = os.path.join(DATA_CUSTOM, 'Head')

    reader = Reader()

    head = reader.read_head(data_folder)

    # --------------------------Hypothesis definition-----------------------------------

    n_samples = 100

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
    all_regions_indices = np.array(range(head.connectivity.number_of_regions))
    healthy_indices = np.delete(all_regions_indices, disease_indices).tolist()
    n_healthy = len(healthy_indices)
    # This is an example of x0 mixed Excitability and Epileptogenicity Hypothesis:
    hyp_x0_E = DiseaseHypothesis(head.connectivity.number_of_regions,
                                 excitability_hypothesis={tuple(x0_indices): x0_values},
                                 epileptogenicity_hypothesis={tuple(e_indices): e_values},
                                 connectivity_hypothesis={})

    # Now running the sensitivity analysis:
    logger.info("running sensitivity analysis PSE LSA...")
    for m in METHODS:
        try:
            model_configuration_service, model_configuration, lsa_service, lsa_hypothesis, sa_results, pse_results = \
                sensitivity_analysis_pse_from_hypothesis(hyp_x0_E,
                                                         head.connectivity.normalized_weights,
                                                         head.connectivity.region_labels,
                                                         n_samples, method=m, half_range=0.1,
                                                         global_coupling=[{"indices": all_regions_indices,
                                                                           "bounds": [0.0, 2 * K_DEF]}],
                                                         healthy_regions_parameters=[
                                                             {"name": "x0", "indices": healthy_indices}],
                                                         logger=logger, save_services=True)

            lsa_service.plot_lsa(lsa_hypothesis, model_configuration, region_labels=head.connectivity.region_labels,
                                 pse_results=pse_results, title=m + "_PSE_LSA_overview_" + lsa_hypothesis.name)
            # , show_flag=True, save_flag=False

            convert_to_h5_model(pse_results).write_to_h5(FOLDER_RES,
                                                         m + "_PSE_LSA_results_" + lsa_hypothesis.name + ".h5")
            convert_to_h5_model(sa_results).write_to_h5(FOLDER_RES,
                                                        m + "_SA_LSA_results_" + lsa_hypothesis.name + ".h5")
        except:
            warnings.warn("Method " + m + " failed!")

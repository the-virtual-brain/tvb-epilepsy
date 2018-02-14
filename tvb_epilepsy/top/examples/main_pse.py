# coding=utf-8

import os
import numpy as np
from tvb_epilepsy.base.constants.configurations import IN_HEAD, FOLDER_RES
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.io.h5_writer import H5Writer
from tvb_epilepsy.plot.plotter import Plotter
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_epilepsy.top.scripts.pse_scripts import pse_from_hypothesis
from tvb_epilepsy.io.h5_reader import H5Reader as Reader

if __name__ == "__main__":
    # -------------------------------Reading data-----------------------------------
    reader = Reader()
    writer = H5Writer()
    head = reader.read_head(IN_HEAD)
    logger = initialize_logger(__name__)

    # --------------------------Manual Hypothesis definition-----------------------------------
    n_samples = 100
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
    hyp_x0_E = HypothesisBuilder().set_nr_of_regions(head.connectivity.number_of_regions
                                                     ).build_mixed_hypothesis(e_values, e_indices,
                                                                              x0_values, x0_indices)

    # Now running the parameter search analysis:
    logger.info("running PSE LSA...")
    model_config, lsa_service, lsa_hypothesis, pse_res = pse_from_hypothesis(hyp_x0_E,
                                                                             head.connectivity.normalized_weights,
                                                                             head.connectivity.region_labels,
                                                                             n_samples, param_range=0.1,
                                                                             global_coupling=[{
                                                                                 "indices": all_regions_indices}],
                                                                             healthy_regions_parameters=[
                                                                                 {"name": "x0_values",
                                                                                  "indices": healthy_indices}],
                                                                             save_services=True)[:4]

    logger.info("Plotting LSA...")
    Plotter().plot_lsa(lsa_hypothesis, model_config, lsa_service.weighted_eigenvector_sum,
                       lsa_service.eigen_vectors_number, region_labels=head.connectivity.region_labels,
                       pse_results=pse_res)

    logger.info("Saving LSA results ...")
    writer.write_dictionary(pse_res, os.path.join(FOLDER_RES, lsa_hypothesis.name + "_PSE_LSA_results.h5"))

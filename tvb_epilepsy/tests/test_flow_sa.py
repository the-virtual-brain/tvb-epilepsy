import os
import numpy
from tvb_epilepsy.base.constants.model_constants import K_DEF
from tvb_epilepsy.io.h5_reader import H5Reader
from tvb_epilepsy.io.h5_writer import H5Writer
from tvb_epilepsy.plot.plotter import Plotter
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_epilepsy.tests.base import BaseTest
from tvb_epilepsy.top.scripts.sensitivity_analysis_sripts import sensitivity_analysis_pse_from_hypothesis


class TestSAFlow(BaseTest):

    def test_sa_pse(self):
        reader = H5Reader()
        head = reader.read_head(self.config.input.HEAD)

        # --------------------------Hypothesis definition-----------------------------------
        n_samples = 100
        x0_indices = [20]
        x0_values = [0.9]
        e_indices = [70]
        e_values = [0.9]
        disease_indices = x0_indices + e_indices
        all_regions_indices = numpy.array(range(head.connectivity.number_of_regions))
        healthy_indices = numpy.delete(all_regions_indices, disease_indices).tolist()

        hyp_x0_E = HypothesisBuilder(self.config).set_nr_of_regions(
            head.connectivity.number_of_regions)._build_mixed_hypothesis(e_values, e_indices, x0_values, x0_indices)

        m = "sobol"
        model_configuration_builder, model_configuration, lsa_service, lsa_hypothesis, sa_results, pse_results = \
            sensitivity_analysis_pse_from_hypothesis(hyp_x0_E, head.connectivity.normalized_weights,
                                                     head.connectivity.region_labels, n_samples, method=m,
                                                     param_range=0.1,
                                                     global_coupling=[
                                                         {"indices": all_regions_indices, "low": 0.0,
                                                          "high": 2 * K_DEF}],
                                                     healthy_regions_parameters=[
                                                         {"name": "x0_values", "indices": healthy_indices}],
                                                     save_services=True, config=self.config)

        Plotter(self.config).plot_lsa(lsa_hypothesis, model_configuration, lsa_service.weighted_eigenvector_sum,
                                      lsa_service.eigen_vectors_number,
                                      region_labels=head.connectivity.region_labels,
                                      pse_results=pse_results,
                                      title=m + "_PSE_LSA_overview_" + lsa_hypothesis.name)

        pse_result_file = os.path.join(self.config.out.FOLDER_RES,
                                       m + "_PSE_LSA_results_" + lsa_hypothesis.name + ".h5")

        writer = H5Writer()
        writer.write_dictionary(pse_results, pse_result_file)

        sa_result_file = os.path.join(self.config.out.FOLDER_RES,
                                      m + "_SA_LSA_results_" + lsa_hypothesis.name + ".h5")
        writer.write_dictionary(sa_results, sa_result_file)

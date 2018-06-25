import os
import numpy
from tvb_infer.tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_infer.tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_infer.tvb_lsa.lsa_plotter import LSAPlotter
from tvb_infer.tests.base import BaseTest


class TestPlotter(BaseTest):
    lsa_plotter = LSAPlotter(BaseTest.config)

    def test_plot_lsa(self):
        figure_name = "LSAPlot"
        hypo_builder = HypothesisBuilder(config=self.config).set_name(figure_name)
        lsa_hypothesis = hypo_builder.build_lsa_hypothesis()
        mc = ModelConfigurationBuilder().build_model_from_E_hypothesis(lsa_hypothesis, numpy.array([1]))

        figure_file = os.path.join(self.config.out.FOLDER_FIGURES, figure_name + ".png")
        assert not os.path.exists(figure_file)

        self.lsa_plotter.plot_lsa(lsa_hypothesis, mc, True, None, region_labels=numpy.array(["a"]), title="")

        assert not os.path.exists(figure_file)

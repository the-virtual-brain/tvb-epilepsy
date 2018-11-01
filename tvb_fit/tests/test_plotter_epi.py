import os
import numpy

from tvb_fit.tests.base import BaseTest

from tvb_fit.tvb_epilepsy.base.model.timeseries import Timeseries, TimeseriesDimensions
from tvb_fit.tvb_epilepsy.service.simulator.epileptor_model_factory import build_EpileptorDP2D_from_model_config
from tvb_fit.tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_fit.tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_fit.tvb_epilepsy.plot.plotter import Plotter


class TestPlotter(BaseTest):
    plotter = Plotter(BaseTest.config)

    def test_plot_state_space(self):
        lsa_hypothesis = HypothesisBuilder(config=self.config).build_lsa_hypothesis()
        mc = ModelConfigurationBuilder("EpileptorDP", numpy.array([[1.0, 0.0], [0.0, 1.0]])). \
                                                        build_model_from_E_hypothesis(lsa_hypothesis)

        model = "6d"
        zmode = "lin"
        # TODO: this figure_name is constructed inside plot method, so it can change
        figure_name = "_" + "Epileptor_" + model + "_z-" + str(zmode)
        file_name = os.path.join(self.config.out.FOLDER_FIGURES, figure_name + ".png")
        assert not os.path.exists(file_name)

        self.plotter.plot_state_space(mc, region_labels=numpy.array(["a", "b"]), special_idx=[0],
                                    model=model, figure_name="")

        assert os.path.exists(file_name)

    def test_plot_lsa(self):
        figure_name = "LSAPlot"
        hypo_builder = HypothesisBuilder(config=self.config).set_name(figure_name)
        lsa_hypothesis = hypo_builder.build_lsa_hypothesis()
        mc = ModelConfigurationBuilder("EpileptorDP", numpy.array([[1.0, 0.0], [0.0, 1.0]])). \
            build_model_from_E_hypothesis(lsa_hypothesis)

        figure_file = os.path.join(self.config.out.FOLDER_FIGURES, figure_name + ".png")
        assert not os.path.exists(figure_file)

        self.plotter.plot_lsa(lsa_hypothesis, mc, True, None, region_labels=numpy.array(["a" "b"]), title="")

        assert not os.path.exists(figure_file)

    def test_plot_sim_results(self):
        lsa_hypothesis = HypothesisBuilder(config=self.config).build_lsa_hypothesis()
        mc = ModelConfigurationBuilder("EpileptorDP", numpy.array([[1.0, 0.0], [0.0, 1.0]])). \
            build_model_from_E_hypothesis(lsa_hypothesis)
        model = build_EpileptorDP2D_from_model_config(mc)

        # TODO: this figure_name is constructed inside plot method, so it can change
        figure_name = "Simulated_TAVG"
        file_name = os.path.join(self.config.out.FOLDER_FIGURES, figure_name + ".png")
        assert not os.path.exists(file_name)

        data_3D = numpy.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]],
                               [[3, 4, 5], [6, 7, 8], [9, 0, 1], [2, 3, 4]],
                               [[5, 6, 7], [8, 9, 0], [1, 2, 3], [4, 5, 6]]])

        self.plotter.plot_simulated_timeseries(
            Timeseries(data_3D, {TimeseriesDimensions.SPACE.value: numpy.array(["r1", "r2", "r3", "r4"]),
                                 TimeseriesDimensions.VARIABLES.value: numpy.array(["x1", "x2", "z"])}, 0, 1),
            model, [0])

        assert os.path.exists(file_name)

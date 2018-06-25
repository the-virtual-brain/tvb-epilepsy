import os
import numpy
from tvb_infer.base.model.timeseries import Timeseries, TimeseriesDimensions
from tvb_infer.tvb_epilepsy.plot.plotter import Plotter
from tvb_infer.tvb_epilepsy.service.simulator.epileptor_model_factory import build_EpileptorDP2D
from tvb_infer.tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_infer.tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_infer.tests.base import BaseTest


class TestPlotter(BaseTest):
    plotter = Plotter(BaseTest.config)

    def test_plot_state_space(self):
        lsa_hypothesis = HypothesisBuilder(config=self.config).build_lsa_hypothesis()
        mc = ModelConfigurationBuilder().build_model_from_E_hypothesis(lsa_hypothesis, numpy.array([1]))

        model = "6d"
        zmode = "lin"
        # TODO: this figure_name is constructed inside plot method, so it can change
        figure_name = "_" + "Epileptor_" + model + "_z-" + str(zmode)
        file_name = os.path.join(self.config.out.FOLDER_FIGURES, figure_name + ".png")
        assert not os.path.exists(file_name)

        self.plotter.plot_state_space(mc, region_labels=numpy.array(["a"]), special_idx=[0],
                                    model=model, zmode=zmode,figure_name="")

        assert os.path.exists(file_name)

    def test_plot_sim_results(self):
        lsa_hypothesis = HypothesisBuilder(config=self.config).build_lsa_hypothesis()
        mc = ModelConfigurationBuilder().build_model_from_E_hypothesis(lsa_hypothesis, numpy.array([1]))
        model = build_EpileptorDP2D(mc)

        # TODO: this figure_name is constructed inside plot method, so it can change
        figure_name = "Simulated_TAVG"
        file_name = os.path.join(self.config.out.FOLDER_FIGURES, figure_name + ".png")
        assert not os.path.exists(file_name)

        data_3D = numpy.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 1, 2]],
                               [[3, 4, 5], [6, 7, 8], [9, 0, 1], [2, 3, 4]],
                               [[5, 6, 7], [8, 9, 0], [1, 2, 3], [4, 5, 6]]])

        self.plotter.plot_simulated_timeseries(
            Timeseries(data_3D, {TimeseriesDimensions.SPACE.value: ["r1", "r2", "r3", "r4"],
                                 TimeseriesDimensions.VARIABLES.value: ["x1", "x2", "z"]}, 0, 1),
            model, [0])

        assert os.path.exists(file_name)

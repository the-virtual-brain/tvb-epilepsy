import os
import numpy
from tvb_epilepsy.plot.plotter import Plotter
from tvb_epilepsy.service.epileptor_model_factory import build_EpileptorDP2D, VOIS
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_epilepsy.service.stochastic_parameter_builder import set_parameter
from tvb_epilepsy.top.scripts.simulation_scripts import prepare_vois_ts_dict
from tvb_epilepsy.tests.base import BaseTest


class TestPlotter(BaseTest):
    plotter = Plotter(BaseTest.config)

    def test_plot_head(self):
        head = self._prepare_dummy_head()
        # TODO: this filenames may change because they are composed inside the plotting functions
        filename1 = "Connectivity_.png"
        filename2 = "HeadStats.png"
        filename3 = "1_-_SEEG_-_Projection.png"

        assert not os.path.exists(os.path.join(self.config.out.FOLDER_FIGURES, filename1))
        assert not os.path.exists(os.path.join(self.config.out.FOLDER_FIGURES, filename2))
        assert not os.path.exists(os.path.join(self.config.out.FOLDER_FIGURES, filename3))

        self.plotter.plot_head(head)

        assert os.path.exists(os.path.join(self.config.out.FOLDER_FIGURES, filename1))
        assert os.path.exists(os.path.join(self.config.out.FOLDER_FIGURES, filename2))
        assert os.path.exists(os.path.join(self.config.out.FOLDER_FIGURES, filename3))

    def test_plot_stochastic_parameter(self):
        K_mean = 10 * 2.5 / 87
        K_std = numpy.min([K_mean - 0.0, 3.0 - K_mean]) / 6.0
        K = set_parameter("K", optimize_pdf=True, use="manual", K_lo=0.0, K_hi=3.0, K_pdf="lognormal",
                          K_pdf_params={"skew": 0.0, "mean": K_mean / K_std}, K_mean=K_mean,
                          K_std=K_std)
        figure_name = "K_parameter"
        figure_file = os.path.join(self.config.out.FOLDER_FIGURES, figure_name + ".png")
        assert not os.path.exists(figure_file)

        self.plotter.plot_stochastic_parameter(K, figure_name=figure_name)

        assert os.path.exists(figure_file)

    def test_plot_lsa(self):
        figure_name = "LSAPlot"
        hypo_builder = HypothesisBuilder(config=self.config).set_name(figure_name)
        lsa_hypothesis = hypo_builder.build_lsa_hypothesis()
        mc = ModelConfigurationBuilder().build_model_from_E_hypothesis(lsa_hypothesis, numpy.array([1]))

        figure_file = os.path.join(self.config.out.FOLDER_FIGURES, figure_name + ".png")
        assert not os.path.exists(figure_file)

        self.plotter.plot_lsa(lsa_hypothesis, mc, True, None, region_labels=numpy.array(["a"]), title="")

        assert not os.path.exists(figure_file)

    def test_plot_state_space(self):
        lsa_hypothesis = HypothesisBuilder(config=self.config).build_lsa_hypothesis()
        mc = ModelConfigurationBuilder().build_model_from_E_hypothesis(lsa_hypothesis, numpy.array([1]))

        model = "6d"
        zmode = "lin"
        # TODO: this figure_name is constructed inside plot method, so it can change
        figure_name = "_" + "Epileptor_" + model + "_z-" + str(zmode)
        file_name = os.path.join(self.config.out.FOLDER_FIGURES, figure_name + ".png")
        assert not os.path.exists(file_name)

        self.plotter.plot_state_space(mc, region_labels=numpy.array(["a"]), special_idx=[0], model=model, zmode=zmode,
                                      figure_name="")

        assert os.path.exists(file_name)

    def test_plot_sim_results(self):
        lsa_hypothesis = HypothesisBuilder(config=self.config).build_lsa_hypothesis()
        mc = ModelConfigurationBuilder().build_model_from_E_hypothesis(lsa_hypothesis, numpy.array([1]))
        model = build_EpileptorDP2D(mc)
        res = prepare_vois_ts_dict(VOIS["EpileptorDP2D"], numpy.array([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]]))
        res['time'] = numpy.array([1, 2, 3])
        res['time_units'] = 'msec'

        # TODO: this figure_name is constructed inside plot method, so it can change
        figure_name = "EpileptorDP2D_Simulated_TAVG"
        file_name = os.path.join(self.config.out.FOLDER_FIGURES, figure_name + ".png")
        assert not os.path.exists(file_name)

        self.plotter.plot_sim_results(model, [0], res)

        assert os.path.exists(file_name)

import os
import numpy
from tvb_epilepsy.plot.plotter import Plotter
from tvb_epilepsy.io.h5_reader import H5Reader
from tvb_epilepsy.base.model.vep.head import Head
from tvb_epilepsy.base.model.vep.sensors import Sensors
from tvb_epilepsy.base.model.vep.surface import Surface
from tvb_epilepsy.base.constants.configurations import FOLDER_FIGURES, DATA_TEST, FOLDER_LOGS, FOLDER_RES
from tvb_epilepsy.service.hypothesis_builder import HypothesisBuilder
from tvb_epilepsy.service.model_configuration_builder import ModelConfigurationBuilder
from tvb_epilepsy.service.stochastic_parameter_builder import set_parameter

head_dir = "head2"


class TestPlotter():
    plotter = Plotter()

    @classmethod
    def setup_class(cls):
        for direc in (FOLDER_LOGS, FOLDER_RES, FOLDER_FIGURES):
            if not os.path.exists(direc):
                os.makedirs(direc)

    def _prepare_dummy_head(self):
        reader = H5Reader()
        connectivity = reader.read_connectivity(os.path.join(DATA_TEST, head_dir, "Connectivity.h5"))
        cort_surface = Surface([], [])
        seeg_sensors = Sensors(numpy.array(["sens1", "sens2"]), numpy.array([[0, 0, 0], [0, 1, 0]]))
        head = Head(connectivity, cort_surface, sensorsSEEG=seeg_sensors)

        return head

    def test_plot_head(self):
        head = self._prepare_dummy_head()
        # TODO: this filenames may change because they are composed inside the plotting functions
        filename1 = "Connectivity_.png"
        filename2 = "HeadStats.png"
        filename3 = "1_-_SEEG_-_Projection.png"

        assert not os.path.exists(os.path.join(FOLDER_FIGURES, filename1))
        assert not os.path.exists(os.path.join(FOLDER_FIGURES, filename2))
        assert not os.path.exists(os.path.join(FOLDER_FIGURES, filename3))

        self.plotter.plot_head(head, figure_dir=FOLDER_FIGURES)

        assert os.path.exists(os.path.join(FOLDER_FIGURES, filename1))
        assert os.path.exists(os.path.join(FOLDER_FIGURES, filename2))
        assert os.path.exists(os.path.join(FOLDER_FIGURES, filename3))

    def test_plot_stochastic_parameter(self):
        K_mean = 10 * 2.5 / 87
        K_std = numpy.min([K_mean - 0.0, 3.0 - K_mean]) / 6.0
        K = set_parameter("K", optimize_pdf=True, use="manual", K_lo=0.0, K_hi=3.0, K_pdf="lognormal",
                          K_pdf_params={"skew": 0.0, "mean": K_mean / K_std}, K_mean=K_mean,
                          K_std=K_std)
        figure_name = "K_parameter"
        figure_file = os.path.join(FOLDER_FIGURES, figure_name + ".png")
        assert not os.path.exists(figure_file)

        self.plotter.plot_stochastic_parameter(K, figure_name=figure_name)

        assert os.path.exists(figure_file)

    def test_plot_lsa(self):
        figure_name = "LSAPlot"
        hypo_builder = HypothesisBuilder().set_name(figure_name)
        lsa_hypothesis = hypo_builder.build_lsa_hypothesis()
        mc = ModelConfigurationBuilder().build_model_from_E_hypothesis(lsa_hypothesis, numpy.array([1]))

        figure_file = os.path.join(FOLDER_FIGURES, figure_name + ".png")
        assert not os.path.exists(figure_file)

        self.plotter.plot_lsa(lsa_hypothesis, mc, True, None, region_labels=numpy.array(["a"]), title="")

        assert not os.path.exists(figure_file)

    def test_plot_state_space(self):
        lsa_hypothesis = HypothesisBuilder().build_lsa_hypothesis()
        mc = ModelConfigurationBuilder().build_model_from_E_hypothesis(lsa_hypothesis, numpy.array([1]))

        model = "6d"
        zmode = "lin"
        # TODO: this figure_name is constructed inside plot method, so it can change
        figure_name = "_" + "Epileptor_" + model + "_z-" + str(zmode)
        file_name = os.path.join(FOLDER_FIGURES, figure_name + ".png")
        assert not os.path.exists(file_name)

        self.plotter.plot_state_space(mc, region_labels=numpy.array(["a"]), special_idx=[0], model=model, zmode=zmode,
                                      figure_name="")

        assert os.path.exists(file_name)

    @classmethod
    def teardown_class(cls):
        for direc in (FOLDER_LOGS, FOLDER_RES, FOLDER_FIGURES):
            for dir_file in os.listdir(direc):
                os.remove(os.path.join(os.path.abspath(direc), dir_file))
            os.removedirs(direc)

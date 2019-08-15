import os

from tvb_fit.tests.base import BaseTest
from tvb_fit.plot.plotter import Plotter


class TestPlotter(BaseTest):
    plotter = Plotter(BaseTest.config)

    def test_plot_head(self):
        head = self._prepare_dummy_head()
        # TODO: this filenames may change because they are composed inside the plotting functions
        filename1 = "Connectivity_.png"
        filename2 = "HeadStats.png"
        filename3 = "1-SEEG-Projection.png"

        assert not os.path.exists(os.path.join(self.config.out.FOLDER_FIGURES, filename1))
        assert not os.path.exists(os.path.join(self.config.out.FOLDER_FIGURES, filename2))
        assert not os.path.exists(os.path.join(self.config.out.FOLDER_FIGURES, filename3))

        self.plotter.plot_head(head)

        assert os.path.exists(os.path.join(self.config.out.FOLDER_FIGURES, filename1))
        assert os.path.exists(os.path.join(self.config.out.FOLDER_FIGURES, filename2))
        # Because there is no gain matrix
        assert not os.path.exists(os.path.join(self.config.out.FOLDER_FIGURES, filename3))

    #TODO: check TypeError: unique() got an unexpected keyword argument 'axis' in prepare_target_stats()
    # def test_plot_probabilistic_parameter(self):
    #     K_mean = 10 * 2.5 / 87
    #     K_std = numpy.min([K_mean - 0.0, 3.0 - K_mean]) / 6.0
    #     K = set_parameter("K", optimize_pdf=True, use="manual", K_lo=0.0, K_hi=3.0, K_pdf="lognormal",
    #                       K_pdf_params={"skew": 0.0, "mean": K_mean / K_std}, K_mean=K_mean,
    #                       K_std=K_std)
    #     figure_name = "K_parameter"
    #     figure_file = os.path.join(self.config.out.FOLDER_FIGURES, figure_name + ".png")
    #     assert not os.path.exists(figure_file)
    #
    #     self.plotter.plot_probabilistic_parameter(K, figure_name=figure_name)
    #
    #     assert os.path.exists(figure_file)

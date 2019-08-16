from tvb_fit.tvb_epilepsy.base.constants.config import FiguresConfig
import matplotlib
matplotlib.use(FiguresConfig().MATPLOTLIB_BACKEND)
from matplotlib import pyplot

import numpy

from tvb_fit.tvb_epilepsy.base.constants.model_constants import X1EQ_CR_DEF, X1_DEF, X0_CR_DEF, X0_DEF
from tvb_fit.tvb_epilepsy.base.computation_utils.calculations_utils import calc_fz, calc_fx1, calc_fx1_2d_taylor, \
                                                             calc_x0_val_to_model_x0
from tvb_fit.tvb_epilepsy.base.computation_utils.equilibrium_computation import calc_eq_y1, def_x1lin

from tvb_utils.data_structures_utils import isequal_string, generate_region_labels
from tvb_plot.base_plotter import BasePlotter


class ModelConfigPlotter(BasePlotter):

    def __init__(self, config=None):
        super(ModelConfigPlotter, self).__init__(config)
        self.HighlightingDataCursor = lambda *args, **kwargs: None
        if matplotlib.get_backend() in matplotlib.rcsetup.interactive_bk and self.config.figures.MOUSE_HOOVER:
            try:
                from mpldatacursor import HighlightingDataCursor
                self.HighlightingDataCursor = HighlightingDataCursor
            except ImportError:
                self.config.figures.MOUSE_HOOVER = False
                self.logger.warning("Importing mpldatacursor failed! No highlighting functionality in plots!")
        else:
            self.logger.warning("Noninteractive matplotlib backend! No highlighting functionality in plots!")
            self.config.figures.MOUSE_HOOVER = False

    # TODO: refactor to not have the plot commands here, although this is a very hard and special case...
    def plot_state_space(self, model_config, region_labels=[], special_idx=[],
                         figure_name="", approximations=False):
        if model_config.model_name == "Epileptor2D":
            model = "2d"
        else:
            model = "6d"
        add_name = " " + "Epileptor " + model + " z-" + str(numpy.where(model_config.zmode[0], "exp", "lin"))
        figure_name = figure_name + add_name

        region_labels = generate_region_labels(model_config.number_of_regions, region_labels, ". ")
        # n_region_labels = len(region_labels)
        # if n_region_labels == model_config.number_of_regions:
        #     region_labels = numpy.array(["%d. %s" % l for l in zip(range(model_config.number_of_regions), region_labels)])
        # else:
        #     region_labels = numpy.array(["%d" % l for l in range(model_config.number_of_regions)])

        # Fixed parameters for all regions:
        x1eq = model_config.x1eq
        zeq = model_config.zeq
        x0 = a = b = d = yc = slope = Iext1 = Iext2 = s = tau1 = tau0 = zmode = 0.0
        for p in ["x0", "a", "b", "d", "yc", "slope", "Iext1", "Iext2", "s", "tau1", "tau0", "zmode"]:
            exec (p + " = numpy.mean(model_config." + p + ")")

        fig = pyplot.figure(figure_name, figsize=FiguresConfig.SMALL_SIZE)

        # Lines:
        x1 = numpy.linspace(-2.0, 1.0, 100)
        if isequal_string(model, "2d"):
            y1 = yc
        else:
            y1 = calc_eq_y1(x1, yc, d=d)
        # x1 nullcline:
        zX1 = calc_fx1(x1, z=0, y1=y1, Iext1=Iext1, slope=slope, a=a, b=b, d=d, tau1=1.0, x1_neg=True, model=model,
                       x2=0.0)  # yc + Iext1 - x1 ** 3 - 2.0 * x1 ** 2
        x1null, = pyplot.plot(x1, zX1, 'b-', label='x1 nullcline', linewidth=1)
        ax = pyplot.gca()
        # z nullcines
        # center point (critical equilibrium point) without approximation:
        # zsq0 = yc + Iext1 - x1sq0 ** 3 - 2.0 * x1sq0 ** 2
        x0e = calc_x0_val_to_model_x0(X0_CR_DEF, yc, Iext1, a=a, b=b, d=d, zmode=model_config.zmode)
        x0ne = calc_x0_val_to_model_x0(X0_DEF, yc, Iext1, a=a, b=b, d=d, zmode=model_config.zmode)
        zZe = calc_fz(x1, z=0.0, x0=x0e, tau1=1.0, tau0=1.0, zmode=model_config.zmode)  # for epileptogenic regions
        zZne = calc_fz(x1, z=0.0, x0=x0ne, tau1=1.0, tau0=1.0, zmode=model_config.zmode)  # for non-epileptogenic regions
        zE1null, = pyplot.plot(x1, zZe, 'g-', label='z nullcline at critical point (e_values=1)', linewidth=1)
        zE2null, = pyplot.plot(x1, zZne, 'g--', label='z nullcline for e_values=0', linewidth=1)
        if approximations:
            # The point of the linear approximation (1st order Taylor expansion)
            x1LIN = def_x1lin(X1_DEF, X1EQ_CR_DEF, len(region_labels))
            x1SQ = X1EQ_CR_DEF
            x1lin0 = numpy.mean(x1LIN)
            # The point of the square (parabolic) approximation (2nd order Taylor expansion)
            x1sq0 = numpy.mean(x1SQ)
            # approximations:
            # linear:
            x1lin = numpy.linspace(-5.5 / 3.0, -3.5 / 3, 30)
            # x1 nullcline after linear approximation:
            # yc + Iext1 + 2.0 * x1lin0 ** 3 + 2.0 * x1lin0 ** 2 - \
            # (3.0 * x1lin0 ** 2 + 4.0 * x1lin0) * x1lin  # x1
            zX1lin = calc_fx1_2d_taylor(x1lin, x1lin0, z=0, y1=yc, Iext1=Iext1, slope=slope, a=a, b=b, d=d, tau1=1.0,
                                        x1_neg=None, order=2)  #
            # center point without approximation:
            # zlin0 = yc + Iext1 - x1lin0 ** 3 - 2.0 * x1lin0 ** 2
            # square:
            x1sq = numpy.linspace(-5.0 / 3, -1.0, 30)
            # x1 nullcline after parabolic approximation:
            # + 2.0 * x1sq ** 2 + 16.0 * x1sq / 3.0 + yc + Iext1 + 64.0 / 27.0
            zX1sq = calc_fx1_2d_taylor(x1sq, x1sq0, z=0, y1=yc, Iext1=Iext1, slope=slope, a=a, b=b, d=d, tau1=1.0,
                                       x1_neg=None, order=3, shape=x1sq.shape)
            sq, = pyplot.plot(x1sq, zX1sq, 'm--', label='Parabolic local approximation', linewidth=2)
            lin, = pyplot.plot(x1lin, zX1lin, 'c--', label='Linear local approximation', linewidth=2)
            pyplot.legend(handles=[x1null, zE1null, zE2null, lin, sq])
        else:
            pyplot.legend(handles=[x1null, zE1null, zE2null])

        # Points:
        ii = range(len(region_labels))
        n_special_idx = len(special_idx)
        if n_special_idx > 0:
            ii = numpy.delete(ii, special_idx)
        points = []
        for i in ii:
            point = pyplot.text(x1eq[i], zeq[i], str(i), fontsize=10, color='k', alpha=0.3,
                                 label=str(i) + '.' + region_labels[i])
            # point, = pyplot.plot(x1eq[i], zeq[i], '*', mfc='k', mec='k',
            #                      ms=10, alpha=0.3, label=str(i) + '.' + region_labels[i])
            points.append(point)
        if n_special_idx > 0:
            for i in special_idx:
                point = pyplot.text(x1eq[i], zeq[i], str(i), fontsize=10, color='r', alpha=0.8,
                                     label=str(i) + '.' + region_labels[i])
                # point, = pyplot.plot(x1eq[i], zeq[i], '*', mfc='r', mec='r', ms=10, alpha=0.8,
                #                      label=str(i) + '.' + region_labels[i])
                points.append(point)
        # ax.plot(x1lin0, zlin0, '*', mfc='r', mec='r', ms=10)
        # ax.axes.text(x1lin0 - 0.1, zlin0 + 0.2, 'e_values=0.0', fontsize=10, color='r')
        # ax.plot(x1sq0, zsq0, '*', mfc='m', mec='m', ms=10)
        # ax.axes.text(x1sq0, zsq0 - 0.2, 'e_values=1.0', fontsize=10, color='m')

        # Vector field
        X1, Z = numpy.meshgrid(numpy.linspace(-2.0, 1.0, 41), numpy.linspace(0.0, 6.0, 31), indexing='ij')
        if isequal_string(model, "2d"):
            y1 = yc
            x2 = 0.0
        else:
            y1 = calc_eq_y1(X1, yc, d=d)
            x2 = 0.0  # as a simplification for faster computation without important consequences
            # x2 = calc_eq_x2(Iext2, y2eq=None, zeq=X1, x1eq=Z, s=s)[0]
        fx1 = calc_fx1(X1, Z, y1=y1, Iext1=Iext1, slope=slope, a=a, b=b, d=d, tau1=tau1, x1_neg=None,
                       model=model, x2=x2)
        fz = calc_fz(X1, Z, x0=x0, tau1=tau1, tau0=tau0, zmode=zmode)
        C = numpy.abs(fx1) + numpy.abs(fz)
        pyplot.quiver(X1, Z, fx1, fz, C, edgecolor='k', alpha=.5, linewidth=.5)
        pyplot.contour(X1, Z, fx1, 0, colors='b', linestyles="dashed")

        ax.set_title("Epileptor states pace at the x1-z phase plane of the" + add_name)
        ax.axes.autoscale(tight=True)
        ax.axes.set_ylim([0.0, 6.0])
        ax.axes.set_xlabel('x1')
        ax.axes.set_ylabel('z')

        if self.config.figures.MOUSE_HOOVER:
            self.HighlightingDataCursor(points[0], formatter='{label}'.format, bbox=dict(fc='white'),
                                        arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5))

        if len(fig.get_label()) == 0:
            fig.set_label(figure_name)
        else:
            figure_name = fig.get_label().replace(": ", "_").replace(" ", "_").replace("\t", "_")

        self._save_figure(None, figure_name)
        self._check_show()
        return fig

    def plot_model_configuration(self, model_configuration, number_of_regions=None, regions_labels=[], x0_indices=[],
                                 e_indices=[], disease_indices=[], title="Model Configuration Overview", figure_name='',
                                 figsize=FiguresConfig.VERY_LARGE_SIZE):
        if number_of_regions is None:
            number_of_regions = len(model_configuration.x0_values)
        if not regions_labels:
            regions_labels = numpy.array([str(ii) for ii in range(number_of_regions)])
        disease_indices = numpy.unique(numpy.concatenate((x0_indices, e_indices, disease_indices), axis=0)).tolist()
        plot_dict_list = model_configuration.prepare_for_plot(x0_indices, e_indices, disease_indices)
        return self.plot_in_columns(plot_dict_list, regions_labels, width_ratios=[],
                                    left_ax_focus_indices=disease_indices, right_ax_focus_indices=disease_indices,
                                    title=title, figure_name=figure_name, figsize=figsize)
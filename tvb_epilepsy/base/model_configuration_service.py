# coding=utf-8
"""
Service to do X0/E Hypothesis configuration.
"""
import numpy
from matplotlib import pyplot
from mpldatacursor import HighlightingDataCursor
from tvb.basic.logger.builder import get_logger
from tvb_epilepsy.base.constants import X1_EQ_CR_DEF, E_DEF, X0_DEF, K_DEF, YC_DEF, I_EXT1_DEF, A_DEF, B_DEF, X1_DEF, \
    X0_CR_DEF, FIG_SIZE, SAVE_FLAG, SHOW_FLAG, FOLDER_FIGURES, FIG_FORMAT, MOUSEHOOVER
from tvb_epilepsy.base.plot_utils import save_figure, check_show
from tvb_epilepsy.base.utils import formal_repr
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.calculations_utils import calc_x0cr_r, calc_coupling, calc_x0, calc_fx1, calc_fx1_2d_taylor, \
    calc_fz, calc_rescaled_x0
from tvb_epilepsy.base.equilibrium_computation import calc_eq_z_2d, eq_x1_hypo_x0_linTaylor, eq_x1_hypo_x0_optimize, \
    def_x1lin, calc_eq_y1
from tvb_epilepsy.base.model_configuration import ModelConfiguration

# NOTES:
# In the future all the related to model configuration parameters might be part of the disease hypothesis:
# yc=YC_DEF, Iext1=I_EXT1_DEF, K=K_DEF, a=A_DEF, b=B_DEF
# For now, we assume default values, or externally set

LOG = get_logger(__name__)


class ModelConfigurationService(object):
    x1EQcr = X1_EQ_CR_DEF

    def __init__(self, number_of_regions, x0=X0_DEF, yc=YC_DEF, Iext1=I_EXT1_DEF, K=K_DEF, a=A_DEF, b=B_DEF, E=E_DEF,
                 x1eq_mode="optimize"):
        self.number_of_regions = number_of_regions
        self.x0 = x0 * numpy.ones((self.number_of_regions,), dtype=numpy.float32)
        self.yc = yc
        self.Iext1 = Iext1
        self.a = a
        self.b = b
        self.x1eq_mode = x1eq_mode
        self.K_unscaled = K * numpy.ones((self.number_of_regions,), dtype=numpy.float32)
        self.K = None
        self._normalize_global_coupling()
        self.E = E * numpy.ones((self.number_of_regions,), dtype=numpy.float32)

    def __repr__(self):
        d = {"01. Number of regions": self.number_of_regions,
             "02. x0": self.x0,
             "03. Iext1": self.Iext1,
             "04. a": self.a,
             "05. b": self.b,
             "06. x1eq_mode": self.x1eq_mode,
             "07. K_unscaled": self.K_unscaled,
             "08. K": self.K,
             "09. E": self.E,
             }
        return formal_repr(self, d)

    def __str__(self):
        return self.__repr__()

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

    def _ensure_equilibrum(self, x1EQ, zEQ):
        temp = x1EQ > self.x1EQcr - 10 ** (-3)
        if temp.any():
            x1EQ[temp] = self.x1EQcr - 10 ** (-3)
            zEQ = self._compute_z_equilibrium(x1EQ)

        return x1EQ, zEQ

    def _compute_x1_equilibrium_from_E(self, e_values):
        array_ones = numpy.ones((self.number_of_regions,), dtype=numpy.float32)
        return ((e_values - 5.0) / 3.0) * array_ones

    def _compute_z_equilibrium(self, x1EQ):
        return calc_eq_z_2d(x1EQ, self.yc, self.Iext1)

    def _compute_critical_x0_scaling(self):
        return calc_x0cr_r(self.yc, self.Iext1, a=self.a, b=self.b)

    def _compute_coupling_at_equilibrium(self, x1EQ, connectivity_matrix):
        return calc_coupling(x1EQ, self.K, connectivity_matrix)

    def _compute_x0(self, x1EQ, zEQ, x0cr, rx0, connectivity_matrix):
        return calc_x0(x1EQ, zEQ, self.K, connectivity_matrix, x0cr, rx0)

    def _compute_e_values(self, x1EQ):
        return 3.0 * x1EQ + 5.0

    def _compute_params_after_equilibration(self, x1EQ, zEQ, connectivity_matrix):
        (x0cr, rx0) = self._compute_critical_x0_scaling()
        Ceq = self._compute_coupling_at_equilibrium(x1EQ, connectivity_matrix)
        x0_values = self._compute_x0(x1EQ, zEQ, x0cr, rx0, connectivity_matrix)
        e_values = self._compute_e_values(x1EQ)
        return x0cr, rx0, Ceq, x0_values, e_values

    def _compute_x1_and_z_equilibrium_from_E(self, e_values):
        x1EQ = self._compute_x1_equilibrium_from_E(e_values)
        zEQ = self._compute_z_equilibrium(x1EQ)
        return x1EQ, zEQ

    def _compute_x1_equilibrium(self, e_indices, x1EQ, zEQ, x0_values, x0cr, rx0, connectivity_matrix):
        x0_indices = numpy.delete(numpy.array(range(connectivity_matrix.shape[0])), e_indices)
        if self.x1eq_mode == "linTaylor":
            x1EQ = \
                eq_x1_hypo_x0_linTaylor(x0_indices, e_indices, x1EQ, zEQ, x0_values, x0cr, rx0,
                                        self.yc, self.Iext1, self.K, connectivity_matrix)[0]
        else:
            x1EQ = \
                eq_x1_hypo_x0_optimize(x0_indices, e_indices, x1EQ, zEQ, x0_values, x0cr, rx0,
                                       self.yc, self.Iext1, self.K, connectivity_matrix)[0]
        return x1EQ

    def _normalize_global_coupling(self):
        self.K = self.K_unscaled / self.number_of_regions

    def configure_model_from_equilibrium(self, x1EQ, zEQ, connectivity_matrix):
        x1EQ, zEQ = self._ensure_equilibrum(x1EQ, zEQ)
        x0cr, rx0, Ceq, x0_values, e_values = self._compute_params_after_equilibration(x1EQ, zEQ, connectivity_matrix)
        model_configuration = ModelConfiguration(self.yc, self.Iext1, self.K, self.a, self.b,
                                                 x0cr, rx0, x1EQ, zEQ, Ceq, x0_values, e_values, connectivity_matrix)
        return model_configuration

    def configure_model_from_E_hypothesis(self, disease_hypothesis):
        # Always normalize K first
        self._normalize_global_coupling()

        # Then apply connectivity disease hypothesis scaling if any:
        connectivity_matrix = disease_hypothesis.get_weights()
        if len(disease_hypothesis.w_indices) > 0:
            connectivity_matrix *= disease_hypothesis.get_connectivity_disease()

        # All nodes except for the diseased ones will get the default epileptogenicity:
        e_values = numpy.array(self.E)
        e_values[disease_hypothesis.e_indices] = disease_hypothesis.e_values

        # Compute equilibrium from epileptogenicity:
        x1EQ, zEQ = self._compute_x1_and_z_equilibrium_from_E(e_values)
        x1EQ, zEQ = self._ensure_equilibrum(x1EQ, zEQ)

        return self.configure_model_from_equilibrium(x1EQ, zEQ, connectivity_matrix)

    def configure_model_from_hypothesis(self, disease_hypothesis):
        # Always normalize K first
        self._normalize_global_coupling()

        # Then apply connectivity disease hypothesis scaling if any:
        connectivity_matrix = disease_hypothesis.get_weights()
        if len(disease_hypothesis.w_indices) > 0:
            connectivity_matrix *= disease_hypothesis.get_connectivity_disease()

        # We assume that all nodes have the default (healthy) excitability:
        x0_values = numpy.array(self.x0)
        # ...and some  excitability-diseased ones:
        x0_values[disease_hypothesis.x0_indices] = disease_hypothesis.x0_values
        # x0 values must have size of len(x0_indices):
        x0_values = numpy.delete(x0_values, disease_hypothesis.e_indices)

        # There might be some epileptogenicity-diseased regions as well:
        # Initialize with the default E
        e_values = numpy.array(self.E)
        # and assign any diseased E_values if any
        e_values[disease_hypothesis.e_indices] = disease_hypothesis.e_values

        # Compute equilibrium from epileptogenicity:
        x1EQ_temp, zEQ_temp = self._compute_x1_and_z_equilibrium_from_E(e_values)

        (x0cr, rx0) = self._compute_critical_x0_scaling()

        # Now, solve the system in order to compute equilibrium:
        x1EQ = self._compute_x1_equilibrium(disease_hypothesis.e_indices, x1EQ_temp, zEQ_temp, x0_values, x0cr, rx0,
                                            connectivity_matrix)
        zEQ = self._compute_z_equilibrium(x1EQ)

        return self.configure_model_from_equilibrium(x1EQ, zEQ, connectivity_matrix)

    def plot_nullclines(self, model_config, region_labels, special_idx, model, zmode, figure_name):
        add_name = " " + "Epileptor " + model + " z-" + str(zmode)
        figure_name = figure_name + add_name

        # Fixed parameters for all regions:
        x1eq = numpy.mean(model_config.x1EQ)
        yc = numpy.mean(model_config.yc)
        Iext1 = numpy.mean(model_config.Iext1)
        x0cr = numpy.mean(model_config.x0cr)  # Critical x0
        r = numpy.mean(model_config.rx0)
        # The point of the linear approximation (1st order Taylor expansion)
        x1LIN = def_x1lin(X1_DEF, X1_EQ_CR_DEF, len(region_labels))
        x1SQ = X1_EQ_CR_DEF
        x1lin0 = numpy.mean(x1LIN)
        # The point of the square (parabolic) approximation (2nd order Taylor expansion)
        x1sq0 = numpy.mean(x1SQ)
        if model != "2d" or zmode != numpy.array("lin"):
            x0cr, r = calc_x0cr_r(yc, Iext1, zmode=zmode, x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF,
                                  x0cr_def=X0_CR_DEF)

        # Lines:

        # x1 nullcline:
        x1 = numpy.linspace(-2.0, 2.0 / 3.0, 100)
        if model == "2d":
            y1 = yc
            b = -2.0
        else:
            y1 = calc_eq_y1(x1, yc, d=5.0)
            b = 3.0
        zX1 = calc_fx1(x1, z=0, y1=y1, Iext1=Iext1, x1_neg=None, model=model,
                       b=b)  # yc + Iext1 - x1 ** 3 - 2.0 * x1 ** 2
        # approximations:
        # linear:
        x1lin = numpy.linspace(-5.5 / 3.0, -3.5 / 3, 30)
        # x1 nullcline after linear approximation
        zX1lin = calc_fx1_2d_taylor(x1lin, x1lin0, z=0, y1=yc, Iext1=Iext1, slope=0.0, a=1.0, b=-2.0, tau1=1.0,
                                    x1_neg=None,
                                    order=2)  # yc + Iext1 + 2.0 * x1lin0 ** 3 + 2.0 * x1lin0 ** 2 - \
        # (3.0 * x1lin0 ** 2 + 4.0 * x1lin0) * x1lin  # x1 nullcline after linear approximation
        # center point without approximation:
        # zlin0 = yc + Iext1 - x1lin0 ** 3 - 2.0 * x1lin0 ** 2
        # square:
        x1sq = numpy.linspace(-5.0 / 3, -1.0, 30)
        # x1 nullcline after parabolic approximation
        zX1sq = calc_fx1_2d_taylor(x1sq, x1sq0, z=0, y1=yc, Iext1=Iext1, slope=0.0, a=1.0, b=-2.0, tau1=1.0,
                                   x1_neg=None, order=3,
                                   shape=x1sq.shape)  # + 2.0 * x1sq ** 2 + 16.0 * x1sq / 3.0 + yc + Iext1 + 64.0 / 27.0
        # center point (critical equilibrium point) without approximation:
        # zsq0 = yc + Iext1 - x1sq0 ** 3 - 2.0 * x1sq0 ** 2
        if model == "2d":
            # z nullcline:
            zZe = calc_fz(x1, z=0.0, x0=X0_CR_DEF, x0cr=x0cr, r=r, zmode=zmode)  # for epileptogenic regions
            zZne = calc_fz(x1, z=0.0, x0=X0_DEF, x0cr=x0cr, r=r, zmode=zmode)  # for non-epileptogenic regions
        else:
            x0e_6d = calc_rescaled_x0(X0_CR_DEF, yc, Iext1, zmode=zmode)
            x0ne_6d = calc_rescaled_x0(X0_DEF, yc, Iext1, zmode=zmode)
            # z nullcline:
            zZe = calc_fz(x1, z=0.0, x0=x0e_6d, zmode=zmode, model="2d")  # for epileptogenic regions
            zZne = calc_fz(x1, z=0.0, x0=x0ne_6d, zmode=zmode, model="2d")  # for non-epileptogenic regions

        fig = pyplot.figure(figure_name, figsize=FIG_SIZE)
        x1null, = pyplot.plot(x1, zX1, 'b-', label='x1 nullcline', linewidth=1)
        ax = pyplot.gca()
        ax.axes.hold(True)
        zE1null, = pyplot.plot(x1, zZe, 'g-', label='z nullcline at critical point (E=1)', linewidth=1)
        zE2null, = pyplot.plot(x1, zZne, 'g--', label='z nullcline for E=0', linewidth=1)
        sq, = pyplot.plot(x1sq, zX1sq, 'm--', label='Parabolic local approximation', linewidth=2)
        lin, = pyplot.plot(x1lin, zX1lin, 'c--', label='Linear local approximation', linewidth=2)
        pyplot.legend(handles=[x1null, zE1null, zE2null, lin, sq])

        ii = range(len(region_labels))
        if special_idx is None:
            ii = numpy.delete(ii, special_idx)

        points = []
        for i in ii:
            point, = pyplot.plot(model_config.x1EQ[i], model_config.zEQ[i], '*', mfc='k', mec='k',
                                 ms=10, alpha=0.3,
                                 label=str(i) + '.' + region_labels[i])
            points.append(point)
        if special_idx is None:
            for i in special_idx:
                point, = pyplot.plot(model_config.x1EQ[i], model_config.zEQ[i], '*', mfc='r', mec='r', ms=10, alpha=0.8,
                                     label=str(i) + '.' + region_labels[i])
                points.append(point)
        # ax.plot(x1lin0, zlin0, '*', mfc='r', mec='r', ms=10)
        # ax.axes.text(x1lin0 - 0.1, zlin0 + 0.2, 'E=0.0', fontsize=10, color='r')
        # ax.plot(x1sq0, zsq0, '*', mfc='m', mec='m', ms=10)
        # ax.axes.text(x1sq0, zsq0 - 0.2, 'E=1.0', fontsize=10, color='m')
        if model == "2d":
            ax.set_title(
                "Equilibria, nullclines and Taylor series approximations \n at the x1-z phase plane of the" +
                add_name + " for x1<0")
        else:
            ax.set_title("Equilibria, nullclines at the x1-z phase plane of the" + add_name + " for x1<0")
        ax.axes.autoscale(tight=True)
        ax.axes.set_xlabel('x1')
        ax.axes.set_ylabel('z')
        ax.axes.set_ylim(2.0, 5.0)
        if MOUSEHOOVER:
            # datacursor( lines[0], formatter='{label}'.format, bbox=dict(fc='white'),
            #           arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )    #hover=True
            HighlightingDataCursor(points[0], formatter='{label}'.format, bbox=dict(fc='white'),
                                   arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5))

        if len(fig.get_label()) == 0:
            fig.set_label(figure_name)
        else:
            figure_name = fig.get_label().replace(": ", "_").replace(" ", "_").replace("\t", "_")

        save_figure(SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figure_name=figure_name)
        check_show(SHOW_FLAG)

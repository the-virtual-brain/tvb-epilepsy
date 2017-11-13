# coding=utf-8
"""
Service to do X0/e_values Hypothesis configuration.
"""
import numpy
from matplotlib import pyplot

from tvb_epilepsy.base.computations.calculations_utils import calc_x0cr_r, calc_coupling, calc_x0, calc_fx1, \
    calc_fx1_2d_taylor, calc_fz, calc_x0_val__to_model_x0, \
    calc_model_x0_to_x0_val
from tvb_epilepsy.base.computations.equilibrium_computation import calc_eq_z, eq_x1_hypo_x0_linTaylor, \
    eq_x1_hypo_x0_optimize, def_x1lin, calc_eq_y1
from tvb_epilepsy.base.configurations import FOLDER_FIGURES, FIG_SIZE, FIG_FORMAT, SAVE_FLAG, SHOW_FLAG, MOUSEHOOVER
from tvb_epilepsy.base.constants import X1_EQ_CR_DEF, E_DEF, X0_DEF, K_DEF, YC_DEF, I_EXT1_DEF, I_EXT2_DEF, A_DEF, \
    B_DEF, D_DEF, SLOPE_DEF, S_DEF, GAMMA_DEF, X1_DEF, X0_CR_DEF
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.model.model_configuration import ModelConfiguration
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, ensure_list
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning
from tvb_epilepsy.base.utils.plot_utils import save_figure, check_show

try:
    # https://github.com/joferkington/mpldatacursor
    # pip install mpldatacursor
    # Not working with the MacosX graphic's backend
    from mpldatacursor import HighlightingDataCursor  # datacursor

    MOUSEHOOVER = True
except:
    warning("\nNo mpldatacursor module found! MOUSEHOOVER will not be available.")
    MOUSEHOOVER = False

# NOTES:
# In the future all the related to model configuration parameters might be part of the disease hypothesis:
# yc=YC_DEF, Iext1=I_EXT1_DEF, K=K_DEF, a=A_DEF, b=B_DEF
# For now, we assume default values, or externally set

logger = initialize_logger(__name__)


class ModelConfigurationService(object):
    x1EQcr = X1_EQ_CR_DEF

    def __init__(self, number_of_regions, x0_values=X0_DEF, e_values=E_DEF, yc=YC_DEF, Iext1=I_EXT1_DEF,
                 Iext2=I_EXT2_DEF, K=K_DEF, a=A_DEF, b=B_DEF, d=D_DEF, slope=SLOPE_DEF, s=S_DEF, gamma=GAMMA_DEF,
                 zmode=numpy.array("lin"), x1eq_mode="optimize"):
        self.number_of_regions = number_of_regions
        self.x0_values = x0_values * numpy.ones((self.number_of_regions,), dtype=numpy.float32)
        self.yc = yc
        self.Iext1 = Iext1
        self.Iext2 = Iext2
        self.a = a
        self.b = b
        self.d = d
        self.slope = slope
        self.s = s
        self.gamma = gamma
        self.zmode = zmode
        self.x1eq_mode = x1eq_mode
        if len(ensure_list(K)) == 1:
            self.K_unscaled = numpy.array(K) * numpy.ones((self.number_of_regions,), dtype=numpy.float32)
        elif len(ensure_list(K)) == self.number_of_regions:
            self.K_unscaled = numpy.array(K)
        else:
            warning("The length of input global coupling K is neither 1 nor equal to the number of regions!" +
                    "\nSetting model_configuration_service.K_unscaled = K_DEF for all regions")
        self.K = None
        self._normalize_global_coupling()
        self.e_values = e_values * numpy.ones((self.number_of_regions,), dtype=numpy.float32)
        self.x0cr = 0.0
        self.rx0 = 0.0
        self._compute_critical_x0_scaling()

    def __repr__(self):
        d = {"01. Number of regions": self.number_of_regions,
             "02. x0_values": self.x0_values,
             "03. e_values": self.e_values,
             "04. K_unscaled": self.K_unscaled,
             "05. K": self.K,
             "06. yc": self.yc,
             "07. Iext1": self.Iext1,
             "08. Iext2": self.Iext2,
             "09. K": self.K,
             "10. a": self.a,
             "11. b": self.b,
             "12. d": self.d,
             "13. s": self.s,
             "14. slope": self.slope,
             "15. gamma": self.gamma,
             "16. zmode": self.zmode,
             "07. x1eq_mode": self.x1eq_mode
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

    def _compute_model_x0(self, x0_values):
        return calc_x0_val__to_model_x0(x0_values, self.yc, self.Iext1, self.a, self.b, self.d, self.zmode)

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
        return calc_eq_z(x1EQ, self.yc, self.Iext1, "2d", slope=self.slope, a=self.a, b=self.b, d=self.d)

    def _compute_critical_x0_scaling(self):
        (self.x0cr, self.rx0) = calc_x0cr_r(self.yc, self.Iext1, a=self.a, b=self.b, d=self.d, zmode=self.zmode)

    def _compute_coupling_at_equilibrium(self, x1EQ, connectivity_matrix):
        return calc_coupling(x1EQ, self.K, connectivity_matrix)

    def _compute_x0_values_from_x0_model(self, x0):
        return calc_model_x0_to_x0_val(x0, self.yc, self.Iext1, self.a, self.b, self.d, self.zmode)

    def _compute_x0_values(self, x1EQ, zEQ, connectivity_matrix):
        x0 = calc_x0(x1EQ, zEQ, self.K, connectivity_matrix)
        return self._compute_x0_values_from_x0_model(x0)

    def _compute_e_values(self, x1EQ):
        return 3.0 * x1EQ + 5.0

    def _compute_params_after_equilibration(self, x1EQ, zEQ, connectivity_matrix):
        self._compute_critical_x0_scaling()
        Ceq = self._compute_coupling_at_equilibrium(x1EQ, connectivity_matrix)
        x0_values = self._compute_x0_values(x1EQ, zEQ, connectivity_matrix)
        e_values = self._compute_e_values(x1EQ)
        x0 = self._compute_model_x0(x0_values)
        return x0, Ceq, x0_values, e_values

    def _compute_x1_and_z_equilibrium_from_E(self, e_values):
        x1EQ = self._compute_x1_equilibrium_from_E(e_values)
        zEQ = self._compute_z_equilibrium(x1EQ)
        return x1EQ, zEQ

    def _compute_x1_equilibrium(self, e_indices, x1EQ, zEQ, x0_values, connectivity_matrix):
        self._compute_critical_x0_scaling()
        x0 = self._compute_model_x0(x0_values)
        x0_indices = numpy.delete(numpy.array(range(connectivity_matrix.shape[0])), e_indices)
        if self.x1eq_mode == "linTaylor":
            x1EQ = \
                eq_x1_hypo_x0_linTaylor(x0_indices, e_indices, x1EQ, zEQ, x0, self.K,
                                        connectivity_matrix, self.yc, self.Iext1, self.a, self.b, self.d)[0]
        else:
            x1EQ = \
                eq_x1_hypo_x0_optimize(x0_indices, e_indices, x1EQ, zEQ, x0, self.K,
                                       connectivity_matrix, self.yc, self.Iext1, self.a, self.b, self.d)[0]
        return x1EQ

    def _normalize_global_coupling(self):
        self.K = self.K_unscaled / self.number_of_regions

    def configure_model_from_equilibrium(self, x1EQ, zEQ, connectivity_matrix):
        # x1EQ, zEQ = self._ensure_equilibrum(x1EQ, zEQ) # We don't this by default anymore
        x0, Ceq, x0_values, e_values = self._compute_params_after_equilibration(x1EQ, zEQ, connectivity_matrix)
        return ModelConfiguration(self.yc, self.Iext1, self.Iext2, self.K, self.a, self.b, self.d,
                                                 self.slope, self.s, self.gamma, x1EQ, zEQ, Ceq, x0, x0_values,
                                                 e_values, self.zmode, connectivity_matrix)

    def configure_model_from_E_hypothesis(self, disease_hypothesis, connectivity_matrix):
        # Always normalize K first
        self._normalize_global_coupling()

        # Then apply connectivity disease hypothesis scaling if any:
        if len(disease_hypothesis.w_indices) > 0:
            connectivity_matrix *= disease_hypothesis.get_connectivity_disease()

        # All nodes except for the diseased ones will get the default epileptogenicity:
        e_values = numpy.array(self.e_values)
        e_values[disease_hypothesis.e_indices] = disease_hypothesis.e_values

        # Compute equilibrium from epileptogenicity:
        x1EQ, zEQ = self._compute_x1_and_z_equilibrium_from_E(e_values)

        return self.configure_model_from_equilibrium(x1EQ, zEQ, connectivity_matrix)

    def configure_model_from_hypothesis(self, disease_hypothesis, connectivity_matrix):
        # Always normalize K first
        self._normalize_global_coupling()

        # Then apply connectivity disease hypothesis scaling if any:
        if len(disease_hypothesis.w_indices) > 0:
            connectivity_matrix *= disease_hypothesis.get_connectivity_disease()

        # We assume that all nodes have the default (healthy) excitability:
        x0_values = numpy.array(self.x0_values)
        # ...and some  excitability-diseased ones:
        x0_values[disease_hypothesis.x0_indices] = disease_hypothesis.x0_values
        # x0_values values must have size of len(x0_indices):
        x0_values = numpy.delete(x0_values, disease_hypothesis.e_indices)

        # There might be some epileptogenicity-diseased regions as well:
        # Initialize with the default e_values
        e_values = numpy.array(self.e_values)
        # and assign any diseased E_values if any
        e_values[disease_hypothesis.e_indices] = disease_hypothesis.e_values

        # Compute equilibrium from epileptogenicity:
        x1EQ_temp, zEQ_temp = self._compute_x1_and_z_equilibrium_from_E(e_values)

        # Now, solve the system in order to compute equilibrium:
        x1EQ = self._compute_x1_equilibrium(disease_hypothesis.e_indices, x1EQ_temp, zEQ_temp, x0_values,
                                            connectivity_matrix)
        zEQ = self._compute_z_equilibrium(x1EQ)

        return self.configure_model_from_equilibrium(x1EQ, zEQ, connectivity_matrix)

    def plot_nullclines_eq(self, model_config, region_labels, special_idx, model, zmode, figure_name,
                           show_flag=SHOW_FLAG, save_flag=SAVE_FLAG,
                           figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT):
        add_name = " " + "Epileptor " + model + " z-" + str(zmode)
        figure_name = figure_name + add_name

        # Fixed parameters for all regions:
        x1eq = model_config.x1EQ
        zeq = model_config.zEQ
        a = numpy.mean(model_config.a)
        b = numpy.mean(model_config.b)
        d = numpy.mean(model_config.d)
        yc = numpy.mean(model_config.yc)
        Iext1 = numpy.mean(model_config.Iext1)
        slope = numpy.mean(model_config.slope)
        # The point of the linear approximation (1st order Taylor expansion)
        x1LIN = def_x1lin(X1_DEF, X1_EQ_CR_DEF, len(region_labels))
        x1SQ = X1_EQ_CR_DEF
        x1lin0 = numpy.mean(x1LIN)
        # The point of the square (parabolic) approximation (2nd order Taylor expansion)
        x1sq0 = numpy.mean(x1SQ)

        # Lines:

        # x1 nullcline:
        x1 = numpy.linspace(-2.0, 2.0, 100)
        y1 = calc_eq_y1(x1, yc, d=d)

        # z nullcine
        zX1 = calc_fx1(x1, z=0, y1=y1, Iext1=Iext1, slope=slope, a=a, b=b, d=d, tau1=1.0, x1_neg=None, model=model,
                       x2=0.0, )  # yc + Iext1 - x1 ** 3 - 2.0 * x1 ** 2

        # approximations:
        # linear:
        x1lin = numpy.linspace(-5.5 / 3.0, -3.5 / 3, 30)
        # x1 nullcline after linear approximation
        zX1lin = calc_fx1_2d_taylor(x1lin, x1lin0, z=0, y1=yc, Iext1=Iext1, slope=slope, a=a, b=b, d=d, tau1=1.0,
                                    x1_neg=None, order=2)  # yc + Iext1 + 2.0 * x1lin0 ** 3 + 2.0 * x1lin0 ** 2 - \
        # (3.0 * x1lin0 ** 2 + 4.0 * x1lin0) * x1lin  # x1 nullcline after linear approximation
        # center point without approximation:
        # zlin0 = yc + Iext1 - x1lin0 ** 3 - 2.0 * x1lin0 ** 2
        # square:
        x1sq = numpy.linspace(-5.0 / 3, -1.0, 30)
        # x1 nullcline after parabolic approximation
        zX1sq = calc_fx1_2d_taylor(x1sq, x1sq0, z=0, y1=yc, Iext1=Iext1, slope=slope, a=a, b=b, d=d, tau1=1.0,
                                   x1_neg=None, order=3,
                                   shape=x1sq.shape)  # + 2.0 * x1sq ** 2 + 16.0 * x1sq / 3.0 + yc + Iext1 + 64.0 / 27.0
        # center point (critical equilibrium point) without approximation:
        # zsq0 = yc + Iext1 - x1sq0 ** 3 - 2.0 * x1sq0 ** 2
        x0e = calc_x0_val__to_model_x0(X0_CR_DEF, yc, Iext1, a=a, b=b, d=d, zmode=zmode)
        x0ne = calc_x0_val__to_model_x0(X0_DEF, yc, Iext1, a=a, b=b, d=d, zmode=zmode)
        # z nullcline:
        zZe = calc_fz(x1, z=0.0, x0=x0e, tau1=1.0, tau0=1.0, zmode=zmode)  # for epileptogenic regions
        zZne = calc_fz(x1, z=0.0, x0=x0ne, tau1=1.0, tau0=1.0, zmode=zmode)  # for non-epileptogenic regions

        fig = pyplot.figure(figure_name, figsize=FIG_SIZE)
        x1null, = pyplot.plot(x1, zX1, 'b-', label='x1 nullcline', linewidth=1)
        ax = pyplot.gca()
        ax.axes.hold(True)
        zE1null, = pyplot.plot(x1, zZe, 'g-', label='z nullcline at critical point (e_values=1)', linewidth=1)
        zE2null, = pyplot.plot(x1, zZne, 'g--', label='z nullcline for e_values=0', linewidth=1)
        sq, = pyplot.plot(x1sq, zX1sq, 'm--', label='Parabolic local approximation', linewidth=2)
        lin, = pyplot.plot(x1lin, zX1lin, 'c--', label='Linear local approximation', linewidth=2)
        pyplot.legend(handles=[x1null, zE1null, zE2null, lin, sq])

        ii = range(len(region_labels))
        if special_idx is None:
            ii = numpy.delete(ii, special_idx)

        points = []
        for i in ii:
            point, = pyplot.plot(x1eq[i], zeq[i], '*', mfc='k', mec='k',
                                 ms=10, alpha=0.3, label=str(i) + '.' + region_labels[i])
            points.append(point)
        if special_idx is None:
            for i in special_idx:
                point, = pyplot.plot(x1eq[i], zeq[i], '*', mfc='r', mec='r', ms=10, alpha=0.8,
                                     label=str(i) + '.' + region_labels[i])
                points.append(point)
        # ax.plot(x1lin0, zlin0, '*', mfc='r', mec='r', ms=10)
        # ax.axes.text(x1lin0 - 0.1, zlin0 + 0.2, 'e_values=0.0', fontsize=10, color='r')
        # ax.plot(x1sq0, zsq0, '*', mfc='m', mec='m', ms=10)
        # ax.axes.text(x1sq0, zsq0 - 0.2, 'e_values=1.0', fontsize=10, color='m')
        if model == "2d":
            ax.set_title(
                "Equilibria, nullclines and Taylor series approximations \n at the x1-z phase plane of the" +
                add_name + " for x1<0")
        else:
            ax.set_title("Equilibria, nullclines at the x1-z phase plane of the" + add_name + " for x1<0")
        ax.axes.autoscale(tight=True)
        ax.axes.set_xlabel('x1')
        ax.axes.set_ylabel('z')
        if MOUSEHOOVER:
            # datacursor( lines[0], formatter='{label}'.format, bbox=dict(fc='white'),
            #           arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5) )    #hover=True
            HighlightingDataCursor(points[0], formatter='{label}'.format, bbox=dict(fc='white'),
                                   arrowprops=dict(arrowstyle='simple', fc='white', alpha=0.5))

        if len(fig.get_label()) == 0:
            fig.set_label(figure_name)
        else:
            figure_name = fig.get_label().replace(": ", "_").replace(" ", "_").replace("\t", "_")

        save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
        check_show(show_flag)

    def update_for_pse(self, values, paths, indices):
        for i, val in enumerate(paths):
            vals = val.split(".")
            if vals[0] == "model_configuration_service":
                getattr(self, vals[1])[indices[i]] = values[i]
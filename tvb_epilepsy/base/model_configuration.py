# coding=utf-8
"""
Class to keep the model configuration values.
This will be used to populate a Model instance needed in order to launch a simulation.
"""

from collections import OrderedDict

import numpy as np

from tvb_epilepsy.base.constants import X0_DEF, K_DEF, YC_DEF, I_EXT1_DEF, A_DEF, B_DEF

from tvb_epilepsy.base.utils import formal_repr, dicts_of_lists_to_lists_of_dicts
from tvb_epilepsy.base.h5_model import object_to_h5_model

from tvb_epilepsy.base.constants import X1_DEF, X1_EQ_CR_DEF, X0_CR_DEF, \
                                        FOLDER_FIGURES, VERY_LARGE_SIZE, SMALL_SIZE, FIG_FORMAT, SAVE_FLAG, SHOW_FLAG
from matplotlib import pyplot
try:
    #https://github.com/joferkington/mpldatacursor
    #pip install mpldatacursor
    #Not working with the MacosX graphic's backend
    from mpldatacursor import HighlightingDataCursor #datacursor
    MOUSEHOOVER = True
except ImportError:
    pass

from tvb_epilepsy.base.plot_tools import plot_in_columns, _save_figure, _check_show
from tvb_epilepsy.base.calculations import calc_fx1, calc_fz, calc_fx1_2d_taylor, calc_rescaled_x0, \
    calc_x0cr_r
from tvb_epilepsy.base.equilibrium_computation import calc_eq_y1, def_x1lin


class ModelConfiguration(object):

    def __init__(self, yc=YC_DEF, Iext1=I_EXT1_DEF, K=K_DEF, a=A_DEF, b=B_DEF,
                 x0cr=None, rx0=None, x1EQ=None, zEQ=None, Ceq=None, x0_values=X0_DEF, e_values=None,
                 connectivity=None):

        # These parameters are used for every Epileptor Model...
        self.x0_values = x0_values
        self.yc = yc
        self.Iext1 = Iext1
        self.K = K
        # ...but these 2 have different values for models with more than 2 dimensions
        self.a = a
        self.b = b

        # These parameters are used only for EpileptorDP2D Model
        self.x0cr = x0cr
        self.rx0 = rx0

        # These parameters are not used for Epileptor Model, but are important to keep (h5 or plotting)
        self.x1EQ = x1EQ
        self.zEQ = zEQ
        self.Ceq = Ceq
        self.e_values = e_values

        self.connectivity_matrix = connectivity

    def __repr__(self):
        d = {
            "01. Excitability": self.x0_values,
            "02. yc": self.yc,
            "03. Iext1": self.Iext1,
            "04. K": self.K,
            "05. a": self.a,
            "06. b": self.b,
            "07. x0cr": self.x0cr,
            "08. rx0": self.rx0,
            "09. x1EQ": self.x1EQ,
            "10. zEQ": self.zEQ,
            "11. Ceq": self.Ceq,
            "12. Epileptogenicity": self.e_values
        }
        return formal_repr(self, d)

    def __str__(self):
        return self.__repr__()

    def _prepare_for_h5(self):
        h5_model = object_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        h5_model.add_or_update_metadata_attribute("Number_of_nodes", len(self.x0_values))

        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

    def prepare_for_plot(self, x0_indices=[], e_indices=[], disease_indices=[]):

        names = ["Excitabilities x0", "Epileptogenicities x0", "x1 Equilibria", "z Equilibria",
                 "Total afferent coupling \n at equilibrium"]

        data = [self.x0_values, self.e_values, self.x1EQ, self.zEQ, self.Ceq]

        disease_indices = np.unique(np.concatenate((x0_indices, e_indices, disease_indices), axis=0)).tolist()
        indices = [x0_indices, e_indices, disease_indices, disease_indices, disease_indices]
        plot_types = ["vector", "vector"]

        return dicts_of_lists_to_lists_of_dicts({"name": names, "data": data, "focus_indices": indices,
                                                 "plot_type": plot_types})

    def plot(self, n_regions=None, region_labels=[], x0_indices=[], e_indices=[], disease_indices=[],
             title="Model Configuration Overview", figure_name='', show_flag=SHOW_FLAG, save_flag=SAVE_FLAG,
             figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE):

        if n_regions == None:
            n_regions = len(self.x0_values)

        if region_labels == []:
            regions_labels = np.array([str(ii) for ii in range(n_regions)])

        disease_indices = np.unique(np.concatenate((x0_indices, e_indices, disease_indices), axis=0)).tolist()

        plot_dict_list = self.prepare_for_plot(x0_indices, e_indices, disease_indices)

        return plot_in_columns(plot_dict_list, region_labels, width_ratios=[], left_ax_focus_indices=disease_indices,
                               right_ax_focus_indices=disease_indices, title=title, figure_name=figure_name,
                               show_flag=show_flag, save_flag=save_flag, figure_dir=figure_dir,
                               figure_format=figure_format, figsize=figsize)

    def plot_nullclines_eq(self, region_labels, special_idx=None, model="2d", zmode=np.array("lin"),
                       x0ne=X0_DEF, x0e=X0_CR_DEF, figure_name='Nullclines and equilibria', show_flag=SHOW_FLAG,
                       save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=SMALL_SIZE):

        add_name = " " + "Epileptor " + model + " z-" + str(zmode)
        figure_name = figure_name + add_name

        # Fixed parameters for all regions:
        x1eq = np.mean(self.x1EQ)
        yc = np.mean(self.yc)
        Iext1 = np.mean(self.Iext1)
        x0cr = np.mean(self.x0cr)  # Critical x0
        r = np.mean(self.rx0)
        # The point of the linear approximation (1st order Taylor expansion)
        x1LIN = def_x1lin(X1_DEF, X1_EQ_CR_DEF, len(region_labels))
        x1SQ = X1_EQ_CR_DEF
        x1lin0 = np.mean(x1LIN)
        # The point of the square (parabolic) approximation (2nd order Taylor expansion)
        x1sq0 = np.mean(x1SQ)
        if model != "2d" or zmode != np.array("lin"):
            x0cr, r = calc_x0cr_r(yc, Iext1, zmode=zmode, x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF,
                                  x0cr_def=X0_CR_DEF)

        # Lines:

        # x1 nullcline:
        x1 = np.linspace(-2.0, 2.0 / 3.0, 100)
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
        x1lin = np.linspace(-5.5 / 3.0, -3.5 / 3, 30)
        # x1 nullcline after linear approximation
        zX1lin = calc_fx1_2d_taylor(x1lin, x1lin0, z=0, y1=yc, Iext1=Iext1, slope=0.0, a=1.0, b=-2.0, tau1=1.0,
                                    x1_neg=None,
                                    order=2)  # yc + Iext1 + 2.0 * x1lin0 ** 3 + 2.0 * x1lin0 ** 2 - \
        # (3.0 * x1lin0 ** 2 + 4.0 * x1lin0) * x1lin  # x1 nullcline after linear approximation
        # center point without approximation:
        # zlin0 = yc + Iext1 - x1lin0 ** 3 - 2.0 * x1lin0 ** 2
        # square:
        x1sq = np.linspace(-5.0 / 3, -1.0, 30)
        # x1 nullcline after parabolic approximation
        zX1sq = calc_fx1_2d_taylor(x1sq, x1sq0, z=0, y1=yc, Iext1=Iext1, slope=0.0, a=1.0, b=-2.0, tau1=1.0,
                                   x1_neg=None, order=3,
                                   shape=x1sq.shape)  # + 2.0 * x1sq ** 2 + 16.0 * x1sq / 3.0 + yc + Iext1 + 64.0 / 27.0
        # center point (critical equilibrium point) without approximation:
        # zsq0 = yc + Iext1 - x1sq0 ** 3 - 2.0 * x1sq0 ** 2
        if model == "2d":
            # z nullcline:
            zZe = calc_fz(x1, z=0.0, x0=x0e, x0cr=x0cr, r=r, zmode=zmode)  # for epileptogenic regions
            zZne = calc_fz(x1, z=0.0, x0=x0ne, x0cr=x0cr, r=r, zmode=zmode)  # for non-epileptogenic regions
        else:
            x0e_6d = calc_rescaled_x0(x0e, yc, Iext1, zmode=zmode)
            x0ne_6d = calc_rescaled_x0(x0ne, yc, Iext1, zmode=zmode)
            # z nullcline:
            zZe = calc_fz(x1, z=0.0, x0=x0e_6d, zmode=zmode, model="2d")  # for epileptogenic regions
            zZne = calc_fz(x1, z=0.0, x0=x0ne_6d, zmode=zmode, model="2d")  # for non-epileptogenic regions

        fig = pyplot.figure(figure_name, figsize=figsize)
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
            ii = np.delete(ii, special_idx)

        points = []
        for i in ii:
            point, = pyplot.plot(self.x1EQ[i], self.zEQ[i], '*', mfc='k', mec='k',
                                 ms=10, alpha=0.3,
                                 label=str(i) + '.' + region_labels[i])
            points.append(point)
        if special_idx is None:
            for i in special_idx:
                point, = pyplot.plot(self.x1EQ[i], self.zEQ[i], '*', mfc='r', mec='r', ms=10, alpha=0.8,
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

        _save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=figure_name)
        _check_show(show_flag)



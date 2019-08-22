# coding=utf-8

from tvb_fit.base.config import FiguresConfig
import matplotlib
matplotlib.use(FiguresConfig().MATPLOTLIB_BACKEND)
from matplotlib import pyplot

import numpy

from tvb_scripts.utils.log_error_utils import raise_value_error
from tvb_scripts.utils.data_structures_utils import linspace_broadcast
from tvb_scripts.plot.base_plotter import BasePlotter


class ProbabilisticPlotter(BasePlotter):

    def __init__(self, config=None):
        super(ProbabilisticPlotter, self).__init__(config)

    def _prepare_distribution_axes(self, distribution, loc=0.0, scale=1.0, x=numpy.array([]), ax=None, linestyle="-",
                                   lgnd=False):
        if len(x) < 1:
            x = linspace_broadcast(distribution._scipy_method("ppf", distribution.loc, distribution.scale, 0.01),
                                   distribution._scipy_method("ppf", distribution.loc, distribution.scale,0.99), 100)
        if x is not None:
            if x.ndim == 1:
                x = x[:, numpy.newaxis]
            pdf = distribution._scipy_method("pdf", loc, scale, x)
            if ax is None:
                _, ax = pyplot.subplots(1, 1)
            for ip, (xx, pp) in enumerate(zip(x.T, pdf.T)):
                ax.plot(xx.T, pp.T, linestyle=linestyle, linewidth=1, label=str(ip), alpha=0.5)
            if lgnd:
                pyplot.legend()
            return ax
        else:
            # TODO: is this message correct??
            raise_value_error("Distribution parameters do not broadcast!")

    def _prepare_parameter_axes(self, parameter, x=numpy.array([]), ax=None, lgnd=False):
        if ax is None:
            _, ax = pyplot.subplots(1, 1)
        x, pdf = parameter.scipy_method("pdf", x)
        if x.ndim == 1:
            x = x[:, numpy.newaxis]
        x, pdf = parameter.scipy_method("pdf", x)
        if ax is None:
            _, ax = pyplot.subplots(1, 1)
        for ip, (xx, pp) in enumerate(zip(x.T, pdf.T)):
            ax.plot(xx.T, pp.T, linestyle="-", linewidth=1, label=str(ip), alpha=0.5)
        if lgnd:
            pyplot.legend()
        ax.set_title(parameter.name + ": " + parameter.type + " distribution")
        return ax

    def plot_distribution(self, distribution, loc=0.0, scale=1.0, x=numpy.array([]), ax=None, linestyle="-", lgnd=True,
                          figure_name=""):
        ax = self._prepare_distribution_axes(loc, scale, x, ax, linestyle, lgnd)
        ax.set_title(distribution.type + " distribution")
        self._save_figure(pyplot.gcf(), figure_name)
        self._check_show()
        return pyplot.gcf(), ax

    def plot_probabilistic_parameter(self, parameter, x=numpy.array([]), ax=None, lgnd=True, figure_name=""):
        ax = self._prepare_parameter_axes(parameter, x, ax, lgnd)
        if len(figure_name) < 1:
            figure_name = "parameter_" + parameter.name
        self._save_figure(pyplot.gcf(), figure_name)
        self._check_show()
        return  pyplot.gcf(), ax

    def plot_probabilistic_model(self, probabilistic_model, figure_name=""):
        _, ax = pyplot.subplots(len(probabilistic_model.parameters), 1, figsize=FiguresConfig.VERY_LARGE_PORTRAIT)
        for ip, p in enumerate(probabilistic_model.parameters.values()):
            self._prepare_parameter_axes(p, x=numpy.array([]), ax=ax[ip], lgnd=False)
        self._save_figure(pyplot.gcf(), figure_name)
        self._check_show()
        return pyplot.gcf(),ax

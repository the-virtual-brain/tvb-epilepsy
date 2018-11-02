# coding=utf-8

from tvb_fit.base.config import FiguresConfig
import matplotlib
matplotlib.use(FiguresConfig().MATPLOTLIB_BACKEND)
from matplotlib import pyplot

import numpy

from tvb_fit.base.utils.data_structures_utils import ensure_list, generate_region_labels
from tvb_fit.base.computations.math_utils import compute_in_degree
from tvb_fit.base.model.virtual_patient.sensors import Sensors, SensorTypes
from tvb_fit.plot.base_plotter import BasePlotter


class HeadPlotter(BasePlotter):

    def __init__(self, config=None):
        super(HeadPlotter, self).__init__(config)

    def _plot_connectivity(self, connectivity, figure_name='Connectivity '):
        pyplot.figure(figure_name + str(connectivity.number_of_regions), self.config.figures.VERY_LARGE_SIZE)
        axes = []
        axes.append(self.plot_regions2regions(connectivity.normalized_weights,
                                              connectivity.region_labels, 121, "normalised weights"))
        axes.append(self.plot_regions2regions(connectivity.tract_lengths,
                                              connectivity.region_labels, 122, "tract lengths"))
        self._save_figure(None, figure_name.replace(" ", "_").replace("\t", "_"))
        self._check_show()
        return pyplot.gcf(), tuple(axes)

    def _plot_connectivity_stats(self, connectivity, figsize=FiguresConfig.VERY_LARGE_SIZE, figure_name='HeadStats '):
        pyplot.figure("Head stats " + str(connectivity.number_of_regions), figsize=figsize)
        areas_flag = len(connectivity.areas) == len(connectivity.region_labels)
        axes=[]
        axes.append(self.plot_vector(compute_in_degree(connectivity.normalized_weights), connectivity.region_labels,
                              111 + 10 * areas_flag, "w in-degree"))
        if len(connectivity.areas) == len(connectivity.region_labels):
            axes.append(self.plot_vector(connectivity.areas, connectivity.region_labels, 122, "region areas"))
        self._save_figure(None, figure_name.replace(" ", "").replace("\t", ""))
        self._check_show()
        return pyplot.gcf(), tuple(axes)

    def _plot_sensors(self, sensors, region_labels, count=1):
        if sensors.gain_matrix is None:
            return count
        figure, ax, cax = self._plot_gain_matrix(sensors, region_labels,
                                                  title=str(count) + " - " + sensors.s_type.value + " - Projection")
        count += 1
        return count, figure, ax, cax

    def _plot_gain_matrix(self, sensors, region_labels, figure=None, title="Projection",
                          show_x_labels=True, show_y_labels=True, x_ticks=numpy.array([]), y_ticks=numpy.array([]),
                          figsize=FiguresConfig.VERY_LARGE_SIZE):
        if not (isinstance(figure, pyplot.Figure)):
            figure = pyplot.figure(title, figsize=figsize)
        ax, cax1 = self._plot_matrix(sensors.gain_matrix, sensors.labels, region_labels, 111, title,
                                     show_x_labels, show_y_labels, x_ticks, y_ticks)
        self._save_figure(None, title)
        self._check_show()
        return figure, ax, cax1

    def plot_head(self, head):
        output = []
        output.append(self._plot_connectivity(head.connectivity))
        output.append(self._plot_connectivity_stats(head.connectivity))
        count = 1
        for s_type in SensorTypes:
            sensors = getattr(head, "sensors" + s_type.value)
            if isinstance(sensors, (list, Sensors)):
                sensors_list = ensure_list(sensors)
                if len(sensors_list) > 0:
                    for s in sensors_list:
                        count, figure, ax, cax = self._plot_sensors(s, head.connectivity.region_labels, count)
                        output.append((figure, ax, cax))
        return tuple(output)

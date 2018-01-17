import os
import numpy as np
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import AgglomerativeClustering

from tvb_epilepsy.base.constants.configurations import SHOW_FLAG, SAVE_FLAG, FOLDER_FIGURES, FIG_FORMAT, VERY_LARGE_SIZE
from tvb_epilepsy.base.model.vep.sensors import Sensors
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error, warning
from tvb_epilepsy.base.utils.math_utils import select_greater_values_array_inds, compute_in_degree
from tvb_epilepsy.base.utils.plot_utils import plot_regions2regions, save_figure, check_show, plot_vector


class HeadService(object):

    def compute_nearest_regions_to_sensors(self, head, sensors=None, target_contacts=None, s_type=Sensors.TYPE_SEEG,
                                           sensors_id=0, n_regions=None, gain_matrix_th=None):
        if not (isinstance(sensors, Sensors)):
            sensors = head.get_sensors_id(s_type=s_type, sensor_ids=sensors_id)
        n_contacts = sensors.labels.shape[0]
        if isinstance(target_contacts, (list, tuple, np.ndarray)):
            target_contacts = ensure_list(target_contacts)
            for itc, tc in enumerate(target_contacts):
                if isinstance(tc, int):
                    continue
                elif isinstance(tc, basestring):
                    target_contacts[itc] = sensors.contact_label_to_index([tc])
                else:
                    raise_value_error("target_contacts[" + str(itc) + "] = " + str(tc) +
                                      "is neither an integer nor a string!")
        else:
            target_contacts = range(n_contacts)
        auto_flag = False
        if n_regions is "all":
            n_regions = head.connectivity.number_of_regions
        elif not (isinstance(n_regions, int)):
            auto_flag = True
        nearest_regions = []
        for tc in target_contacts:
            projs = sensors.gain_matrix[tc]
            inds = np.argsort(projs)[::-1]
            if auto_flag:
                n_regions = select_greater_values_array_inds(projs[inds], threshold=gain_matrix_th)
            inds = inds[:n_regions]
            nearest_regions.append((inds, head.connectivity.region_labels[inds], projs[inds]))
        return nearest_regions

    def plot_head(self, head, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES,
                  figure_format=FIG_FORMAT):
        # plot connectivity
        self._plot_connectivity(head.connectivity, show_flag, save_flag, figure_dir, figure_format)
        self._plot_connectivity_stats(head.connectivity, show_flag, save_flag, figure_dir, figure_format)
        # plot sensor gain_matrixs
        count = 1
        for s_type in Sensors.SENSORS_TYPES:
            sensors = getattr(head, "sensors" + s_type)
            if isinstance(sensors, (list, Sensors)):
                sensors_list = ensure_list(sensors)
                if len(sensors_list) > 0:
                    for s in sensors_list:
                        count = self._plot_sensors(s, head.connectivity.region_labels, count, show_flag,
                                                   save_flag, figure_dir, figure_format)

    def _plot_connectivity(self, connectivity, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES,
                           figure_format=FIG_FORMAT, figure_name='Connectivity ', figsize=VERY_LARGE_SIZE):
        # plot connectivity
        pyplot.figure(figure_name + str(connectivity.number_of_regions), figsize)
        # plot_regions2regions(conn.weights, conn.region_labels, 121, "weights")
        plot_regions2regions(connectivity.normalized_weights, connectivity.region_labels, 121, "normalised weights")
        plot_regions2regions(connectivity.tract_lengths, connectivity.region_labels, 122, "tract lengths")
        if save_flag:
            save_figure(figure_dir=figure_dir, figure_format=figure_format,
                        figure_name=figure_name.replace(" ", "_").replace("\t", "_"))
        check_show(show_flag=show_flag)

    def _plot_connectivity_stats(self, connectivity, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG,
                                 figure_dir=FOLDER_FIGURES,
                                 figure_format=FIG_FORMAT,
                                 figsize=VERY_LARGE_SIZE, figure_name='HeadStats '):
        pyplot.figure("Head stats " + str(connectivity.number_of_regions), figsize=figsize)
        areas_flag = len(connectivity.areas) == len(connectivity.region_labels)
        ax = plot_vector(compute_in_degree(connectivity.normalized_weights), connectivity.region_labels,
                         111 + 10 * areas_flag,
                         "w in-degree")
        ax.invert_yaxis()
        if len(connectivity.areas) == len(connectivity.region_labels):
            ax = plot_vector(connectivity.areas, connectivity.region_labels, 122, "region areas")
            ax.invert_yaxis()
        if save_flag:
            save_figure(figure_dir=figure_dir, figure_format=figure_format,
                        figure_name=figure_name.replace(" ", "").replace("\t", ""))
        check_show(show_flag=show_flag)

    def _plot_sensors(self, sensors, region_labels, count=1, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG,
                      figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT):
        # plot sensors:
        if sensors.gain_matrix is None:
            return count
        self._plot_gain_matrix(sensors, region_labels, title=str(count) + " - " + sensors.s_type + " - Projection",
                               show_flag=show_flag, save_flag=save_flag, figure_dir=figure_dir,
                               figure_format=figure_format)
        count += 1
        return count

    def _plot_gain_matrix(self, sensors, region_labels, figure=None, title="Projection", y_labels=1, x_labels=1,
                          x_ticks=np.array([]), y_ticks=np.array([]), show_flag=SHOW_FLAG, save_flag=SAVE_FLAG,
                          figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE, figure_name=''):
        if not (isinstance(figure, pyplot.Figure)):
            figure = pyplot.figure(title, figsize=figsize)
        n_sensors = sensors.number_of_sensors
        n_regions = len(region_labels)
        if len(x_ticks) == 0:
            x_ticks = np.array(range(n_sensors), dtype=np.int32)
        if len(y_ticks) == 0:
            y_ticks = np.array(range(n_regions), dtype=np.int32)
        cmap = pyplot.set_cmap('autumn_r')
        img = pyplot.imshow(sensors.gain_matrix[x_ticks][:, y_ticks].T, cmap=cmap, interpolation='none')
        pyplot.grid(True, color='black')
        if y_labels > 0:
            region_labels = np.array(["%d. %s" % l for l in zip(range(n_regions), region_labels)])
            pyplot.yticks(y_ticks, region_labels[y_ticks])
        else:
            pyplot.yticks(y_ticks)
        if x_labels > 0:
            sensor_labels = np.array(["%d. %s" % l for l in zip(range(n_sensors), sensors.labels)])
            pyplot.xticks(x_ticks, sensor_labels[x_ticks], rotation=90)
        else:
            pyplot.xticks(x_ticks)
        ax = figure.get_axes()[0]
        ax.autoscale(tight=True)
        pyplot.title(title)
        divider = make_axes_locatable(ax)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        pyplot.colorbar(img, cax=cax1)  # fraction=0.046, pad=0.04) #fraction=0.15, shrink=1.0
        if figure_name == "":
            figure_name = title
        save_figure(save_flag, figure_dir=figure_dir, figure_format=figure_format, figure_name=title)
        check_show(show_flag)
        return figure

    def select_sensors_power(self, sensors, power, selection=[], power_th=0.5):
        if len(selection) == 0:
            selection = range(sensors.number_of_sensors)
        return (np.array(selection)[select_greater_values_array_inds(power, power_th)]).tolist()

    def select_sensors_rois(self, sensors, rois=None, initial_selection=[], gain_matrix_th=0.5):
        if len(initial_selection) == 0:
            initial_selection = range(sensors.number_of_sensors)
        selection = []
        if sensors.gain_matrix is None:
            raise_value_error("Projection matrix is not set!")
        else:
            for proj in sensors.gain_matrix[initial_selection].T[rois]:
                selection += (
                    np.array(initial_selection)[select_greater_values_array_inds(proj, gain_matrix_th)]).tolist()
        return np.unique(selection).tolist()

    def select_sensors_corr(self, sensors, distance, initial_selection=[], n_electrodes=10, sensors_per_electrode=1,
                            power=None, group_electrodes=False):
        if len(initial_selection) == 0:
            initial_selection = range(sensors.number_of_sensors)
        n_sensors = len(initial_selection)
        if n_sensors > 2:
            initial_selection = np.array(initial_selection)
            distance = 1.0 - distance
            if group_electrodes:
                elec_labels, elec_inds = sensors.group_sensors_to_electrodes(sensors.labels[initial_selection])
                if len(elec_labels) >= 2:
                    noconnectivity = np.ones((n_sensors, n_sensors))
                    for ch in elec_inds:
                        noconnectivity[np.meshgrid(ch, ch)] = 0.0
                    distance = distance * noconnectivity
            n_electrodes = np.minimum(np.maximum(n_electrodes, 3), n_sensors // sensors_per_electrode)
            clustering = AgglomerativeClustering(n_electrodes, affinity="precomputed", linkage="average")
            clusters_labels = clustering.fit_predict(distance)
            selection = []
            for cluster_id in range(len(np.unique(clusters_labels))):
                cluster_inds = np.where(clusters_labels == cluster_id)[0]
                n_select = np.minimum(sensors_per_electrode, len(cluster_inds))
                if power is not None and len(ensure_list(power)) == n_sensors:
                    inds_select = np.argsort(power[cluster_inds])[-n_select:]
                else:
                    inds_select = range(n_select)
                selection.append(initial_selection[cluster_inds[inds_select]])
            return np.unique(np.hstack(selection)).tolist()
        else:
            warning("Number of sensors' left < 6!\n" + "Skipping clustering and returning all of them!")
            return initial_selection

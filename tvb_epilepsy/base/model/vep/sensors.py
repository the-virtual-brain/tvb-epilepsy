
from collections import OrderedDict

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tvb_epilepsy.base.constants.configurations import FOLDER_FIGURES, VERY_LARGE_SIZE, FIG_FORMAT, SAVE_FLAG, SHOW_FLAG
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error, warning
from tvb_epilepsy.base.utils.data_structures_utils import reg_dict, formal_repr, sort_dict, ensure_list, \
                                                                       split_string_text_numbers, construct_import_path
from tvb_epilepsy.base.utils.math_utils import compute_projection, select_greater_values_array_inds
from tvb_epilepsy.base.utils.plot_utils import save_figure, check_show
from tvb_epilepsy.base.h5_model import convert_to_h5_model


class Sensors(object):

    TYPE_EEG = 'EEG'
    TYPE_MEG = "MEG"
    TYPE_SEEG = "SEEG"
    SENSORS_TYPES = [TYPE_SEEG, TYPE_MEG, TYPE_EEG]

    labels = np.array([])
    locations = np.array([])
    orientations = None
    projection = None
    s_type = ''

    def __init__(self, labels, locations, orientations=None, projection=None, s_type=TYPE_SEEG):
        self.labels = labels
        self.locations = locations
        self.orientations = orientations
        self.projection = projection
        self.s_type = s_type
        self.channel_labels = np.array([])
        self.channel_inds = np.array([])
        self.channel_labels, self.channel_inds = self.group_contacts_to_electrodes()
        self.group_contacts_to_electrodes()
        self.context_str = "from " + construct_import_path(__file__) + " import Sensors"
        self.create_str = "Sensors(np.array([]), np.array([]), s_type='" + self.s_type + "')"

    def summary(self):
        d = {"1. sensors type": self.s_type,
             "2. locations": reg_dict(self.locations, self.labels)}
        return formal_repr(self, OrderedDict(sorted(d.items(), key=lambda t: t[0]) ) )


    @property
    def number_of_sensors(self):
        return self.locations.shape[0]

    def __repr__(self):
        d = {"1. sensors' type": self.s_type,
             "2. number of sensors": self.number_of_sensors,
             "3. labels": reg_dict(self.labels),
             "4. locations": reg_dict(self.locations, self.labels),
             "5. orientations": reg_dict(self.orientations, self.labels),
             "6. projection": self.projection}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "Sensors")
        h5_model.add_or_update_metadata_attribute("Number_of_sensors", self.number_of_sensors)
        h5_model.add_or_update_metadata_attribute("Sensors_subtype", self.s_type)
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

    def contact_label_to_index(self, labels):
        indexes = []
        for label in labels:
            try:
                indexes.append(np.where([np.array(lbl) == np.array(label) for lbl in self.labels])[0][0])
            except:
                print("WTF")
        if len(indexes) == 1:
            return indexes[0]
        else:
            return indexes

    def calculate_projection(self, connectivity):
        return compute_projection(self.locations, connectivity.centers, normalize=95, ceil=1.0)

    def group_contacts_to_electrodes(self, labels=None):
        if labels is None:
            labels = self.labels
        contact_names = np.array(split_string_text_numbers(labels))
        channel_labels = np.unique(contact_names[:, 0])
        channel_inds = []
        for chlbl in channel_labels:
            channel_inds.append(np.where(contact_names[:, 0] == chlbl)[0])
        return channel_labels, channel_inds

    def select_contacts_rois(self, rois=None, initial_selection=[], projection_th=0.5):
        if len(initial_selection) == 0:
            initial_selection = range(self.number_of_sensors)
        selection = []
        if self.projection is None:
            raise_value_error("Projection matrix is not set!")
        else:
            for proj in self.projection[initial_selection].T[rois]:
                selection += (np.array(initial_selection)[select_greater_values_array_inds(proj, projection_th)]).tolist()
        return np.unique(selection).tolist()

    def select_contacts_power(self, power, selection=[], power_th=0.5):
        if len(selection) == 0:
            selection = range(self.number_of_sensors)
        return (np.array(selection)[select_greater_values_array_inds(power, power_th)]).tolist()

    def select_contacts_corr(self, distance, initial_selection=[], n_electrodes=10, contacts_per_electrode=1,
                             power=None, group_electrodes=False):
        if len(initial_selection) == 0:
            initial_selection = range(self.number_of_sensors)
        n_contacts = len(initial_selection)
        if n_contacts > 2:
            initial_selection = np.array(initial_selection)
            distance = 1.0 - distance
            if group_electrodes:
                channel_labels, channel_inds = self.group_contacts_to_electrodes(self.labels[initial_selection])
                if len(channel_labels) >= 2:
                    noconnectivity = np.ones((n_contacts, n_contacts))
                    for ch in channel_inds:
                        noconnectivity[np.meshgrid(ch, ch)] = 0.0
                    distance = distance * noconnectivity
            n_electrodes = np.minimum(np.maximum(n_electrodes, 3), n_contacts//contacts_per_electrode)
            clustering = AgglomerativeClustering(n_electrodes, affinity="precomputed", linkage="average")
            clusters_labels = clustering.fit_predict(distance)
            selection = []
            for cluster_id in range(len(np.unique(clusters_labels))):
                cluster_inds = np.where(clusters_labels == cluster_id)[0]
                n_select = np.minimum(contacts_per_electrode, len(cluster_inds))
                if power is not None and len(ensure_list(power)) == n_contacts:
                    inds_select = np.argsort(power[cluster_inds])[-n_select:]
                else:
                    inds_select = range(n_select)
                selection.append(initial_selection[cluster_inds[inds_select]])
            return selection
        else:
            warning("Number of contacts' left < 6!\n" + "Skipping clustering and returning all of them!")
            return initial_selection

    def plot_projection(self, region_labels, figure=None, title="Projection", y_labels=1, x_labels=1,
             x_ticks=np.array([]), y_ticks=np.array([]), show_flag=SHOW_FLAG, save_flag=SAVE_FLAG,
             figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT, figsize=VERY_LARGE_SIZE, figure_name=''):
        if not (isinstance(figure, pyplot.Figure)):
            figure = pyplot.figure(title, figsize=figsize)
        n_sensors = self.number_of_sensors
        n_regions = len(region_labels)
        if len(x_ticks) == 0:
            x_ticks = np.array(range(n_sensors), dtype=np.int32)
        if len(y_ticks) == 0:
            y_ticks = np.array(range(n_regions), dtype=np.int32)
        cmap = pyplot.set_cmap('autumn_r')
        img = pyplot.imshow(self.projection[x_ticks][:, y_ticks].T, cmap=cmap, interpolation='none')
        pyplot.grid(True, color='black')
        if y_labels > 0:
            region_labels = np.array(["%d. %s" % l for l in zip(range(n_regions), region_labels)])
            pyplot.yticks(y_ticks, region_labels[y_ticks])
        else:
            pyplot.yticks(y_ticks)
        if x_labels > 0:
            sensor_labels = np.array(["%d. %s" % l for l in zip(range(n_sensors), self.labels)])
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

    def plot(self, region_labels, count=1, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG,
             figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT):
        # plot sensors:
        if self.projection is None:
            return count
        self.plot_projection(region_labels, title=str(count) + " - " + self.s_type + " - Projection",
                     show_flag=show_flag, save_flag=save_flag, figure_dir=figure_dir,
                     figure_format=figure_format)
        count += 1
        return count
import numpy as np

from tvb_epilepsy.base.configurations import FOLDER_FIGURES
from tvb_epilepsy.base.constants import SHOW_FLAG, SAVE_FLAG, FIG_FORMAT
from tvb_epilepsy.base.model.vep.surface import Surface
from tvb_epilepsy.base.model.vep.sensors import Sensors, plot_sensor_dict
from tvb_epilepsy.base.model.vep.connectivity import Connectivity
from tvb_epilepsy.base.utils.data_structures_utils import reg_dict, formal_repr, sort_dict, ensure_list
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.utils.math_utils import curve_elbow_point


class Head(object):
    """
    One patient virtualization. Fully configured for defining hypothesis on it.
    """

    def __init__(self, connectivity, cortical_surface, rm, vm, t1, name='',
                 eeg_sensors_dict={}, meg_sensors_dict={}, seeg_sensors_dict={}):

        self.connectivity = connectivity
        self.cortical_surface = cortical_surface
        self.region_mapping = rm
        self.volume_mapping = vm
        self.t1_background = t1

        self.sensorsEEG = eeg_sensors_dict
        self.sensorsMEG = meg_sensors_dict
        self.sensorsSEEG = seeg_sensors_dict

        if len(name) == 0:
            self.name = 'Head' + str(self.number_of_regions)
        else:
            self.name = name

        self.children_dict = {"Connectivity": Connectivity("", np.array([]), np.array([])),
                              "Surface": Surface(np.array([]), np.array([])),
                              "Sensors": Sensors(np.array([]), np.array([]))}

    @property
    def number_of_regions(self):
        return self.connectivity.number_of_regions

    def filter_regions(self, filter_arr):
        return self.connectivity.region_labels[filter_arr]

    def __repr__(self):
        d = {"1. name": self.name,
             "2. connectivity": self.connectivity,
             "3. RM": reg_dict(self.region_mapping, self.connectivity.region_labels),
             "4. VM": reg_dict(self.volume_mapping, self.connectivity.region_labels),
             "5. surface": self.cortical_surface,
             "6. T1": self.t1_background,
             "7. SEEG": self.sensorsSEEG,
             "8. EEG": self.sensorsEEG,
             "9. MEG": self.sensorsMEG }
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    def compute_nearest_regions_to_sensors(self, s_type, target_contacts=None, id_sensor=0, n_regions=None, th=0.95):
        if s_type is "EEG":
            sensors_dict = self.sensorsEEG
        elif s_type is "MEG":
            sensors_dict = self.sensorsMEG
        else:
            sensors_dict = self.sensorsSEEG
        sensors = sensors_dict.keys()[id_sensor]
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
            n_regions = self.connectivity.number_of_regions
        elif n_regions is "auto" or not(isinstance(n_regions, int)):
            auto_flag = True
        nearest_regions = []
        for tc in target_contacts:
            projs = sensors_dict[sensors][tc]
            inds = np.argsort(projs)[::-1]
            n_regions = curve_elbow_point(projs[inds])
            nearest_regions.append((inds[:n_regions],
                                    self.connectivity.region_labels[inds[:n_regions]],
                                    projs[inds[:n_regions]]))
        return nearest_regions

    def plot(self, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT):

        # plot connectivity
        self.connectivity.plot(show_flag, save_flag, figure_dir, figure_format)

        self.connectivity.plot_stats(show_flag, save_flag, figure_dir,figure_format)

        # plot sensor projections
        count = plot_sensor_dict(self.sensorsSEEG, self.connectivity.region_labels, 1, show_flag, save_flag,
                                 figure_dir, figure_format)
        count = plot_sensor_dict(self.sensorsEEG, self.connectivity.region_labels, count, show_flag, save_flag,
                                 figure_dir, figure_format)
        count = plot_sensor_dict(self.sensorsMEG, self.connectivity.region_labels, count, show_flag, save_flag,
                                 figure_dir, figure_format)
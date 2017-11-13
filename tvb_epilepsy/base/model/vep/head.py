import numpy as np

from tvb_epilepsy.base.configurations import FOLDER_FIGURES, FIG_FORMAT, SAVE_FLAG, SHOW_FLAG
from tvb_epilepsy.base.utils.log_error_utils import warning, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import reg_dict, formal_repr, sort_dict, ensure_list
from tvb_epilepsy.base.utils.math_utils import curve_elbow_point
from tvb_epilepsy.base.model.vep.surface import Surface
from tvb_epilepsy.base.model.vep.sensors import Sensors, TYPE_SEEG, SENSORS_TYPES
from tvb_epilepsy.base.model.vep.connectivity import Connectivity


class Head(object):
    """
    One patient virtualization. Fully configured for defining hypothesis on it.
    """

    def __init__(self, connectivity, cortical_surface, rm, vm, t1, name='', **kwargs):

        self.connectivity = connectivity
        self.cortical_surface = cortical_surface
        self.region_mapping = rm
        self.volume_mapping = vm
        self.t1_background = t1
        self.sensorsSEEG = None
        self.sensorsEEG = None
        self.sensorsMEG = None
        for s_type in SENSORS_TYPES:
            self.set_sensors(kwargs.get("sensors" + s_type), sensors_type=s_type)

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

    def get_sensors(self, sensors_type=TYPE_SEEG):
        if np.in1d(sensors_type.upper(), SENSORS_TYPES):
            return getattr(self, "sensors" + sensors_type)
        else:
            raise_value_error("Invalid input sensor type " + str(sensors_type))

    def set_sensors(self, input_sensors, sensors_type=TYPE_SEEG, reset=False):
        if input_sensors is None:
            return
        sensors = self.get_sensors(sensors_type)
        if reset == False or sensors is None:
            sensors = []
        for s in ensure_list(input_sensors):
            if isinstance(s, Sensors) and (s.s_type == sensors_type):
                if s.projection == None:
                    warning("No projection found in sensors! Computing and adding projection!")
                    s.projection = s.calculate_projection(self.connectivity)
                # if s.orientations == None:
                #     warning("No orientations found in sensors!")
                sensors.append(s)
            else:
                if s is not None:
                    raise_value_error("Input sensors:\n" + str(s) +
                                      "\nis not a valid Sensors object of type " + str(sensors_type) + "!")
        if len(sensors) == 0:
            setattr(self, "sensors"+sensors_type, None)
        elif len(sensors) == 1:
            setattr(self, "sensors" + sensors_type, sensors[0])
        else:
            setattr(self, "sensors" + sensors_type, sensors)

    def get_sensors_id(self, sensors_type=TYPE_SEEG, sensor_ids=0):
        sensors = self.get_sensors(sensors_type)
        if sensors is None:
            return sensors
        else:
            out_sensors = []
            for iS, s in enumerate(ensure_list(sensors)):
                if np.in1d(iS, sensor_ids):
                    out_sensors.append(sensors[iS])
            if len(out_sensors) == 0:
                return None
            elif len(out_sensors) == 1:
                return out_sensors[0]
            else:
                return out_sensors

    def compute_nearest_regions_to_sensors(self, sensors, target_contacts=None, id_sensor=0, n_regions=None, th=0.95):
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
            projs = sensors.projection[tc]
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
        count = 1
        for s_type in SENSORS_TYPES:
            sensors = getattr(self, "sensors" + s_type)
            if isinstance(sensors, (list, Sensors)):
                for s in ensure_list(sensors):
                    count = s.plot(self.connectivity.region_labels, count, show_flag, save_flag, figure_dir,
                                   figure_format)

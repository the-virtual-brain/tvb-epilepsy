
import os

import numpy as np

from tvb_epilepsy.base.constants.configurations import FOLDER_FIGURES, FIG_FORMAT, SAVE_FLAG, SHOW_FLAG
from tvb_epilepsy.base.utils.log_error_utils import warning, raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import reg_dict, formal_repr, sort_dict, ensure_list, \
                                                                                                  construct_import_path
from tvb_epilepsy.base.utils.math_utils import select_greater_values_array_inds
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.model.vep.sensors import Sensors


class Head(object):
    """
    One patient virtualization. Fully configured for defining hypothesis on it.
    """

    def __init__(self, connectivity, cortical_surface, rm={}, vm={}, t1={}, name='', **kwargs):
        self.connectivity = connectivity
        self.cortical_surface = cortical_surface
        self.region_mapping = rm
        self.volume_mapping = vm
        self.t1_background = t1
        self.sensorsSEEG = []
        self.sensorsEEG = []
        self.sensorsMEG = []
        for s_type in Sensors.SENSORS_TYPES:
            self.set_sensors(kwargs.get("sensors" + s_type), s_type=s_type)
        if len(name) == 0:
            self.name = 'Head' + str(self.number_of_regions)
        else:
            self.name = name
        path = construct_import_path(__file__).split("head")[0]
        self.context_str = "from " + path + "head"  + " import Head; " + \
                           "from " + path + "connectivity" + " import Connectivity; " + \
                           "from " + path + "surface" + " import Surface; "
        self.create_str = "Head(Connectivity('" + self.connectivity.file_path + "', np.array([]), np.array([])), " + \
                               "Surface(np.array([]), np.array([])))"

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

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model(self)
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HeadModel")
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = self.name + ".h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

    def write_to_folder(self, folder, conn_filename="Connectivity", cortsurf_filename="CorticalSurface"):
        if not(os.path.isdir(folder)):
            os.mkdir(folder)
        self.connectivity.write_to_h5(folder, conn_filename + ".h5", connectivity_variants=True)
        # TODO create classes and write functions for the rest of the contents of a Head
        # self.cortical_surface.write_to_h5(folder, cortsurf_filename + ".h5")
        for sensor_list in (ensure_list(self.sensorsSEEG), ensure_list(self.sensorsEEG), ensure_list(self.sensorsMEG)):
            for sensors in sensor_list:
                sensors.write_to_h5(folder, "Sensors" + sensors.s_type + "_" + str(sensors.number_of_sensors) + ".h5")


    def get_sensors(self, s_type=Sensors.TYPE_SEEG):
        if np.in1d(s_type.upper(), Sensors.SENSORS_TYPES):
            return getattr(self, "sensors" + s_type)
        else:
            raise_value_error("Invalid input sensor type " + str(s_type))

    def set_sensors(self, input_sensors, s_type=Sensors.TYPE_SEEG, reset=False):
        if input_sensors is None:
            return
        sensors = ensure_list(self.get_sensors(s_type))
        if reset is False or len(sensors) == 0:
            sensors = []
        for s in ensure_list(input_sensors):
            if isinstance(s, Sensors) and (s.s_type == s_type):
                if s.gain_matrix is None or s.gain_matrix.shape != (s.number_of_sensors, self.number_of_regions):
                    warning("No correctly sized gain matrix found in sensors! Computing and adding gain matrix!")
                    s.gain_matrix = s.compute_gain_matrix(self.connectivity)
                # if s.orientations == None or s.orientations.shape != (s.number_of_sensors, 3):
                #     warning("No orientations found in sensors!")
                sensors.append(s)
            else:
                if s is not None:
                    raise_value_error("Input sensors:\n" + str(s) +
                                      "\nis not a valid Sensors object of type " + str(s_type) + "!")
        if len(sensors) == 0:
            setattr(self, "sensors"+s_type, [])
        else:
            setattr(self, "sensors" + s_type, sensors)

    def get_sensors_id(self, s_type=Sensors.TYPE_SEEG, sensor_ids=0):
        sensors = self.get_sensors(s_type)
        if sensors is None:
            return sensors
        else:
            out_sensors = []
            sensors = ensure_list(sensors)
            for iS, s in enumerate(sensors):
                if np.in1d(iS, sensor_ids):
                    out_sensors.append(sensors[iS])
            if len(out_sensors) == 0:
                return None
            elif len(out_sensors) == 1:
                return out_sensors[0]
            else:
                return out_sensors

    def compute_nearest_regions_to_sensors(self, sensors=None, target_contacts=None, s_type=Sensors.TYPE_SEEG, sensors_id=0, n_regions=None, gain_matrix_th=None):
        if not(isinstance(sensors, Sensors)):
            sensors = self.get_sensors_id(s_type=s_type, sensors_ids=sensors_id)
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
        elif not(isinstance(n_regions, int)):
            auto_flag = True
        nearest_regions = []
        for tc in target_contacts:
            projs = sensors.gain_matrix[tc]
            inds = np.argsort(projs)[::-1]
            if auto_flag:
                n_regions = select_greater_values_array_inds(projs[inds], threshold=gain_matrix_th)
            inds = inds[:n_regions]
            nearest_regions.append((inds, self.connectivity.region_labels[inds], projs[inds]))
        return nearest_regions

    def plot(self, show_flag=SHOW_FLAG, save_flag=SAVE_FLAG, figure_dir=FOLDER_FIGURES, figure_format=FIG_FORMAT):
        # plot connectivity
        self.connectivity.plot(show_flag, save_flag, figure_dir, figure_format)
        self.connectivity.plot_stats(show_flag, save_flag, figure_dir,figure_format)
        # plot sensor gain_matrixs
        count = 1
        for s_type in Sensors.SENSORS_TYPES:
            sensors = getattr(self, "sensors" + s_type)
            if isinstance(sensors, (list, Sensors)):
                sensors_list = ensure_list(sensors)
                if len(sensors_list) > 0:
                    for s in sensors_list:
                        count = s.plot(self.connectivity.region_labels, count, show_flag, save_flag, figure_dir,
                                       figure_format)

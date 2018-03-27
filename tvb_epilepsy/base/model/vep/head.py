# coding=utf-8

import numpy as np
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error, initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import reg_dict, formal_repr, sort_dict, ensure_list
from tvb_epilepsy.base.model.vep.sensors import Sensors, SENSORS_TYPES


class Head(object):
    """
    One patient virtualization. Fully configured for defining hypothesis on it.
    """
    logger = initialize_logger(__name__)

    def __init__(self, connectivity, cortical_surface, rm={}, vm={}, t1={}, name='', **kwargs):
        self.connectivity = connectivity
        self.cortical_surface = cortical_surface
        self.region_mapping = rm
        self.volume_mapping = vm
        self.t1_background = t1
        self.sensorsSEEG = []
        self.sensorsEEG = []
        self.sensorsMEG = []
        for s_type in SENSORS_TYPES:
            self.set_sensors(kwargs.get("sensors" + s_type.value), s_type=s_type)
        if len(name) == 0:
            self.name = 'Head' + str(self.number_of_regions)
        else:
            self.name = name

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
             "9. MEG": self.sensorsMEG}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    def get_sensors(self, s_type=Sensors.TYPE_SEEG):
        if np.in1d(s_type, [stype for stype in SENSORS_TYPES]):
            return getattr(self, "sensors" + s_type.value)
        else:
            raise_value_error("Invalid input sensor type " + str(s_type.value))

    def set_sensors(self, input_sensors, s_type=Sensors.TYPE_SEEG, reset=False):
        if input_sensors is None:
            return
        sensors = ensure_list(self.get_sensors(s_type))
        if reset is False or len(sensors) == 0:
            sensors = []
        for s in ensure_list(input_sensors):
            if isinstance(s, Sensors) and (s.s_type == s_type):
                if s.gain_matrix is None or s.gain_matrix.shape != (s.number_of_sensors, self.number_of_regions):
                    self.logger.warning("No correctly sized gain matrix found in sensors! "
                                        "Computing and adding gain matrix!")
                    s.gain_matrix = s.compute_gain_matrix(self.connectivity)
                # if s.orientations == None or s.orientations.shape != (s.number_of_sensors, 3):
                #     self.logger.warning("No orientations found in sensors!")
                sensors.append(s)
            else:
                if s is not None:
                    raise_value_error("Input sensors:\n" + str(s) +
                                      "\nis not a valid Sensors object of type " + str(s_type.value) + "!")
        if len(sensors) == 0:
            setattr(self, "sensors" + s_type.value, [])
        else:
            setattr(self, "sensors" + s_type.value, sensors)

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

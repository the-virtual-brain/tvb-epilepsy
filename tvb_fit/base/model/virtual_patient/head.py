# coding=utf-8
from collections import OrderedDict
import numpy as np
from tvb_fit.base.utils.log_error_utils import raise_value_error, initialize_logger
from tvb_fit.base.utils.data_structures_utils import reg_dict, formal_repr, sort_dict, ensure_list
from tvb_fit.base.model.virtual_patient.sensors import Sensors, SensorTypes


class Head(object):
    """
    One patient virtualization. Fully configured for defining hypothesis on it.
    """
    logger = initialize_logger(__name__)

    def __init__(self, connectivity, cortical_surface=None, subcortical_surface=None,
                 cortical_region_mapping=np.array([]), subcortical_region_mapping=np.array([]),
                 vm=np.array([]), t1=np.array([]), name='', **kwargs):
        self.connectivity = connectivity
        self.cortical_surface = cortical_surface
        self.subcortical_surface = subcortical_surface
        self.cortical_region_mapping = cortical_region_mapping
        self.subcortical_region_mapping = subcortical_region_mapping
        self.volume_mapping = vm
        self.t1_background = t1
        self.sensorsSEEG = OrderedDict()
        self.sensorsEEG = OrderedDict()
        self.sensorsMEG = OrderedDict()
        for s_type in SensorTypes:
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
             "3. cortical region mapping": reg_dict(self.cortical_region_mapping, self.connectivity.region_labels),
             "4. subcortical region mapping": reg_dict(self.subcortical_region_mapping,
                                                       self.connectivity.region_labels),
             "5. VM": reg_dict(self.volume_mapping, self.connectivity.region_labels),
             "6. cortical surface": self.cortical_surface,
             "7. subcortical surface": self.cortical_surface,
             "8. T1": self.t1_background,
             "9. SEEG": self.sensorsSEEG,
             "10. EEG": self.sensorsEEG,
             "11. MEG": self.sensorsMEG}
        return formal_repr(self, sort_dict(d))

    def __str__(self):
        return self.__repr__()

    def get_sensors(self, s_type=SensorTypes.TYPE_SEEG):
        if np.in1d(s_type, [stype for stype in SensorTypes]):
            return getattr(self, "sensors" + s_type.value)
        else:
            raise_value_error("Invalid input sensor type " + str(s_type))

    def set_sensors(self, input_sensors, s_type=SensorTypes.TYPE_SEEG, reset=False):
        if input_sensors is None:
            return
        sensors = self.get_sensors(s_type)
        if reset is False or len(sensors) == 0:
            sensors = OrderedDict()
        for s_name, s in input_sensors.items():
            if isinstance(s, Sensors) and (s.s_type == s_type):
                if s.gain_matrix is None or s.gain_matrix.shape != (s.number_of_sensors, self.number_of_regions):
                    self.logger.warning("No correctly sized gain matrix found in sensors!")
                sensors[s_name] = s
            else:
                if s is not None:
                    raise_value_error("Input sensors:\n" + str(s) +
                                      "\nis not a valid Sensors object of type " + str(s_type) + "!")
        setattr(self, "sensors" + s_type.value, sensors)

    def get_sensors_by_name(self, name, s_type=SensorTypes.TYPE_SEEG):
        sensors = self.get_sensors(s_type)
        if sensors is None:
            return sensors
        else:
            out_sensors = OrderedDict()
            for s_name, s in sensors.items():
                if s_name.lower().find(name.lower()) >= 0:
                    out_sensors[s.name] = s
            if len(out_sensors) == 0:
                return None
            elif len(out_sensors) == 1:
                return out_sensors.values()[0]
            else:
                return out_sensors

    def get_sensors_by_index(self, s_type=SensorTypes.TYPE_SEEG, sensor_ids=0):
        sensors = self.get_sensors(s_type)
        if sensors is None:
            return sensors
        else:
            sensors = sensors.values()
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

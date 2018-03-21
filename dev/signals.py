
from abc import ABCMeta, abstractmethod

import numpy as np

from tvb_epilepsy.base.utils.log_error_utils import raise_value_error, warning
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.datatypes.labelled_array import LabelledArray
from tvb_epilepsy.base.model.vep.connectivity import Connectivity
from tvb_epilepsy.base.model.vep.sensors import SENSORS_TYPES, Sensors


# general comments:
#
#  - we will need builders, readers, writers...
#
#  - we need also a service that does some of the signal analysis (filtering, time-frequency analysis, normalizations etc)
# this service should call the functions inside base/computations/analyzers_utils.py, so that the latter can be
# also used independently
#
# - we could substitute all n_ variables with the more explicity number_of_ that is used elsewhere
# however I find it too long
#
# - ideally, _data should hold not numpy arrays, but arrays with names for each dimension and element,
# so that we can index and slice them accordingly, maybe use pandas?
#
# - instead of holding labels and locations for regions and/or sensors,
#   we could associate a connectivity or sensors instance to each kind of Signal


# TODO: find a better solution for this
TimeUnits = {"ms": 1e-3, "msec": 1e-3, "s": 1, "sec": 1}


class Signal(object):

    __metaclass__ = ABCMeta

    source = "simulation" # "simulation", "empirical" etc
    _time = np.array([]) # size = data.shape[0]
    _time_units = {"ms": 1e-3} # {unit keys: number to be multiplied to the basic unit of 1 sec}
    _labels = [] #
    _locations = np.array([]) #np.array((n_space, 3)) # or by reference to a sensor or connectivity instance, look below

    # Alternatively, labels and locations could be referenced from a connectivity or sensors object associated with the Signal
    # _reference_object      # a connectivity or sensors object
    # _space_inds           # a indices to make reference to the connectivity or sensors object

    def __init__(self, time, time_units=TimeUnits["ms"], labels=[], locations=[], source="simulation"):
        self._time = time
        self._time_units = {time_units, TimeUnits[time_units]}
        self._labels = list(labels)
        self._locations = locations
        self.source = source
        self.check()

    def check(self):
        self.check_data_shape()
        self.check_time()
        # ...etc other possible checks

    def permute(self, new_inds):
        # permute dimensions
        # maybe use pandas?
        pass

    def _get_time(self):
        return self.time

    def _set_time(self, time):
        self.time = time
        self.check_time()
        return self

    time = property(_get_time, _set_time)

    def _get_time_units(self):
        return self._time_units

    def _set_time_units(self, time_units):
        # some code to check that we get the correct kind of dict()
        self._time_units = time_units
        return self

    time_units = property(_get_time_units, _set_time_units)

    @property
    # or sampling_time, or sample_period, any such combination you prefer
    def time_length(self):
        return self.time.size

    @property
    # or sampling_time, or sample_period, any such combination you prefer
    def dt(self):
        return np.diff(self.time).mean()

    @property
    # or sampling_frequency, sample_freq or whatever such combination
    def fs(self):
        return 1 / self.dt / self._time_units.values()[0]  # therefore in Hz

    def _get_labels(self):
        return self._labels[0]

    def _set_labels(self, labels):
        self._labels[0] = np.array(labels)
        self.check_labels()
        return self

    labels = property(_get_labels, _set_labels)

    # def _get_samples_labels(self):
    #     if len(self._labels) > 1:
    #         return self._labels[1]
    #     else:
    #         return np.array([])
    #
    # def _set_samples_labels(self, labels):
    #     self._labels[:1] += [np.array(labels)]
    #     self.check_labels()
    #     return self
    #
    # samples_labels = property(_get_samples_labels, _set_samples_labels)

    def _get_locations(self):
        return self._locations

    def _set_locations(self, locations):
        self._locations = locations
        self.check_locations()
        return self

    locations = property(_get_locations, _set_locations)

    @abstractmethod
    def check_time(self):
        pass

    def check_labels(self):
        # the number of labels should be either 1 or equal to the number of elements to the correspodning dimension
        pass

    def check_locations(self):
        # the size of locations should be (n_space, 3)
        pass

    def check_data_shape(self):
        # make sure that all data have the same shape
        pass

    def remove_samples_axis(self):
        # remove the third axis when empty
        # remove possible labels as well
        # self.labels = self.labels[:1]
        # return self
        pass

    def new_samples_axis(self, labels):
        # create an empty third axis
        # self.labels = self.labels[:1] + [np.array(labels)]
        # return self
        pass


class RegionsSignal(Signal):

    _data = []

    _state_variables = []

    _connectivity_of_reference = None

    def __init__(self, data, state_variables, time, time_units=TimeUnits["ms"], labels=[], locations=np.array([]),
                 source="simulation", connectivity=None):
        super(self, RegionsSignal).__init__(time, time_units, labels, locations, source)
        self._data = [LabelledArray(d, labels) for d in data]
        self._state_variables = state_variables
        self._connectivity_of_reference = connectivity

    def _get_state_variable_index(self, state_variable_label):
        try:
            return self._state_variables.index(state_variable_label)
        except:
            raise_value_error("Failed to set signal with the label " + state_variable_label + "!")

    def __getattr__(self, attr):
        # use: signal.x1
        return self._data[self._get_state_variable_index(attr)]

    def __setattr__(self, attr, data):
        # use: signal.x1 = np.array
        self._data[self._get_state_variable_index(attr)] = LabelledArray(data, self._data[0]._labels)
        self.check()
        return self

    def slice(self):
        # return a new instance of the Signal that corresponds to a slice
        # operate by label -> index
        # maybe use pandas?
        pass

    @property
    def shape(self):
        self.check_data_shape()
        return self._data[0].shape

    @property
    def dims(self):
        return self._data[0].ndim

    @property
    def n_regions(self):
        return self._data[0].shape[1]

    @property
    def n_samples(self):
        self.check_data_shape()
        if self.data_dims == 3:
            return self._data[0].shape[2]
        else:
            return 1

    def check_time(self):
        self.time = self.time.flatten()
        self.check_data_shape()  # should this be redone here?
        if self.time_length != self.data.values()[0].shape[0]:
            raise_value_error("Length of time " + str(self.time_length) +
                              " and data time axis " + str(self.data[0].shape[0]) + " do not match!")

    def _get_variables(self):
        return self._state_variables

    def _set_variables(self, labels):
        self._state_variables = labels
        #TODO: add some self-consistency checking
        return self

    variables = property(_get_variables, _set_variables)

    def _get_centers(self):
        return super(RegionsSignal, self)._get_locations()

    def _set_centers(self, centers):
        return super(RegionsSignal, self)._set_locations(centers)

    centers = property(_get_centers, _set_centers)

    def _get_regions_labels(self):
        return super(RegionsSignal, self)._get_labels()

    def _set_regions_labels(self, regions_labels):
        return super(RegionsSignal, self)._set_labels(regions_labels)

    regions_labels = property(_get_regions_labels, _set_regions_labels)

    @property
    def connectivity(self):
        if isinstance(self._connectivity_of_reference, Connectivity):
            return self._connectivity_of_reference
        else:
            warning("There are no Connectivity of reference for this RegionsSignal!")
            return []

    @property
    def number_of_regions(self):
        return self.n_space()

    @property
    def regions_indices(self):
        try:
            return np.where([self.connectivity.region_labels.index(region_label)
                             for region_label in self.regions_labels])[0]
        except:
            raise_value_error("Failed to find signal's region_labels inside the reference Connectivity!")


class SensorSignal(Signal):

    _data = np.array([])

    _sensor_type = SENSORS_TYPES.SEEG_TYPE

    _sensors_of_reference = None

    def __init__(self, data, time, time_units=TimeUnits["ms"], labels=[], locations=[],
                 source="simulation", sensors=None):
        super(self, Signal).__init__(time, time_units, labels, locations, source)
        self._data = LabelledArray(data, labels)
        self._sensors_of_reference = sensors

    def __getattr__(self, attr):
        # use: signal.seeg
        if isequal_string(self._sensor_type, attr):
            return self._data
        else:
            raise_value_error("Failed to retrieve sensors' signal with the label " + attr + "!")

    def __setattr__(self, attr, data):
        # use: signal.x1 = np.array
        if isequal_string(self._sensor_type, attr):
            self._data = np.array(data)
            self.check()
            return self
        else:
            raise_value_error("Failed to set sensors' signal with the label " + attr + "!")

    @property
    def shape(self):
        self.check_data_shape()
        return self._data.shape

    @property
    def dims(self):
        return self._data.ndim

    @property
    def n_regions(self):
        return self._data.shape[1]

    @property
    def n_sensors(self):
        self.check_data_shape()
        if self.data_dims == 3:
            return self._data.shape[2]
        else:
            return 1

    def check_time(self):
        self.time = self.time.flatten()
        self.check_data_shape()  # should this be redone here?
        if self.time_length != self.data.values()[0].shape[0]:
            raise_value_error("Length of time " + str(self.time_length) +
                              " and data time axis " + str(self.data.shape[0]) + " do not match!")

    def _get_sensors_labels(self):
        return super(SensorSignal, self)._get_labels()

    def _set_sensors_labels(self, sensors_labels):
        return super(SensorSignal, self)._set_labels(sensors_labels)

    sensors_labels = property(_get_sensors_labels, _set_sensors_labels)

    @property
    def sensors(self):
        if isinstance(self._sensors_of_reference, Sensors):
            return self._sensors_of_reference
        else:
            warning("There are no Sensors of reference for this SensorsSignal!")
            return []

    @property
    def sensors_indices(self):
        if self.sensors is not None:
            try:
                return np.where([self.sensors.labels.index(sensor_label) for sensor_label in self.sensors_labels])[0]
            except:
                raise_value_error("Failed to find signal's sensors_labels inside the reference Sensors!")

    @property
    def gain_matrix(self):
        if self.sensors is not None:
            return self.sensors.gain_matrix[self.sensors_indices]

    @property
    def number_of_sensors(self):
        return self.n_space()

    def get_bipolar(self):
        # make a copy of self, make data, labels and locations bipolar, and return the new instance
        # data by taking the difference sensor_i - sensor_j
        # labels by creating new labels as label_i - label_j
        # locations by one of the two options: either assign location_i, or by taking also location_i - location_j
        # in the latter case though we break the indexing reference to the original sensors' object.
        # instead, in the former case we can have the bipolar channel i-j correspond to the monopolar i,
        # just with a different label
        #return bipolar instance
        pass



from collections import OrderedDict
import numpy as np

from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list

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


class Signal(object):

    source = "" # "simulation", "empirical" etc
    _data = OrderedDict() # i.e., {"source": np.array((n_times, n_regions, n_samples))}, {"SEEG0": np.array((n_times, n_sensors, n_samples))}
    _time = np.array([]) # size = data.shape[0]
    _time_units = {"ms": 1e-3} # {unit keys: number to be multiplied to the basic unit of 1 sec}
    _labels = [np.array([])] # list of labels'arrays, of size =1 or equal to length of dimension
                            # i.e., [sensor_labels, np.array("samples")] or [sensor_labels, np.array(["mean", "median", "std])]
    _locations = np.array([]) #np.array((n_space, 3)) # or by reference to a sensor or connectivity instance, look below

    # Alternatively, labels and locations could be referenced from a connectivity or sensors object associated with the Signal
    # _reference_object      # a connectivity or sensors object
    # _space_inds           # a indices to make reference to the connectivity or sensors object

    def __getattr__(self, attr):
        # use: signal.SEEG0
        return self._data.get(attr, None)

    def __setattr__(self, attr):
        # use: signal.SEEG0
        return self._data.get(attr, None)

    def check(self):
        self.check_data_shape()
        self.check_time()
        # ...etc other possible checks

    def some_name_of_slicing_operator(self):
        # return only the data that corresponds to a slice
        # operate by label -> index
        # maybe use pandas?
        pass

    def some_name_of_slicing_operator2(self):
        # return a new instance of the Signal that corresponds to a slice
        # operate by label -> index
        # maybe use pandas?
        pass

    def permute(self, new_inds):
        # permute dimensions
        # maybe use pandas?
        pass

    @property
    def shape(self):
        self.check_data_shape()
        return self._data.values([0]).shape

    @property
    def dims(self):
        self.check_data_shape()
        return self._data.values([0]).ndim

    @property
    def n_space(self):
        self.check_data_shape()
        return self._data.values([0]).shape[1]

    @property
    def n_times(self):
        self.check_data_shape()
        self.chekc_time()
        return self._data.values([0]).shape[0]

    @property
    def n_samples(self):
        self.check_data_shape()
        if self.data_dims == 3:
            return self._data.values([0]).shape[2]
        else:
            return 1

    @property
    def data_labels(self):
        return self.data.keys()

    def substitute_data_labels(self, oldlabels, newlabels):
        new_data = OrderedDict()
        for oldlabel, newlabel in zip(ensure_list(oldlabels), ensure_list(newlabels)):
            new_data[newlabel] = self._data[oldlabel]
        self._data = new_data
        return self

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
    def dt(self):
        return np.diff(self.time).mean()

    @property
    # or sampling_frequency, sample_freq or whatever such combination
    def fs(self):
        return 1 / self.dt / self._time_units.values()[0]  # therefore in Hz


    def _get_space_labels(self):
        return self._labels[0]

    def _set_space_labels(self, labels):
        self._labels[0] = np.array(labels)
        self.check_labels()
        return self

    space_labels = property(_get_space_labels, _set_space_labels)

    def _get_samples_labels(self):
        if len(self._labels) > 1:
            return self._labels[1]
        else:
            return np.array([])

    def _set_samples_labels(self, labels):
        self._labels[:1] += [np.array(labels)]
        self.check_labels()
        return self

    samples_labels = property(_get_samples_labels, _set_samples_labels)

    def _get_locations(self):
        return self._locations

    def _set_locations(self, locations):
        self._locations = locations
        self.check_locations()
        return self

    locations = property(_get_locations, _set_locations)

    def _set_samples_labels(self, labels):
        self._labels[:1] += [np.array(labels)]
        self.check_labels()
        return self

    samples_labels = property(_get_samples_labels, _set_samples_labels)

    def check_time(self):
        self.time = self.time.flatten()
        self.n_times = self.time.size
        self.check_data_shape() # should this be redone here?
        if self.n_times != self.data.values()[0].shape[0]:
            raise_value_error("Length of time " + str(self.n_times) +
                              " and data time axis " + str(self.data.values()[0].shape[0]) + " do not match!")

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

    def _get_centers(self):
        return super(RegionsSignal, self)._get_locations()

    def _set_centers(self, centers):
        return super(RegionsSignal, self)._set_locations(centers)

    centers = property(_get_centers, _set_centers)

    def _get_regions_labels(self):
        return super(RegionsSignal, self)._get_space_labels()

    def _set_regions_labels(self, region_labels):
        return super(RegionsSignal, self)._set_space_labels(region_labels)

    regions_labels = property(_get_regions_labels, _set_regions_labels)


    # regions_inds = super(RegionsSignal, self).space_inds

    @property
    def connectivity(self):
        return super(RegionsSignal, self)._reference_object

    @property
    def n_regions(self):
        return self.n_space()


class SensorSignal(Signal):

    # sensors_inds = indices of the sensors in the corresponding sensors instance

    def _get_sensors_labels(self):
        return super(SensorSignal, self)._get_space_labels()

    def _set_sensors_labels(self, region_labels):
        return super(SensorSignal, self)._set_space_labels(region_labels)

    sensors_labels = property(_get_sensors_labels, _set_sensors_labels)

    @property
    def n_sensors(self):
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



from collections import OrderedDict

import numpy

from tvb_epilepsy.base.utils.log_error_utils import raise_value_error
from tvb_epilepsy.base.datatypes.labelled_array import verify_index


class TimeseriesDimensions(object):
    VARIABLES = "variables"
    TIME = "time"
    SPACE = "space"
    SAMPLES = "samples"

    def getAll(self):
        return [self.VARIABLES, self.TIME, self.SPACE, self.SAMPLES]


# TODO: find a better solution for this
TimeUnits = {"ms": 1e+3, "msec": 1e+3, "s": 1.0, "sec": 1.0}


class Timeseries(object):
    dimensions = TimeseriesDimensions().getAll()

    # dimension_labels = {"space": [], "variables": []}

    def __init__(self, data, dimension_labels, time_start, sampling_period, time_unit="ms"):
        if data.ndim < 4:
            data = numpy.expand_dims(data, 3)
        self.data = data
        self.dimension_labels = dimension_labels
        self.time_start = time_start
        self.sampling_period = sampling_period
        self.time_unit = time_unit

    @property
    def number_of_variables(self):
        return self.data.shape[0]

    @property
    def number_of_times(self):
        return self.data.shape[1]

    def get_time_line(self):
        return numpy.arange(self.time_start, self.get_end_time(), self.sampling_period)

    def get_end_time(self):
        return self.time_start + self.number_of_times * self.sampling_period

    @property
    def time(self):
        return self.get_time_line()

    @property
    def sampling_frequency(self):
        return TimeUnits[self.time_unit] / self.sampling_period

    @property
    def number_of_labels(self):
        return self.data.shape[2]

    @property
    def number_of_samples(self):
        return self.data.shape[3]

    def _get_variable_index(self, variable):
        try:
            return self.dimension_labels[TimeseriesDimensions.VARIABLES].index(variable)
        except:
            raise_value_error("Failed to retrieve state variable with the label " + variable + "!" +
                              "\nExisting state variables: " +
                               str(self.dimension_labels[TimeseriesDimensions.VARIABLES]))

    def _get_label_index(self, label):
        try:
            return self.dimension_labels[TimeseriesDimensions.SPACE].index(label)
        except:
            raise_value_error("Failed to retrieve index of label " + label + "!" +
                              "Existing space labels: " + self.dimension_labels[TimeseriesDimensions.SPACE])

    def __getattr__(self, variable):
        # use: signal.x1
        return Timeseries(numpy.expand_dims(self.data[self._get_variable_index(variable)], 0),
                          OrderedDict({TimeseriesDimensions.SPACE: self.dimension_labels[TimeseriesDimensions.SPACE]}),
                          self.time_start, self.sampling_period, self.time_unit)

    def _verify_index(self, index):
        if len(index) > 1:
            # Convert possible strings in the index of space labels
            index = list(index)
            index[1] = verify_index((index[1], ), [self.dimension_labels[TimeseriesDimensions.SPACE]])[0]
            index = tuple(index)
        return index

    def _expand_index_with_variable(self, index):
        return tuple([slice(None)] + list(index))

    def __getitem__(self, index):
        return self.data.__getitem__(self._expand_index_with_variable(self._verify_index(index))).squeeze()

    def __setitem__(self, index, data):
        data = numpy.expand_dims(data, 0)  # create variable axis assuming  for index
        if data.ndim < 4:
            data = numpy.expand_dims(data, 3)  # if samples axis is missing..., create it as well...
        self.data.__setitem__(self._expand_index_with_variable(self._verify_index(index)), data)
        return self

    def _get_indices_for_labels(self, list_of_labels):
        list_of_indices_for_labels = []
        for label in list_of_labels:
            list_of_indices_for_labels.append(self._get_label_index(label))
        return list_of_indices_for_labels

    def get_subspace_by_labels(self, list_of_labels):
        subspace_data = self.data[:, :, self._get_indices_for_labels(list_of_labels), :]
        subspace_dimension_labels = dict(self.dimension_labels)
        subspace_dimension_labels[TimeseriesDimensions.SPACE] = list_of_labels
        return Timeseries(subspace_data, subspace_dimension_labels, self.time_start, self.sampling_period, self.time_unit)

    def get_subspace_by_index(self, list_of_index):
        subspace_data = self.data[:, :, list_of_index, :]
        subspace_dimension_labels = dict(self.dimension_labels)
        subspace_dimension_labels[TimeseriesDimensions.SPACE] = \
            self.dimension_labels[TimeseriesDimensions.SPACE[list_of_index]]
        return Timeseries(subspace_data, subspace_dimension_labels, self.time_start, self.sampling_period, self.time_unit)

    def get_time_window_by_index(self, index_start, index_end):
        return Timeseries(self.data[:, index_start:index_end, :, :], self.dimension_labels,
                          self.get_time_line()[index_start],
                          self.sampling_period, self.time_unit)

    def get_time_window(self, unit_start, unit_end):
        # O(1)
        timeline = self.get_time_line()
        index_start, = numpy.where(timeline == unit_start)[0]
        index_end, = numpy.where(timeline == unit_end)[0]
        return self.get_time_window_by_index(index_start, index_end)

    def get_squeezed_data(self):
        pass

    def get_lfp(self):
        # compute if not exists
        pass

    def decimate_time(self, sampling_period):
        pass

    def get_sample_window(self, index_start, index_end):
        pass

    def get_sample_window_by_percentile(self, percentile_start, percentile_end):
        pass

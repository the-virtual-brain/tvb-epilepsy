from collections import OrderedDict

import numpy


class TimeseriesDimensions(object):
    TIME = "time"
    SPACE = "space"
    STATE_VARIABLES = "state_variables"
    SAMPLES = "samples"

    def getAll(self):
        return [self.TIME, self.SPACE, self.STATE_VARIABLES, self.SAMPLES]


class Timeseries(object):
    dimensions = TimeseriesDimensions().getAll()

    # dimension_labels = {"space": [], "state_variables": []}

    def __init__(self, data, dimension_labels, time_start, time_step, time_unit="ms"):
        self.data = data
        self.dimension_labels = dimension_labels
        self.time_start = time_start
        self.time_step = time_step
        self.time_unit = time_unit

    def get_time_line(self):
        return numpy.arange(self.time_start, self.get_end_time(), self.time_step)

    def get_end_time(self):
        return self.time_start + self.data.shape[0] * self.time_step

    def get_squeezed_data(self):
        pass

    # TODO: have possibility to access this by Signal.sv_name
    def get_state_variable(self, sv_name):
        sv_data = self.data[:, :, self.dimension_labels[TimeseriesDimensions.STATE_VARIABLES].index(sv_name)]
        return Timeseries(sv_data,
                          OrderedDict({TimeseriesDimensions.SPACE: self.dimension_labels[TimeseriesDimensions.SPACE]}),
                          self.time_start, self.time_step, self.time_unit)

    def get_lfp(self):
        # compute if not exists
        pass

    def _get_indices_for_labels(self, list_of_labels):
        list_of_indices_for_labels = []
        for label in list_of_labels:
            list_of_indices_for_labels.append(self.dimension_labels[TimeseriesDimensions.SPACE].index(label))
        return list_of_indices_for_labels

    def get_subspace_by_labels(self, list_of_labels):
        list_of_indices_for_labels = self._get_indices_for_labels(list_of_labels)
        subspace_data = self.data[:, list_of_indices_for_labels, :]
        subspace_dimension_labels = self.dimension_labels
        subspace_dimension_labels[TimeseriesDimensions.SPACE] = list_of_labels
        return Timeseries(subspace_data, subspace_dimension_labels, self.time_start, self.time_step, self.time_unit)

    def get_subspace_by_index(self, list_of_index):
        subspace_data = self.data[:, list_of_index, :]
        subspace_dimension_labels = self.dimension_labels
        subspace_dimension_labels[TimeseriesDimensions.SPACE] = self.dimension_labels[
            TimeseriesDimensions.SPACE[list_of_index]]
        return Timeseries(subspace_data, subspace_dimension_labels, self.time_start, self.time_step, self.time_unit)

    def get_time_window(self, index_start, index_end):
        return Timeseries(self.data[index_start:index_end, :, :], self.dimension_labels,
                          self.get_time_line()[index_start],
                          self.time_step, self.time_unit)

    def get_time_window_by_units(self, unit_start, unit_end):
        # O(1)
        timeline = self.get_time_line()
        index_start, = numpy.where(timeline == unit_start)[0]
        index_end, = numpy.where(timeline == unit_end)[0]
        return self.get_time_window(index_start, index_end)

    def decimate_time(self, time_step):
        pass

    def get_sample_window(self, index_start, index_end):
        pass

    def get_sample_window_by_units(self, unit_start, unit_end):
        pass

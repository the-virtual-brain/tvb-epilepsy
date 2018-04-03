import numpy
from enum import Enum
from copy import deepcopy
from collections import OrderedDict
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger


class TimeseriesDimensions(Enum):
    TIME = "time"
    SPACE = "space"
    STATE_VARIABLES = "state_variables"
    SAMPLES = "samples"


class PossibleStateVariables(Enum):
    X1 = "x1"
    X2 = "x2"
    LFP = "lfp"
    Z = "z"
    Y1 = "y1"
    Y2 = "y2"
    G = "g"
    SLOPE_T = "slope_t"
    IEXT2_T = "Iext2_t"
    IEXT1_T = "Iext1_t"
    K_T = "K_t"
    X0_T = "x0_t"


class Timeseries(object):
    logger = initialize_logger(__name__)

    dimensions = TimeseriesDimensions

    # dimension_labels = {"space": [], "state_variables": []}

    def __init__(self, data, dimension_labels, time_start, time_step, time_unit="ms"):
        self.data = self.prepare_4D(data)
        self.dimension_labels = dimension_labels
        self.time_start = time_start
        self.time_step = time_step
        self.time_unit = time_unit

    def prepare_4D(self, data):
        if data.ndim < 2:
            self.logger.error("The data array is expected to be at least 2D!")
            raise ValueError
        if data.ndim < 4:
            if data.ndim == 2:
                data = numpy.expand_dims(data, 2)
            data = numpy.expand_dims(data, 3)
        return data

    @property
    def space_labels(self):
        return self.dimension_labels[TimeseriesDimensions.SPACE.value]

    @property
    def end_time(self):
        return self.time_start + (self.data.shape[0] - 1) * self.time_step

    @property
    def time_line(self):
        return numpy.arange(self.time_start, self.end_time + self.time_step, self.time_step)

    @property
    def squeezed_data(self):
        return numpy.squeeze(self.data)

    def _get_index_of_state_variable(self, sv_label):
        try:
            sv_index = self.dimension_labels[TimeseriesDimensions.STATE_VARIABLES.value].index(sv_label)
        except KeyError:
            self.logger.error("There are no state variables defined for this instance. Its shape is: %s",
                              self.data.shape)
            raise
        except ValueError:
            self.logger.error("Cannot access index of state variable label: %s. Existing state variables: %s" % (
                sv_label, self.dimension_labels[TimeseriesDimensions.STATE_VARIABLES.value]))
            raise
        return sv_index

    def get_state_variable(self, sv_label):
        sv_data = self.data[:, :, self._get_index_of_state_variable(sv_label), :]
        return Timeseries(numpy.expand_dims(sv_data, 2),
                          OrderedDict({TimeseriesDimensions.SPACE.value: self.dimension_labels[
                              TimeseriesDimensions.SPACE.value]}),
                          self.time_start, self.time_step, self.time_unit)

    @property
    def lfp(self):
        if TimeseriesDimensions.STATE_VARIABLES.value not in self.dimension_labels.keys():
            self.logger.error("No state variables are defined for this instance!")
            raise ValueError

        if PossibleStateVariables.LFP.value in self.dimension_labels[TimeseriesDimensions.STATE_VARIABLES.value]:
            return self.get_state_variable(PossibleStateVariables.LFP.value)
        if PossibleStateVariables.X1.value in self.dimension_labels[
            TimeseriesDimensions.STATE_VARIABLES.value] and PossibleStateVariables.X2.value in self.dimension_labels[
            TimeseriesDimensions.STATE_VARIABLES.value]:
            self.logger.info("%s are computed using %s and %s state variables!" % (
                PossibleStateVariables.LFP.value, PossibleStateVariables.X1.value, PossibleStateVariables.X2.value))
            y0_ts = self.get_state_variable(PossibleStateVariables.X1.value)
            y2_ts = self.get_state_variable(PossibleStateVariables.X2.value)
            lfp_data = y2_ts.data - y0_ts.data
            lfp_dim_labels = OrderedDict(
                {TimeseriesDimensions.SPACE.value: self.dimension_labels[TimeseriesDimensions.SPACE.value],
                 TimeseriesDimensions.STATE_VARIABLES.value: [PossibleStateVariables.LFP.value]})
            return Timeseries(lfp_data, lfp_dim_labels, self.time_start, self.time_step, self.time_unit)
        self.logger.error(
            "%s is not computed and cannot be computed now because state variables %s and %s are not defined!" % (
                PossibleStateVariables.LFP.value, PossibleStateVariables.X1.value, PossibleStateVariables.X2.value))
        raise ValueError

    def _get_indices_for_labels(self, list_of_labels):
        list_of_indices_for_labels = []
        for label in list_of_labels:
            try:
                space_index = self.dimension_labels[TimeseriesDimensions.SPACE.value].index(label)
            except ValueError:
                self.logger.error("Cannot access index of space label: %s. Existing space labels: %s" % (
                    label, self.dimension_labels[TimeseriesDimensions.SPACE.value]))
                raise
            list_of_indices_for_labels.append(space_index)
        return list_of_indices_for_labels

    def get_subspace_by_labels(self, list_of_labels):
        list_of_indices_for_labels = self._get_indices_for_labels(list_of_labels)
        subspace_data = self.data[:, list_of_indices_for_labels, :, :]
        subspace_dimension_labels = deepcopy(self.dimension_labels)
        subspace_dimension_labels[TimeseriesDimensions.SPACE.value] = list_of_labels
        if subspace_data.ndim == 3:
            subspace_data = numpy.expand_dims(subspace_data, 1)
        return Timeseries(subspace_data, subspace_dimension_labels, self.time_start, self.time_step, self.time_unit)

    def _check_space_indices(self, list_of_index):
        for index in list_of_index:
            if index < 0 or index > self.data.shape[1]:
                self.logger.error("Some of the given indices are out of region range: [0, %s]", self.data.shape[1])
                raise IndexError

    def get_subspace_by_index(self, list_of_index):
        self._check_space_indices(list_of_index)
        subspace_data = self.data[:, list_of_index, :, :]
        subspace_dimension_labels = deepcopy(self.dimension_labels)
        subspace_dimension_labels[TimeseriesDimensions.SPACE.value] = numpy.array(self.dimension_labels[
                                                                                      TimeseriesDimensions.SPACE.value])[
            list_of_index]
        if subspace_data.ndim == 3:
            subspace_data = numpy.expand_dims(subspace_data, 1)
        return Timeseries(subspace_data, subspace_dimension_labels, self.time_start, self.time_step, self.time_unit)

    def _get_time_unit_for_index(self, time_index):
        return self.time_start + time_index * self.time_step

    def get_time_window(self, index_start, index_end):
        if index_start < 0 or index_end > self.data.shape[0] - 1:
            self.logger.error("The time indices are outside time series interval: [%s, %s]" % (0, self.data.shape[0]))
            raise IndexError
        subtime_data = self.data[index_start:index_end, :, :, :]
        if subtime_data.ndim == 3:
            subtime_data = numpy.expand_dims(subtime_data, 0)
        return Timeseries(subtime_data, self.dimension_labels,
                          self._get_time_unit_for_index(index_start), self.time_step, self.time_unit)

    def _get_index_for_time_unit(self, time_unit):
        return int((time_unit - self.time_start) / self.time_step)

    def get_time_window_by_units(self, unit_start, unit_end):
        end_time = self.end_time
        if unit_start < self.time_start or unit_end > end_time:
            self.logger.error("The time units are outside time series interval: [%s, %s]" % (self.time_start, end_time))
            raise ValueError
        index_start = self._get_index_for_time_unit(unit_start)
        index_end = self._get_index_for_time_unit(unit_end)
        return self.get_time_window(index_start, index_end)

    def decimate_time(self, time_step):
        if time_step % self.time_step != 0:
            self.logger.error("Cannot decimate time if new time step is not a multiple of the old time step")
            raise ValueError

        index_step = int(time_step / self.time_step)
        time_data = self.data[::index_step, :, :, :]

        return Timeseries(time_data, self.dimension_labels, self.time_start, time_step, self.time_unit)

    def get_sample_window(self, index_start, index_end):
        subsample_data = self.data[:, :, :, index_start:index_end]
        if subsample_data.ndim == 3:
            subsample_data = numpy.expand_dims(subsample_data, 3)
        return Timeseries(subsample_data, self.dimension_labels, self.time_start, self.time_step, self.time_unit)

    def get_sample_window_by_percentile(self, percentile_start, percentile_end):
        pass

    def __getattr__(self, attr_name):
        state_variables_keys = []
        if TimeseriesDimensions.STATE_VARIABLES.value in self.dimension_labels.keys():
            state_variables_keys = self.dimension_labels[TimeseriesDimensions.STATE_VARIABLES.value]
            if attr_name in self.dimension_labels[TimeseriesDimensions.STATE_VARIABLES.value]:
                return self.get_state_variable(attr_name)
        space_keys = []
        if (TimeseriesDimensions.SPACE.value in self.dimension_labels.keys()):
            space_keys = self.dimension_labels[TimeseriesDimensions.SPACE.value]
            if attr_name in self.dimension_labels[TimeseriesDimensions.SPACE.value]:
                return self.get_subspace_by_labels([attr_name])
        self.logger.error(
            "Attribute %s is not defined for this instance! You can use the folllowing labels: state_variables = %s and space = %s" %
            (attr_name, state_variables_keys, space_keys))
        raise AttributeError

    def _get_index_for_slice_label(self, slice_label, slice_idx):
        if slice_idx == 1:
            return self._get_indices_for_labels([slice_label])[0]
        if slice_idx == 2:
            return self._get_index_of_state_variable(slice_label)

    def _check_for_string_slice_indices(self, current_slice, slice_idx):
        slice_label1 = current_slice.start
        slice_label2 = current_slice.stop

        if isinstance(slice_label1, basestring):
            slice_label1 = self._get_index_for_slice_label(slice_label1, slice_idx)
        if isinstance(slice_label2, basestring):
            slice_label2 = self._get_index_for_slice_label(slice_label2, slice_idx)

        return slice(slice_label1, slice_label2, current_slice.step)

    def _get_string_slice_index(self, current_slice_string, slice_idx):
        return self._get_index_for_slice_label(current_slice_string, slice_idx)

    def __getitem__(self, slice_tuple):
        slice_list = []
        for idx, current_slice in enumerate(slice_tuple):
            if isinstance(current_slice, slice):
                slice_list.append(self._check_for_string_slice_indices(current_slice, idx))
            else:
                if isinstance(current_slice, basestring):
                    slice_list.append(self._get_string_slice_index(current_slice, idx))
                else:
                    slice_list.append(current_slice)

        return self.data[tuple(slice_list)]

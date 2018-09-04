from collections import OrderedDict

import numpy as np
from tvb_fit.tvb_epilepsy.base.constants.model_inversion_constants import BIPOLAR, OBSERVATION_MODELS
from tvb_fit.base.utils.log_error_utils import initialize_logger, warning, raise_error
from tvb_fit.base.utils.data_structures_utils import formal_repr, ensure_list, isequal_string, find_labels_inds
from tvb_fit.base.computations.math_utils import select_greater_values_array_inds
from tvb_fit.service.head_service import HeadService
from tvb_fit.service.timeseries_service import TimeseriesService


#TODO: Would it make sense to have this in tvb_fit?
class ModelInversionService(object):

    logger = initialize_logger(__name__)

    active_regions_selection_methods = ["E", "LSA", "sensors"]
    active_regions_exlude = []
    active_e_th = 0.1
    active_x0_th = 0.1
    active_lsa_th = None

    def __init__(self):
        self.logger.info("Model Inversion Service instance created!")

    def _repr(self, d=OrderedDict()):
        for ikey, (key, val) in enumerate(self.__dict__.items()):
            d.update({key:  val})
        return d

    def __repr__(self, d=OrderedDict()):
        return formal_repr(self, self._repr(d))

    def __str__(self):
        return self.__repr__()

    def exclude_regions(self, active_regions):
        new_active_regions = []
        for region in active_regions:
            if region not in self.active_regions_exlude:
                new_active_regions.append(region)
        return new_active_regions

    def update_active_regions_e_values(self, probabilistic_model, e_values, reset=False):
        active_regions = probabilistic_model.active_regions.tolist()
        if reset:
            active_regions = []
        if len(e_values) > 0:
            active_regions += select_greater_values_array_inds(e_values, self.active_e_th).tolist()
            active_regions = self.exclude_regions(active_regions)
            probabilistic_model.update_active_regions(active_regions)
        else:
            warning("Skipping active regions setting by E values because no such values were provided!")
        return probabilistic_model

    def update_active_regions_x0_values(self, probabilistic_model, x0_values, reset=False):
        active_regions = probabilistic_model.active_regions.tolist()
        if reset:
            active_regions = []
        if len(x0_values) > 0:
            active_regions += select_greater_values_array_inds(x0_values, self.active_x0_th).tolist()
            active_regions = self.exclude_regions(active_regions)
            probabilistic_model.update_active_regions(active_regions)
        else:
            warning("Skipping active regions setting by x0 values because no such values were provided!")
        return probabilistic_model

    def update_active_regions_lsa(self, probabilistic_model, lsa_propagation_strengths, reset=False):
        active_regions = probabilistic_model.active_regions.tolist()
        if reset:
            active_regions = []
        if len(lsa_propagation_strengths) > 0:
            ps_strengths = lsa_propagation_strengths / np.max(lsa_propagation_strengths)
            active_regions += select_greater_values_array_inds(ps_strengths,self.active_lsa_th).tolist()
            active_regions = self.exclude_regions(active_regions)
            probabilistic_model.update_active_regions(active_regions)
        else:
            self.logger.warning("No LSA results found (empty propagations_strengths vector)!" +
                                "\nSkipping of setting active regions according to LSA!")
        return probabilistic_model

    def update_active_regions(self, probabilistic_model, e_values=[], x0_values=[],
                              lsa_propagation_strengths=[], reset=False):
        if reset:
            probabilistic_model.update_active_regions([])
        for m in ensure_list(self.active_regions_selection_methods):
            if isequal_string(m, "E"):
                probabilistic_model = self.update_active_regions_e_values(probabilistic_model, e_values, reset=False)
            elif isequal_string(m, "x0"):
                probabilistic_model = self.update_active_regions_x0_values(probabilistic_model, x0_values, reset=False)
            elif isequal_string(m, "LSA"):
                probabilistic_model = self.update_active_regions_lsa(probabilistic_model, lsa_propagation_strengths,
                                                                     reset=False)
        return probabilistic_model


class ODEModelInversionService(ModelInversionService):

    active_seeg_th = None
    bipolar = BIPOLAR
    manual_selection = []
    auto_selection = "power"  # auto_selection=False,
    power_th = None
    gain_matrix_th = None
    gain_matrix_percentile = 99.0
    n_signals_per_roi = 1
    normalization = "baseline-amplitude"
    decim_ratio = 1
    cut_target_data_tails = [0, 0]
    sensors_per_electrode = 2
    group_electrodes = True

    def __init__(self):
        super(ODEModelInversionService, self).__init__()
        self.ts_service = TimeseriesService()

    def update_active_regions_sensors(self, probabilistic_model, sensors, reset=False):
        active_regions = probabilistic_model.active_regions.tolist()
        if reset:
            active_regions = []
        if sensors is not None:
            active_regions += sensors.get_stronger_gain_matrix_inds(self.gain_matrix_th,
                                                                    self.gain_matrix_percentile)[1].tolist()
            active_regions = self.exclude_regions(active_regions)
            probabilistic_model.update_active_regions(active_regions)
        else:
            self.logger.warning("No LSA results found (empty propagations_strengths vector)!" +
                                "\nSkipping of setting active regions according to LSA!")
        return probabilistic_model

    def update_active_regions_target_data(self, target_data, probabilistic_model, sensors, reset=False):
        if reset:
            probabilistic_model.update_active_regions([])
        if target_data:
            active_regions = probabilistic_model.active_regions.tolist()
            gain_matrix = np.array(sensors.gain_matrix)
            signals_inds = sensors.get_sensors_inds_by_sensors_labels(target_data.space_labels)
            if len(signals_inds) != 0:
                gain_matrix = gain_matrix[signals_inds]
                for proj in gain_matrix:
                    active_regions += select_greater_values_array_inds(proj, self.gain_matrix_th,
                                                                       self.n_signals_per_roi).tolist()
                active_regions = self.exclude_regions(active_regions)
                probabilistic_model.update_active_regions(active_regions)
            else:
                warning("Skipping active regions setting by seeg power because no data were assigned to sensors!")
        else:
            warning("Skipping active regions setting by seeg power because no target data were provided!")
        return probabilistic_model, sensors.gain_matrix[signals_inds][:, probabilistic_model.active_regions]

    def update_active_regions(self, probabilistic_model, sensors=None, target_data=None, e_values=[], x0_values=[],
                              lsa_propagation_strengths=[], reset=False):
        if reset:
            probabilistic_model.update_active_regions([])
        probabilistic_model = \
            super(ODEModelInversionService, self).update_active_regions(probabilistic_model, e_values, x0_values,
                                                                        lsa_propagation_strengths, reset=False)
        if sensors is None:
            gain_matrix = None
        else:
            if "sensors" in self.active_regions_selection_methods:
                probabilistic_model = self.update_active_regions_sensors(probabilistic_model, sensors, reset=False)
            gain_matrix = sensors.gain_matrix[:, probabilistic_model.active_regions]
            if target_data is not None:
                signals_inds = sensors.get_sensors_inds_by_sensors_labels(target_data.space_labels)
                gain_matrix = gain_matrix[signals_inds]
                if "target_data" in self.active_regions_selection_methods:
                    probabilistic_model, gain_matrix = \
                        self.update_active_regions_target_data(target_data, probabilistic_model, sensors, reset=False)
        return probabilistic_model, gain_matrix

    def select_target_data_sensors(self, target_data, sensors, rois, power=np.array([]),
                                   n_groups=None, members_per_group=None):
        if n_groups is None:
            n_groups = sensors.number_of_electrodes * self.sensors_per_electrode
        if members_per_group is None:
            if n_groups > sensors.number_of_electrodes:
                members_per_group = 1
            else:
                members_per_group = self.sensors_per_electrode
        if self.auto_selection.find("rois") >= 0:
            signals_inds = []
            if self.auto_selection.find("power"):
                self.auto_selection = self.auto_selection.replace("power", "")
                signals_inds = self.ts_service.select_by_power(target_data, power, self.power_th)[1].tolist()
            if sensors.gain_matrix is not None:
                signals_inds += \
                    self.ts_service.select_by_rois_proximity(target_data, sensors.gain_matrix.T[rois],
                                                             self.gain_matrix_th, n_signals=self.n_signals_per_roi)[1].\
                        tolist()
            target_data = target_data.get_subspace_by_index(np.unique(signals_inds))
        if self.auto_selection.find("correlation-power") >= 0 and target_data.number_of_labels > 1:
            if self.group_electrodes:
                disconnectivity = HeadService().sensors_in_electrodes_disconnectivity(sensors, target_data.space_labels)
            target_data, _ = self.ts_service.select_by_correlation_power(target_data, disconnectivity=disconnectivity,
                                                                         n_groups=n_groups,
                                                                         members_per_group=members_per_group)
        elif self.auto_selection.find("gain-power") >= 0 and target_data.number_of_labels > 1:
            if self.group_electrodes:
                disconnectivity = HeadService().sensors_in_electrodes_disconnectivity(sensors, target_data.space_labels)
            signals_inds = sensors.get_sensors_inds_by_sensors_labels(target_data.space_labels)
            target_data, _ = \
                self.ts_service.select_by_gain_matrix_power(target_data, sensors.gain_matrix[signals_inds],
                                                            disconnectivity=disconnectivity, n_groups=n_groups,
                                                            members_per_group=members_per_group)
        elif self.auto_selection.find("power") >= 0:
            target_data, _ = self.ts_service.select_by_power(target_data, power, self.power_th)
        return target_data

    def set_gain_matrix(self, target_data, probabilistic_model, sensors=None):
        if probabilistic_model.observation_model in OBSERVATION_MODELS.SEEG.value:
            signals_inds = sensors.get_sensors_inds_by_sensors_labels(target_data.space_labels)
            gain_matrix = np.array(sensors.gain_matrix[signals_inds][:, probabilistic_model.active_regions])
        else:
            gain_matrix = np.eye(target_data.number_of_labels)
        return gain_matrix

    def set_target_data_and_time(self, target_data, probabilistic_model, head=None, sensors=None, sensor_id=0,
                                 power=np.array([])):
        if sensors is None and head is not None:
            try:
                sensors = sensors.head.get_sensors_id(sensor_ids=sensor_id)
            except:
                if probabilistic_model.observation_model in OBSERVATION_MODELS.SEEG.value:
                    raise_error("No sensors instance! Needed for gain_matrix computation!")
                else:
                    pass
        if len(self.manual_selection) > 0:
            target_data = target_data.get_subspace_by_index(self.manual_selection)
        if self.auto_selection:
            if probabilistic_model.observation_model in OBSERVATION_MODELS.SEEG.value:
                target_data = self.select_target_data_sensors(target_data, sensors,
                                                              probabilistic_model.active_regions, power)
            else:
                target_data = target_data.get_subspace_by_index(probabilistic_model.active_regions)
        if self.decim_ratio > 1:
            target_data = self.ts_service.decimate(target_data, self.decim_ratio)
        if np.any(np.array(self.cut_target_data_tails)):
            target_data = target_data.get_time_window(np.maximum(self.cut_target_data_tails[0], 0),
                                                      target_data.time_length -
                                                        np.maximum(self.cut_target_data_tails[1], 0))
        if self.bipolar:
            target_data = target_data.get_bipolar()
        # TODO: decide about target_data' normalization for the different (sensors', sources' cases)
        if self.normalization:
            target_data = self.ts_service.normalize(target_data, self.normalization)
        probabilistic_model.time = target_data.time_line
        probabilistic_model.time_length = len(probabilistic_model.time)
        probabilistic_model.number_of_target_data = target_data.number_of_labels
        return target_data, probabilistic_model, self.set_gain_matrix(target_data, probabilistic_model, sensors)


class SDEModelInversionService(ODEModelInversionService):

    def __init__(self):
        super(SDEModelInversionService, self).__init__()

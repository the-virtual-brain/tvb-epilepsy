from collections import OrderedDict

import numpy as np
from tvb_epilepsy.base.constants.model_inversion_constants import *
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr, ensure_list, assert_arrays
from tvb_epilepsy.base.computations.math_utils import select_greater_values_array_inds
from tvb_epilepsy.base.epileptor_models import *
from tvb_epilepsy.service.head_service import HeadService
from tvb_epilepsy.service.timeseries_service import TimeseriesService
from tvb_epilepsy.plot.plotter import Plotter


class ModelInversionService(object):

    logger = initialize_logger(__name__)

    active_regions_selection_methods = ["E", "LSA"]
    active_regions_th = 0.1

    def __init__(self, **kwargs):
        self.logger.info("Model Inversion Service instance created!")

    def _repr(self, d=OrderedDict()):
        for ikey, (key, val) in enumerate(self.__dict__.iteritems()):
            d.update({key:  val})
        return d

    def __repr__(self, d=OrderedDict()):
        return formal_repr(self, self._repr(d))

    def __str__(self):
        return self.__repr__()

    def update_active_regions_e_values(self, stats_model, e_values, active_regions_th=0.1, reset=False):
        if reset:
            stats_model.update_active_regions([])
        if len(e_values) > 0:
            stats_model.update_active_regions(stats_model.active_regions +
                                              select_greater_values_array_inds(e_values, active_regions_th).tolist())
        return stats_model

    def update_active_regions_x0_values(self, stats_model, x0_values, active_regions_th=0.1, reset=False):
        if reset:
            stats_model.update_active_regions([])
        if len(x0_values) > 0:
            stats_model.update_active_regions(stats_model.active_regions +
                                          select_greater_values_array_inds(x0_values, active_regions_th).tolist())
        return stats_model

    def update_active_regions_lsa(self, stats_model, lsa_propagation_strengths, active_regions_th=None, reset=False):
        if reset:
            stats_model.update_active_regions([])
        if len(lsa_propagation_strengths) > 0:
            ps_strengths = lsa_propagation_strengths / np.max(lsa_propagation_strengths)
            stats_model.update_active_regions(stats_model.active_regions +
                                              select_greater_values_array_inds(ps_strengths,
                                                                               active_regions_th).tolist())
        else:
            self.logger.warning("No LSA results found (empty propagations_strengths vector)!" +
                                "\nSkipping of setting active_regions according to LSA!")
        return stats_model

    def update_active_regions(self, stats_model, reset=False, **kwargs):
        if reset:
            stats_model.update_active_regions([])
        for m, th in zip(*assert_arrays([ensure_list(self.active_regions_selection_methods), \
                                         ensure_list(self.active_regions_th)])):
            if isequal_string(m, "E"):
                stats_model = self.update_active_regions_e_values(stats_model, kwargs.get("e_values", []), th)
            elif isequal_string(m, "x0"):
                stats_model = self.update_active_regions_x0_values(stats_model, kwargs.get("x0_values", []), th)
            elif isequal_string(m, "LSA"):
                stats_model = self.update_active_regions_lsa(stats_model,
                                                             kwargs.get("lsa_propagation_strength", []), th)
        return stats_model


class ODEModelInversionService(ModelInversionService):
    
    bipolar = BIPOLAR
    manual_selection = []
    auto_selection = "power"  # auto_selection=False,
    normalization = "baseline-amplitude"
    decim_ratio = 1
    cut_target_data_tails = [0, 0]
    n_electrodes = 10
    sensors_per_electrode = 1
    group_electrodes = True
    plotter = Plotter()
    
    def __init__(self, **kwargs):
        super(ODEModelInversionService, self).__init__(**kwargs)
        self.ts_service = TimeseriesService()

    def update_active_regions_seeg(self, target_data, stats_model, sensors, active_regions_th=None, reset=False):
        if reset:
            stats_model.update_active_regions([])
        active_regions = stats_model.active_regions
        gain_matrix = numpy.array(sensors.gain_matrix)
        seeg_inds = sensors.get_sensors_inds_by_sensors_labels(target_data.space_labels)
        if len(seeg_inds) != 0:
            gain_matrix = gain_matrix[seeg_inds]
        for proj in gain_matrix:
            active_regions += select_greater_values_array_inds(proj, active_regions_th).tolist()
            stats_model.update_active_regions(active_regions)
        return stats_model

    def update_active_regions(self, stats_model, reset=False, **kwargs):
        if reset:
            stats_model.update_active_regions([])
        active_regions_th = ensure_list(self.active_regions_th)
        if "seeg" in self.active_regions_selection_methods:
            seeg_index = self.active_regions_selection_methods.index("seeg")
            if len(active_regions_th) == len(self.active_regions_selection_methods):
                seeg_th = active_regions_th.pop(seeg_index)
            else:
                seeg_th = active_regions_th[0]
            seeg_index = self.active_regions_selection_methods.pop(seeg_index)
        else:
            seeg_index = -1
        stats_model = \
            super(ODEModelInversionService, self).update_active_regions(stats_model, reset=False, **kwargs)
        if seeg_index >= 0:
            stats_model = self.update_active_regions_seeg(stats_model, kwargs.get("sensors", None), seeg_th)
        return stats_model

    def select_target_data_seeg(self, target_data, sensors, rois, **kwargs):
        if self.auto_selection.find("rois") >= 0:
            if sensors.gain_matrix is not None:
                target_data = self.ts_service.select_by_rois_proximity(target_data, sensors.gain_matrix.T[rois],
                                                                   kwargs.get("gain_matrix_th", None))
        if self.auto_selection.find("correlation-power") >= 0 and target_data.number_of_labels > 1:
            if kwargs.get("group_electrodes", True):
                disconnectivity = HeadService().sensors_in_electrodes_disconnectivity(sensors, target_data.space_labels)
            target_data = self.ts_service.select_by_correlation_power(target_data, disconnectivity=disconnectivity,
                                                                      n_groups=kwargs.get("n_electrodes"),
                                                                      members_per_group=
                                                                                 kwargs.get("sensors_per_electrode", 1))
        elif self.auto_selection.find("power"):
            target_data = self.ts_service.select_by_power(target_data,  kwargs.get("power", np.array([])),
                                                      kwargs.get("power_th", None))
        return target_data

    def select_target_data_source(self, target_data, rois, **kwargs):
        if self.auto_selection.find("rois") >= 0:
            head = kwargs.get("head")
            target_data = self.ts_service.select_by_rois(target_data, rois, head.connectivity.region_labels)
        if self.auto_selection.find("power") >= 0:
            target_data = self.ts_service.select_by_power(target_data, kwargs.get("power", np.array([])),
                                                          kwargs.get("power_th", None))
        return target_data

    def set_gain_matrix(self, target_data, stats_model, sensors=None):
        if stats_model.observation_model in OBSERVATION_MODELS.SEEG.value:
            signals_inds = sensors.get_sensors_inds_by_sensors_labels(target_data.space_labels)
            gain_matrix = np.array(sensors.gain_matrix[signals_inds], stats_model.active_regions)
        else:
            gain_matrix = np.ones((target_data.number_of_labels, target_data.number_of_labels))
        return gain_matrix

    def set_target_data_and_time(self, target_data, stats_model, sensors=None, **kwargs):
        if len(self.manual_selection) > 0:
            target_data = target_data.get_subspace_by_index(self.manual_selection)
        if self.auto_selection:
            if stats_model.observation_model in OBSERVATION_MODELS.SEEG.value:
                target_data = self.select_target_data_seeg(target_data, sensors,
                                                           stats_model.active_regions, **kwargs)
            else:
                target_data = self.select_target_data_source(target_data, stats_model.active_regions,  **kwargs)

        if self.decim_ratio > 1:
            target_data = self.ts_service.decimate(target_data, self.decim_ratio)
        if np.any(self.cut_target_data_tails > 0):
            target_data = target_data.get_time_window_by_units(self.cut_target_data_tails[0], 
                                                               target_data.time_length-self.cut_target_data_tails[1])
        if self.bipolar:
            target_data = target_data.get_bipolar()
        # TODO: decide about target_data' normalization for the different (sensors', sources' cases)
        if self.normalization:
            target_data = self.ts_service.normalize(target_data, self.normalization)
        stats_model.time = target_data.time_line
        stats_model.dt = target_data.time_step
        stats_model.number_of_target_data = target_data.number_of_labels
        return target_data, stats_model, self.set_gain_matrix(target_data, stats_model, sensors)


class SDEModelInversionService(ODEModelInversionService):

    def __init__(self, **kwargs):
        super(SDEModelInversionService, self).__init__(**kwargs)

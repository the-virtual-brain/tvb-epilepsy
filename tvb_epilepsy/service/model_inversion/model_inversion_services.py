
import numpy as np
from tvb_epilepsy.base.constants.model_inversion_constants import *
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list, sort_dict, assert_arrays, extract_dict_stringkeys
from tvb_epilepsy.base.computations.math_utils import select_greater_values_array_inds
from tvb_epilepsy.base.epileptor_models import *
from tvb_epilepsy.service.head_service import HeadService
from tvb_epilepsy.service.signal_processor import decimate_signals, cut_signals_tails, normalize_signals
from tvb_epilepsy.top.scripts.fitting_data_scripts import prepare_seeg_observable, prepare_signal_observable


class ModelInversionService(object):

    logger = initialize_logger(__name__)

    number_of_regions = 0
    target_data_type = ""

    def __init__(self, number_of_regions, **kwargs):
        self.number_of_regions =number_of_regions
        self.target_data_type = kwargs.get("target_data_type", "")
        self.logger.info("Model Inversion Service instance created!")

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

    def update_active_regions(self, stats_model, methods=["E", "x0", "LSA"], reset=False, **kwargs):
        if reset:
            stats_model.update_active_regions([])
        for m, th in zip(*assert_arrays([ensure_list(methods),
                                         ensure_list(kwargs.get("active_regions_th", None))])):
            if isequal_string(m, "E"):
                stats_model = self.update_active_regions_e_values(stats_model, kwargs.get("e_values", []), th)
            elif isequal_string(m, "x0"):
                stats_model = self.update_active_regions_x0_values(stats_model, kwargs.get("x0_values", []), th)
            elif isequal_string(m, "LSA"):
                stats_model = self.update_active_regions_lsa(stats_model,
                                                             kwargs.get("lsa_propagation_strength", []), th)
        return stats_model


class ODEModelInversionService(ModelInversionService):

    signals_inds = []

    def __init__(self, number_of_regions, **kwargs):
        super(ODEModelInversionService, self).__init__(number_of_regions, **kwargs)
        self.signals_inds = range(self.number_of_regions)

    def update_active_regions_seeg(self, stats_model, gain_matrix, active_regions_th=None, seeg_inds=[], reset=False):
        if reset:
            stats_model.update_active_regions([])
        active_regions = stats_model.active_regions
        if gain_matrix is not None:
            if len(seeg_inds) == 0:
                seeg_inds = self.signals_inds
                if len(seeg_inds) != 0:
                    gain_matrix = gain_matrix[seeg_inds]
                for proj in gain_matrix:
                    active_regions += select_greater_values_array_inds(proj, active_regions_th).tolist()
                    stats_model.update_active_regions(active_regions)
        else:
            self.logger.warning(
                "Projection is not found!" + "\nSkipping of setting active_regios according to SEEG power!")
        return stats_model

    def update_active_regions(self, stats_model, methods=["E", "x0", "LSA"], reset=False, **kwargs):
        if reset:
            stats_model.update_active_regions([])
        for m, th in zip(*assert_arrays([ensure_list(methods),
                                         ensure_list(kwargs.get("active_regions_th", None))])):
            if isequal_string(m, "E"):
                stats_model = self.update_active_regions_e_values(stats_model, kwargs.get("e_values", []), th)
            elif isequal_string(m, "x0"):
                stats_model = self.update_active_regions_x0_values(stats_model, kwargs.get("x0_values", []), th)
            elif isequal_string(m, "LSA"):
                stats_model = self.update_active_regions_lsa(stats_model,
                                                             kwargs.get("lsa_propagation_strength", []), th)
            elif isequal_string(m, "seeg"):
                stats_model = self.update_active_regions_seeg(stats_model, kwargs.get("gain_matrix", None), th,
                                                              seeg_inds=kwargs.get("seeg_inds", []))
        return stats_model


    def select_signals_seeg(self, signals, sensors, rois, auto_selection, **kwargs):
        inds = range(signals.shape[1])
        head_service = HeadService()
        if auto_selection.find("rois") >= 0:
            if sensors.gain_matrix is not None:
                current_selection = head_service.select_sensors_rois(sensors, kwargs.get("rois", rois),
                                                                     self.signals_inds,
                                                                     kwargs.get("gain_matrix_th", None))
                inds = np.where([s in current_selection for s in self.signals_inds])[0]
                self.signals_inds = np.array(self.signals_inds)[inds].tolist()
                signals = signals[:, inds]
        if auto_selection.find("correlation-power") >= 0:
            power = kwargs.get("power", np.sum((signals - np.mean(signals, axis=0)) ** 2, axis=0) / signals.shape[0])
            correlation = kwargs.get("correlation", np.corrcoef(signals.T))
            current_selection = head_service.select_sensors_corr(sensors, correlation, self.signals_inds, power=power,
                                                                 n_electrodes=kwargs.get("n_electrodes"),
                                                                 sensors_per_electrode=kwargs.get(
                                                                     "sensors_per_electrode", 1),
                                                                 group_electrodes=kwargs.get("group_electrodes", True))
            inds = np.where([s in current_selection for s in self.signals_inds])[0]
            self.signals_inds = np.array(self.signals_inds)[inds].tolist()
        elif auto_selection.find("power"):
            power = kwargs.get("power", np.sum(signals ** 2, axis=0) / signals.shape[0])
            inds = select_greater_values_array_inds(power, kwargs.get("power_th", None))
            self.signals_inds = (np.array(self.signals_inds)[inds]).tolist()
        return signals[:, inds]

    def select_signals_source(self, signals, rois, auto_selection, **kwargs):
        if auto_selection.find("rois") >= 0:
            if kwargs.get("rois", rois):
                inds = np.where([s in rois for s in self.signals_inds])[0]
                signals = signals[:, inds]
                self.signals_inds = np.array(self.signals_inds)[inds].tolist()
        if auto_selection.find("power") >= 0:
            power = kwargs.get("power", np.sum((signals - np.mean(signals, axis=0)) ** 2, axis=0) / signals.shape[0])
            inds = select_greater_values_array_inds(power, kwargs.get("power_th", None))
            signals = signals[:, inds]
            self.signals_inds = (np.array(self.signals_inds)[inds]).tolist()
        return signals

    def set_empirical_target_data(self, target_data, **kwargs):
        self.target_data_type = "empirical_seeg"
        if isinstance(target_data, dict):
            signals = np.array(target_data.get("signals", target_data.get("target_data", None)))
        else:
            signals = np.array(target_data)
        manual_selection = kwargs.get("manual_selection", [])
        if len(manual_selection) > 0:
            self.signals_inds = manual_selection
        else:
            self.signals_inds = range(signals.shape[1])
        if len(self.signals_inds) < signals.shape[1]:
            signals = signals[:, self.signals_inds]
        (time_length, number_of_signals) = signals.shape
        return signals, number_of_signals, time_length

    def set_simulated_target_data(self, target_data, stats_model, dynamical_model, **kwargs):
        self.signals_inds = range(self.number_of_regions)
        self.target_data_type = "simulated_source"
        signals = np.array([])
        time = target_data["time"].flatten()
        dt = np.diff(time).mean()
        signals_labels = kwargs.get("signals_labels")
        if stats_model.observation_model.value in OBSERVATION_MODELS.SEEG.value:
            self.target_data_type = "simulated_seeg"
            sensors = kwargs.get("sensors")
            gain_matrix = sensors.gain_matrix
            self.signals_inds = range(gain_matrix.shape[0])
            if not stats_model.observation_model is OBSERVATION_MODELS.SEEG_LOGPOWER:
                signals = extract_dict_stringkeys(sort_dict(target_data), kwargs.get("seeg_dataset", "SEEG0"),
                                                  modefun="find", two_way_search=True, break_after=1)
                if len(signals) > 0:
                    signals = signals.values()[0]
            if signals.size == 0:
                signals = np.array(target_data.get("source", target_data["x1"]))
                if stats_model.observation_model is OBSERVATION_MODELS.SEEG_LOGPOWER:
                    signals = np.log(np.dot(gain_matrix, np.exp(signals.T))).T
                else:
                    signals = (np.dot(gain_matrix, signals.T)).T
            signals, time, self.signals_inds, signals_labels = \
                prepare_seeg_observable(signals, time, dynamical_model,
                                        kwargs.get("times_on_off", [time[0], time[-1]]),
                                        signals_labels, kwargs.get("manual_selection", []),
                                        win_len_ratio=kwargs.get("win_len_ratio", WIN_LEN_RATIO),
                                        low_freq=kwargs.get("low_freq", LOW_FREQ),
                                        high_freq=kwargs.get("high_freq", HIGH_FREQ),
                                        bipolar=kwargs.get("bipolar", BIPOLAR),
                                        log_flag=kwargs.get("log_flag", LOG_FLAG),
                                        plotter=kwargs.get("plotter", False))
        else:
            # if statistical_model.observation_expression == "x1z_offset":
            #     signals = ((target_data["x1"].T - np.expand_dims(self.x1eq, 1)).T +
            #                (target_data["z"].T - np.expand_dims(self.zeq, 1)).T) / 2.75
            #     # TODO: a better normalization
            # elif statistical_model.observation_expression == "x1_offset":
            #     # TODO: a better normalization
            #     signals = (target_data["x1"].T - np.expand_dims(self.x1eq, 1)).T / 2.0
            # else: # statistical_models.observation_expression == "source"
            signals = np.array(target_data.get("source", target_data["x1"]))
            signals, time, self.signals_inds = \
                prepare_signal_observable(signals, time, dynamical_model,
                                          kwargs.get("times_on_off", [time[0], time[-1]]),
                                          signals_labels, rois=kwargs.get("manual_selection", []),
                                          win_len_ratio=kwargs.get("win_len_ratio", WIN_LEN_RATIO),
                                          low_freq=kwargs.get("low_freq", LOW_FREQ),
                                          high_freq=kwargs.get("high_freq", HIGH_FREQ),
                                          log_flag=kwargs.get("log_flag", LOG_FLAG),
                                          plotter=kwargs.get("plotter", False))[:3]
            target_data["signals"] = np.array(signals)
        (time_length, number_of_signals) = signals.shape
        return signals, target_data, signals_labels, number_of_signals, time, time_length, dt

    def normalize_signals(self, signals, normalization=None):
        return normalize_signals(signals, normalization)

    def set_target_data_and_time(self, target_data, stats_model, dynamical_model, **kwargs):
        if self.target_data_type.lower().find("simul") > -1:
            signals, target_data, signals_labels, number_of_signals, time, time_length, dt = \
                self.set_simulated_target_data(target_data, stats_model, dynamical_model, **kwargs)
        else:  # isequal_string(target_data_type, "empirical"):
            signals, number_of_signals, time_length = self.set_empirical_target_data(target_data, **kwargs)
            dt = kwargs.get("dt", 1.0)
            time = target_data.get("time", np.arange(stats_model.dt * (time_length - 1)))
        if kwargs.get("auto_selection", True) is not False:
            if stats_model.observation_model not in OBSERVATION_MODELS.SEEG.value:
                signals = self.select_signals_source(signals, stats_model.active_regions,
                                                     kwargs.pop("auto_selection", "rois"), **kwargs)
            else:
                signals = self.select_signals_seeg(signals, stats_model.active_regions,
                                                   kwargs.pop("auto_selection", "rois-correlation-power"), **kwargs)
        if kwargs.get("decimate", 1) > 1:
            signals, time, dt, time_length = decimate_signals(signals, time, kwargs.get("decimate"))
        if np.sum(kwargs.get("cut_signals_tails", (0, 0))) > 0:
            signals, time, time_length = cut_signals_tails(signals, time, kwargs.get("cut_signals_tails"))
        # TODO: decide about signals' normalization for the different (sensors', sources' cases)
        signals = self.normalize_signals(signals, kwargs.get("normalization", None))
        stats_model.number_of_signals = number_of_signals
        stats_model.time_length = time_length
        stats_model.dt = dt
        stats_model.time = time
        return signals, time, stats_model, signals_labels, target_data


class SDEModelInversionService(ODEModelInversionService):

    def __init__(self, number_of_regions, **kwargs):
        super(SDEModelInversionService, self).__init__(number_of_regions, **kwargs)
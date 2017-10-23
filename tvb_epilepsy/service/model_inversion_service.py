import numpy as np

from tvb_epilepsy.base.constants import X1_EQ_CR_DEF, X1_DEF, X0_DEF, X0_CR_DEF
from tvb_epilepsy.base.utils import raise_value_error, formal_repr, sort_dict, ensure_list, initialize_logger, \
                                    compute_projection, warning
from tvb_epilepsy.base.computations.calculations_utils import calc_x0cr_r
from tvb_epilepsy.base.model.model_vep import Head, Sensors
from tvb_epilepsy.base.model.model_configuration import ModelConfiguration
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.service.epileptor_model_factory import model_noise_intensity_dict
from tvb_epilepsy.tvb_api.epileptor_models import *
from tvb_epilepsy.custom.simulator_custom import EpileptorModel
from tvb.simulator.models import Epileptor


AVAILABLE_DYNAMICAL_MODELS = (Epileptor, EpileptorModel, EpileptorDP2D, EpileptorDP, EpileptorDPrealistic)


LOG = initialize_logger(__name__)


class ModelInversionService(object):

    def __init__(self, model_configuration, hypothesis, target_data, active_regions=None,  active_regions_th=0.1,
                 dynamical_model=None, head=None,  time=None,
                 observation_model=1, observation_expression=1, sensors=None, channel_inds=None, mixing=None,
                 euler_method=-1, logger=LOG, **kwargs):

        self.euler_method = euler_method
        self.observation_model = observation_model
        self.obervation_expression = observation_expression

        logger.info("Checking input model configuration...")
        if not(isinstance(model_configuration, ModelConfiguration)):
            raise_value_error("Input model configuration is not a ModelConfiguration object:\n"
                              + str(model_configuration))

        logger.info("Checking input hypothesis...")
        if not(isinstance(hypothesis, DiseaseHypothesis)):
            raise_value_error("Input hypothesis is not a DiseaseHypothesis object:\n" + str(hypothesis))

        logger.info("Setting (non) active regions...")
        self.n_regions = self.hypothesis.number_of_regions
        active_regions_flag = np.zeros((self.n_regions,), dtype="i")
        self.active_regions_th = active_regions_th
        if active_regions is None:
            # Initialize as all those regions whose equilibria lie further away from the healthy equilibrium:
            self.active_regions = np.where(model_configuration.e_values > self.active_regions_th)[0]
            # If LSA has been run, add all regions with a propagation strength greater than the minimal one:
            if len(hypothesis.propagation_strengths) > 0:
                self.active_regions = np.unique(self.active_regions.tolist() +
                                                np.where(hypothesis.propagation_strengths /
                                                         np.max(hypothesis.propagation_strengths)
                                                         > active_regions_th)[0].tolist())
            else:
                self.active_regions = active_regions
        self.active_regions_flag[self.active_regions] = 1
        self.n_active_regions = len(self.active_regions)
        self.nonactive_regions = np.where(1 - self.active_regions_flag)[0]
        self.n_nonactive_regions = len(self.nonactive_regions)

        logger.info("Setting epileptor parameters...")
        self.epileptor_params = {}
        for p in ["K", "a", "b", "d", "yc", "Iext1", "slope"]:
            temp = getattr(model_configuration, p)
            if isinstance(temp, (np.ndarray, list)):
                if np.all(temp[0], np.array(temp)):
                    temp = temp[0]
                else:
                    raise_not_implemented_error("Statistical models where not all regions have the same value " +
                                                " for parameter " + p + " are not implemented yet!")
            self.epileptor_params.update({p: temp})
        x0cr, rx0 = calc_x0cr_r(self.epileptor_params["yc"], self.epileptor_params["Iext1"], self.epileptor_params["a"],
                                self.epileptor_params["b"], self.epileptor_params["d"], zmode=np.array("lin"),
                                x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF, x0cr_def=X0_CR_DEF, test=False,
                                shape=None, calc_mode="non_symbol")

        logger.info("Checking input dynamical model and setting time scale and noise default parameters...")
        if isinstance(dynamical_model, AVAILABLE_DYNAMICAL_MODELS):
            self.dynamical_model = dynamical_model._ui_name
            noise_intensity = kwargs.get("noise_intensity", model_noise_intensity_dict[self.dynamical_model])
            self.sig_def = np.mean(noise_intensity)
            if isinstance(self.dynamic_model, (Epileptor, EpileptorModel)):
                self.tau1_def = kwargs.get("tau1", np.mean(1.0 / dynamical_model.r))
                self.tau0_def = kwargs.get("tau0", np.mean(dynamical_model.tt))

            elif isinstance(self.dynamic_model, (EpileptorDP, EpileptorDP2D, EpileptorDPrealistic)):
                self.tau1_def = kwargs.get("tau1", np.mean(dynamical_model.tau1))
                self.tau0_def = kwargs.get("tau0", np.mean(dynamical_model.tau0))

        else:
            self.tau1_def = kwargs.get("tau1", 0.5)
            self.tau0_def = kwargs.get("tau0", 30)
            self.sig_def = kwargs.get("noise_intensity", 10 ** -4)
        self.sig_eq_def = kwargs.get("sig_eq", (X1_EQ_CR_DEF - X1_DEF) / 3.0)
        self.sig_init_def = kwargs.get("sig_init", 0.1)

        logger.info("Reading head and sensor information & setting mixing/gain matrix...")

        if isinstance(head, Head):
            self.region_labels = head.connectivity.region_labels[self.active_regions]
            if isinstance(getattr(head, "sensorsSEEG"), dict):
                self.sensor_labels = head.sensorsSEEG.keys()[0].labels[channel_inds]
                self.mixing = head.sensorsSEEG.items()[0][channel_inds][:, self.active_regions]
            elif isinstance(sensors, Sensors):
                self.sensor_labels = sensors.labels[channel_inds]
                self.mixing = sensors.calculate_projection(head.connectivity)[channel_inds][:, self.active_regions]
            else:
                self.sensor_labels = kwargs.get("sensor_labels")
                sensor_locations = kwargs.get("sensor_locations")
                if isinstance(sensor_locations, np.ndarray):
                    if sensor_locations.shape[0] > len(channel_inds):
                        sensor_locations = sensor_locations[channel_inds]
                    self.mixing = \
                        compute_projection(sensor_locations, head.connectivity.centers)[:, self.active_regions]
        else:
            self.region_labels = kwargs.get("region_labels")
            region_centers = kwargs.get("region_centers")
            if isinstance(region_centers, np.ndarray):
                if region_centers.shape[0] > self.n_active_regions:
                    region_centers = region_centers[self.active_regions]
                if isinstance(sensors, Sensors):
                    self.sensor_labels = sensors.labels[channel_inds]
                    self.mixing = compute_projection(sensors.locations, region_centers)[channel_inds]
                else:
                    self.sensor_labels = kwargs.get("sensor_labels")
                    sensor_locations = kwargs.get("sensor_locations")
                    if isinstance(sensor_locations, np.ndarray):
                        if sensor_locations.shape[0] > len(channel_inds):
                            sensor_locations = sensor_locations[channel_inds]
                        self.mixing = compute_projection(sensor_locations, region_centers)
            else:
                self.mixing = kwargs.get("mixing")
                if not (isinstance(self.mixing, np.ndarray)):
                    raise_value_error("Mixing matrix is required by observation model " + str(self.observation_model) +
                                      "but not given or computed from input!")

        logger.info("Reading and setting observation signals...")
        if isinstance(target_data, np.ndarray):
            self.signals = target_data
            self.data_type = kwargs.get("data_type", "empirical")
        elif isinstance(target_data, dict):
            self.data_type = "simulated"
            self.signals = target_data.get("signals", None)
            if self.signals is None:
                if self.observation_expression == 1:
                    self.signals = (target_data["x1"][:, self.active_regions].T -
                                    np.expand_dims(self.model_configuration.x1EQ[self.active_regions], 1)).T + \
                                   (target_data["z"][:, active_regions].T -
                                    np.expand_dims(model_configuration.zEQ[self.active_regions], 1)).T
                    # TODO: a better normalization
                    self.signals = self.signals / 2.75
                elif self.observation_expression == 2:
                    # TODO: a better normalization
                    self.signals = (target_data["x1"][:, self.active_regions].T -
                                    np.expand_dims(model_configuration.x1EQ[self.active_regions], 1)).T / 2.0
                else:
                    self.signals = target_data["x1"][:, self.active_regions]
                if observation_model != 3:
                    self.signals = (np.dot(self.mixing, self.signals.T)).T
        else:
            raise_value_error("Input target data is neither a ndarray of empirical data nor a dictionary of "
                              "simulated data:\n" + str(target_data))
        (self.n_times, self.n_signals) = self.signals

        if time is not None:
            self.time = time
            if self.n_times != len(time):
                warning("The length of the time vector doesn't match the one of observation signals!")
            self.dt = np.mean(np.diff(self.time))
        else:
            self.dt = kwargs.get("dt", 1000.0 / 1024)




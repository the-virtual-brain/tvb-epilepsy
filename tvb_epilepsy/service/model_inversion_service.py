import numpy as np

from tvb_epilepsy.base.constants import X1_EQ_CR_DEF, X1_DEF, X0_DEF, X0_CR_DEF
from tvb_epilepsy.base.utils import raise_value_error, formal_repr, sort_dict, ensure_list, initialize_logger, \
                                    compute_projection, warning
from tvb_epilepsy.base.computations.calculations_utils import calc_x0cr_r
from tvb_epilepsy.base.model.model_vep import Head, Sensors
from tvb_epilepsy.base.model.model_configuration import ModelConfiguration
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.model.statistical_model import Parameter, StatisticalModel
from tvb_epilepsy.service.epileptor_model_factory import model_noise_intensity_dict
from tvb_epilepsy.tvb_api.epileptor_models import *
from tvb_epilepsy.custom.simulator_custom import EpileptorModel
from tvb.simulator.models import Epileptor


AVAILABLE_DYNAMICAL_MODELS = (Epileptor, EpileptorModel, EpileptorDP2D, EpileptorDP, EpileptorDPrealistic)


LOG = initialize_logger(__name__)


class ModelInversionService(object):

    def __init__(self, logger=LOG):
        logger.info("Model Inversion Service instance created!")

    def configure_statistical_model_for_vep_autoregress(self, statistical_model_name, parameters, model_configuration,
                                                        hypothesis,  euler_method="backward",
                                                        observation_model="logpower",
                                                        observation_expression="x1z_offset",
                                                        active_regions=None,  active_regions_th=0.1,
                                                        dynamical_model=None, head=None, target_data=None,
                                                        logger=LOG, **kwargs):

        logger.info("Checking input dynamical model and setting time scale and noise default parameters...")
        if isinstance(dynamical_model, AVAILABLE_DYNAMICAL_MODELS):
            noise_intensity = kwargs.get("noise_intensity", model_noise_intensity_dict[dynamical_model._ui_name])
            sig_def = np.mean(noise_intensity)
            if isinstance(self.dynamic_model, (Epileptor, EpileptorModel)):
                tau1_def = kwargs.get("tau1", np.mean(1.0 / dynamical_model.r))
                tau0_def = kwargs.get("tau0", np.mean(dynamical_model.tt))

            elif isinstance(self.dynamic_model, (EpileptorDP, EpileptorDP2D, EpileptorDPrealistic)):
                tau1_def = kwargs.get("tau1", np.mean(dynamical_model.tau1))
                self.time_scales.update("tau0", kwargs.get("tau0", np.mean(dynamical_model.tau0)))

        else:
            tau1_def = kwargs.get("tau1", 0.5)
            tau0_def = kwargs.get("tau0", 30)
            sig_def = kwargs.get("noise_intensity", 10 ** -4)
        sig_eq_def = kwargs.get("sig_eq", (X1_EQ_CR_DEF - X1_DEF) / 3.0)
        sig_init_def = kwargs.get("sig_init", 0.1)

        parameters = []

        # Generative model:
        # Epileptor:
        parameters.append(Parameter("x1eq", low=kwargs.get("x1eq_lo", X1_DEF),
                                            high=kwargs.get("x1eq_hi", X1_EQ_CR_DEF),
                                            loc=model_configuration.x1EQ,
                                            scale="parameter sig_eq",
                                            shape=(hypothesis.number_of_regions,),
                                            distribution="normal"))
        parameters.append(Parameter("K", low=kwargs.get("K_lo", model_configuration.K / 10),
                                         high=kwargs.get("K_hi", 10*model_configuration.K)),
                                         loc=model_configuration.K,
                                         scale=kwargs("K_sc", model_configuration.K),
                                         shape=(1,),
                                         distribution="gamma")
        parameters.append(Parameter("tau1", low=kwargs.get("tau1_lo", 0.1),
                                            high=kwargs.get("tau1_hi", 1.0),
                                            loc=tau1_def,
                                            scale=None,
                                            shape=(1,),
                                            distribution="uniform"))
        parameters.append(Parameter("tau0", low=kwargs.get("tau0_lo", 3.0),
                                            high=kwargs.get("tau0_hi", 30000.0),
                                            loc=tau0_def,
                                            scale=kwargs.get("tau0_sc", tau0_def),
                                            shape=(1,),
                                            distribution="gamma"))
        # Coupling:
        parameters.append(Parameter("EC", low=kwargs.get("ec_lo", 10 ** -6),
                                          high=kwargs.get("ec_hi", 100.0),
                                          loc=model_configuration.connectivity_matrix,
                                          scale=kwargs.get("ec_sc", 1.0),
                                          shape=model_configuration.connectivity_matrix.shape,
                                          distribution="gamma"))
        # Integration:
        parameters.append(Parameter("sig", low=kwargs.get("sig_lo", sig_def / 10.0),
                                            high=kwargs.get("sig_hi", 10*sig_def),
                                           loc=sig_def,
                                           scale=kwargs.get("sig_sc", sig_def),
                                           shape=(1,),
                                           distribution="gamma"))
        parameters.append(Parameter("sig_eq", low=kwargs.get("sig_eq_lo", sig_eq_def / 10.0),
                                    high=kwargs.get("sig_hi", 3 * sig_eq_def),
                                    loc=sig_eq_def,
                                    scale=kwargs.get("sig_sc", sig_eq_def),
                                    shape=(1,),
                                    distribution="gamma"))
        parameters.append(Parameter("sig_init", low=kwargs.get("sig_init_lo", sig_init_def / 10.0),
                                    high=kwargs.get("sig_hi", 3 * sig_init_def),
                                    loc=sig_init_def,
                                    scale=kwargs.get("sig_sc", sig_init_def),
                                    shape=(1,),
                                    distribution="gamma"))

        # Observation model
        parameters.append(Parameter("eps", low=kwargs.get("eps_lo", 0.0),
                                    high=kwargs.get("eps_hi", 1.0),
                                    loc=kwargs.get("eps_loc", 0.1),
                                    scale=kwargs.get("sig_sc", 0.1),
                                    shape=(1,),
                                    distribution="gamma"))
        parameters.append(Parameter("scale_signal", low=kwargs.get("scale_signal_lo", 0.1),
                                    high=kwargs.get("scale_signal_hi", 2.0),
                                    loc=kwargs.get("scale_signal_loc", 1.0),
                                    scale=kwargs.get("scale_signal", 1.0),
                                    shape=(1,),
                                    distribution="gamma"))
        parameters.append(Parameter("offset_signal", low=kwargs.get("offset_signal_lo", 0.0),
                                    high=kwargs.get("offset_signal_hi", 1.0),
                                    loc=kwargs.get("offset_signal_loc", 0.0),
                                    scale=kwargs.get("offset_signal", 0.1),
                                    shape=(1,),
                                    distribution="gamma"))

        return StatisticalModel(statistical_model_name, "vep_autoregress", parameters, hypothesis.number_of_regions,
                                n_active_regions, n_signals, n_times, euler_method, observation_model,
                                observation_expression)




        logger.info("Reading and setting observation signals...")
        if isinstance(target_data, np.ndarray):
            self.signals = target_data
            self.data_type = kwargs.get("data_type", "empirical")
        elif isinstance(target_data, dict):
            self.data_type = "simulated"
            self.signals = target_data.get("signals", None)
            if self.signals is None:
                if self.observation_expression == 1:
                    self.signals = (target_data["x1"].T - np.expand_dims(self.model_configuration.x1EQ, 1)).T + \
                                   (target_data["z"].T - np.expand_dims(model_configuration.zEQ, 1)).T
                    # TODO: a better normalization
                    self.signals = self.signals / 2.75
                elif self.observation_expression == 2:
                    # TODO: a better normalization
                    self.signals = (target_data["x1"].T - np.expand_dims(model_configuration.x1EQ, 1)).T / 2.0
                else:
                    self.signals = target_data["x1"]
                if observation_model != 3:
                    self.signals = (np.dot(self.mixing, self.signals.T)).T
        else:
            raise_value_error("Input target data is neither a ndarray of empirical data nor a dictionary of "
                              "simulated data:\n" + str(target_data))
        (self.n_times, self.n_signals) = self.signals

        logger.info("Reading/setting mixing/gain matrix...")
        if isinstance(head, Head):
            self.region_labels = head.connectivity.region_labels
            if isinstance(getattr(head, "sensorsSEEG"), dict):
                self.sensor_labels = head.sensorsSEEG.keys()[0].labels
                self.sensor_locations = head.sensorsSEEG.keys()[0].locations
                self.mixing = head.sensorsSEEG.items()[0][channel_inds]
            elif isinstance(sensors, Sensors):
                self.sensor_labels = sensors.labels
                self.sensor_locations = sensors.locations
                self.mixing = sensors.calculate_projection(head.connectivity)
            else:
                self.sensor_labels = kwargs.get("sensor_labels")
                self.sensor_locations = kwargs.get("sensor_locations")
                self.mixing = compute_projection(self.sensor_locations, head.connectivity.centers)

        else:
            self.region_labels = kwargs.get("region_labels")
            region_centers = kwargs.get("region_centers")
            if isinstance(sensors, Sensors):
                self.sensor_labels = sensors.labels
                self.mixing = compute_projection(sensors.locations, region_centers)
            else:
                self.sensor_labels = kwargs.get("sensor_labels")
                sensor_locations = kwargs.get("sensor_locations")
                if isinstance(sensor_locations, np.ndarray):
                    self.mixing = compute_projection(sensor_locations, region_centers)
        if not(isinstance(self.mixing, np.ndarray)):
            self.mixing = kwargs.get("mixing")
        if not(isinstance(self.mixing, np.ndarray)):
            if observation_model != 3:
                raise_value_error("Mixing matrix is required by observation model " + str(self.observation_model) +
                                      "but not given or computed from input!")

        if channel_inds is not None and isinstance(self.mixing, np.ndarray):
            if len(channel_inds) < self.mixing.shape[0]:
                self.mixing = self.mixing[channel_inds]

        # Active regions are given by:
        # 1. regions with equilibria > active_regions_th
        # 2. regions of LSA propagation strength (if any) > active_regions_th
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

        if time is not None:
            self.time = time
            if self.n_times != len(time):
                warning("The length of the time vector doesn't match the one of observation signals!")
            self.dt = np.mean(np.diff(self.time))
        else:
            self.dt = kwargs.get("dt", 1000.0 / 1024)


    def prepare_data_for_stan(self, model_configuration, statistical_model, logger=LOG, **kwargs):
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
        self.epileptor_params.update({"x0cr": x0cr, "rx0": rx0})
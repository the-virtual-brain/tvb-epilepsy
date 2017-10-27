import numpy as np
from tvb.simulator.models import Epileptor

from tvb_epilepsy.base.computations.calculations_utils import calc_x0cr_r
from tvb_epilepsy.base.constants import X1_EQ_CR_DEF, X1_DEF, X0_DEF, X0_CR_DEF
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.model.model_configuration import ModelConfiguration
from tvb_epilepsy.base.model.model_vep import Head, Connectivity
from tvb_epilepsy.base.model.statistical_model import Parameter, StatisticalModel
from tvb_epilepsy.base.utils.math_utils import select_greater_values_array_inds
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning
from tvb_epilepsy.custom.simulator_custom import EpileptorModel
from tvb_epilepsy.service.epileptor_model_factory import model_noise_intensity_dict
from tvb_epilepsy.tvb_api.epileptor_models import *

AVAILABLE_DYNAMICAL_MODELS = (Epileptor, EpileptorModel, EpileptorDP2D, EpileptorDP, EpileptorDPrealistic)


LOG = initialize_logger(__name__)


class ModelInversionService(object):

    def __init__(self, hypothesis, model_configuration, head=None, dynamical_model=None, logger=LOG, **kwargs):
        self.results = {}
        self.model = None
        self.model_code = ""
        self.model_code_path = ""
        self.model_data = {}
        self.target_data_type = ""
        self.target_data = None
        self.n_times = 0
        self.n_signals = 0
        self.time = None
        self.dt = None
        if isinstance(hypothesis, DiseaseHypothesis):
            self.hypothesis = hypothesis
            logger.info("Input hypothesis set...")
        else:
            raise_value_error("Invalid input hypothesis!:\n" + str(hypothesis))
        if isinstance(model_configuration, ModelConfiguration):
            self.model_config = model_configuration
            logger.info("Input model configuration set...")
        else:
            raise_value_error("Invalid input model configuration!:\n" + str(model_configuration))
        if isinstance(head, Head):
            self.head = head
            logger.info("Input head set...")
        else:
            self.head = None
        if isinstance(dynamical_model, AVAILABLE_DYNAMICAL_MODELS):
            self.dynamical_model = dynamical_model
        else:
            self.dynamical_model = None
            warning("Invalid/No input dynamical model!:\n" + str(dynamical_model) +
                    "\nSetting default values for time scales and dynamic noise intensity parameters!")
        logger.info("Model Inversion Service instance created!")

    def get_default_tau0(self):
        if isinstance(self.dynamical_model, AVAILABLE_DYNAMICAL_MODELS):
            if isinstance(self.dynamic_model, (Epileptor, EpileptorModel)):
                return np.mean(self.dynamical_model.tt)
            elif isinstance(self.dynamic_model, (EpileptorDP, EpileptorDP2D, EpileptorDPrealistic)):
                return np.mean(self.dynamical_model.tau0)
        else:
            return 30.0

    def get_default_tau1(self):
        if isinstance(self.dynamical_model, AVAILABLE_DYNAMICAL_MODELS):
            if isinstance(self.dynamic_model, (Epileptor, EpileptorModel)):
                return np.mean(1.0 / self.dynamical_model.r)
            elif isinstance(self.dynamic_model, (EpileptorDP, EpileptorDP2D, EpileptorDPrealistic)):
                return np.mean(self.dynamical_model.tau1)
        else:
            return 0.5

    def get_default_sig(self):
        if isinstance(self.dynamical_model, AVAILABLE_DYNAMICAL_MODELS):
            if self.dynamic_model.nvars == 2:
                return model_noise_intensity_dict[self.dynamical_model._ui_name][1]
            else:
                return model_noise_intensity_dict[self.dynamical_model._ui_name][2]
        else:
            return 10 ** -4

    def get_default_sig_eq(self, x1eq_def=X1_DEF, x1eq_cr=X1_EQ_CR_DEF):
        return (x1eq_cr - x1eq_def) / 3.0

    def get_default_sig_init(self):
        return 0.1

    def get_projection(self, raise_error=False):
        projection = None
        if self.head is not None:
            if isinstance(self.head.sensorsSEEG, dict):
                projection = self.head.sensorsSEEG.items()[0]
        if projection is None and raise_error:
            raise_value_error("No projection found!")
        else:
            return projection

    def get_region_labels(self, raise_error=False):
        region_labels = None
        if self.head is not None:
            if isinstance(self.head.connectivity, Connectivity):
                region_labels = self.head.connectivity.region_labels
        if region_labels is None and raise_error:
            raise_value_error("No region labels found!")
        else:
            return region_labels

    def get_sensor_labels(self, raise_error=False):
        sensor_labels = None
        if self.head is not None:
            if isinstance(self.head.sensorsSEEG, dict):
                sensor_labels = self.head.sensorsSEEG.keys()[0].labels
        if sensor_labels is None and raise_error:
            raise_value_error("No sensor labels found!")
        else:
            return sensor_labels

    def set_empirical_target_data(self, target_data, time, **kwargs):
        self.target_data_type = kwargs.get("target_data_type", "empirical")
        self.target_data = target_data
        (self.n_times, self.n_signals) = self.target_data.shape
        time = np.array(time)
        if time.size == 1:
            self.dt = time
            self.time = np.arange(self.dt * (self.n_times - 1))
        elif time.size == self.n_times:
            self.time = time
            self.dt = np.mean(self.time)
        else:
            raise_value_error("Input time is neither a scalar nor a vector of length equal to target_data.shape[0]!" +
                              "\ntime = " + str(time))

    def set_simulated_target_data(self, statistical_model, target_data, **kwargs):
        #TODO: this function needs to be improved substantially. It lacks generality right now.
        self.target_data_type = "simulated"
        self.target_data = target_data.get("signals", None)
        if statistical_model.observation_expression == "x1z_offset":
            self.target_data = (target_data["x1"].T - np.expand_dims(self.model_config.x1EQ, 1)).T + \
                               (target_data["z"].T - np.expand_dims(self.model_config.zEQ, 1)).T
            # TODO: a better normalization
            self.target_data = target_data["x1"] / 2.75
        elif statistical_model.observation_expression == "x1_offset":
            # TODO: a better normalization
            self.target_data = (target_data["x1"].T - np.expand_dims(self.model_config.x1EQ, 1)).T / 2.0
        else: # statistical_model.observation_expression == "x1"
            self.target_data = target_data["x1"]
        if statistical_model.observation_model.find("seeg") > 0:
            self.target_data = (np.dot(kwargs.get("projection"), self.signals.T)).T
        (self.n_times, self.n_signals) = self.target_data
        self.time = target_data["time"]
        if self.time.size != self.n_times:
            raise_value_error("Input time is not a vector of length equal to target_data.shape[0]!" +
                              "\ntime = " + str(self.time))
        self.dt = np.mean(self.time)

    def get_epileptor_parameters(self, logger=LOG):
        logger.info("Unpacking epileptor parameters...")
        epileptor_params = {}
        for p in ["a", "b", "d", "yc", "Iext1", "slope"]:
            temp = getattr(self.model_config, p)
            if isinstance(temp, (np.ndarray, list)):
                if np.all(temp[0], np.array(temp)):
                    temp = temp[0]
                else:
                    raise_not_implemented_error("Statistical models where not all regions have the same value " +
                                                " for parameter " + p + " are not implemented yet!")
            self.epileptor_params.update({p: temp})
        x0cr, rx0 = calc_x0cr_r(epileptor_params["yc"], epileptor_params["Iext1"], epileptor_params["a"],
                                epileptor_params["b"], epileptor_params["d"], zmode=np.array("lin"),
                                x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF, x0cr_def=X0_CR_DEF, test=False,
                                shape=None, calc_mode="non_symbol")
        epileptor_params.update({"x0cr": x0cr, "rx0": rx0})
        return epileptor_params

    def update_active_regions_e_values(self, statistical_model, active_regions_th=0.1, reset=False, logger=LOG):
        if reset:
            statistical_model.update_active_regions([])
        return statistical_model.update_active_regions(statistical_model.active_regions +
                        select_greater_values_array_inds(self.model_configuration.e_values, active_regions_th).tolist())

    def update_active_regions_x0_values(self, statistical_model, active_regions_th=0.1, reset=False, logger=LOG):
        if reset:
            statistical_model.update_active_regions([])
        return statistical_model.update_active_regions(statistical_model.active_regions +
                       select_greater_values_array_inds(self.model_configuration.x0_values, active_regions_th).tolist())

    def update_active_regions_lsa(self, statistical_model, active_regions_th=0.1, reset=False, logger=LOG):
        if reset:
            statistical_model.update_active_regions([])
        if len(self.hypothesis.propagation_strengths) > 0:
            ps_strengths = self.hypothesis.propagation_strengths / np.max(self.hypothesis.propagation_strengths)
        return statistical_model.update_active_regions(statistical_model.active_regions +
                                             select_greater_values_array_inds(ps_strengths, active_regions_th).tolist())

    def update_active_regions_seeg(self, statistical_model, active_regions_th=0.5, projection=None, seeg_inds=None,
                                   reset=False, logger=LOG):
        if reset:
            statistical_model.update_active_regions()
        if projection is None:
            projection = self.get_projection(raise_error=True)
        if projection is not None:
            active_regions = statistical_model.active_regions
            if seeg_inds is not None:
                projection = projection[seeg_inds]
            for proj in projection:
                active_regions += select_greater_values_array_inds(proj, active_regions_th).tolist()
            return statistical_model.update_active_regions(active_regions.tolist())

    def update_active_regions(self, statistical_model, methods=["e_values", "LSA"], logger=LOG, **kwargs):
        n_methods = len(methods)
        active_regions_th = kwargs.get("active_regions_th", [None])
        n_thresholds = len(active_regions_th)
        if n_thresholds != n_methods:
            if n_thresholds ==1 and n_methods > 1:
                active_regions_th = np.repeat(active_regions_th, n_methods).tolist()
            else:
                raise_value_error("Number of input methods:\n" + str(methods) +
                                  "and active region thresholds:\n" + str(active_regions_th) +
                                  "does not match!")
        for m, th in methods, active_regions_th:
            if isequal_string(m, "e_values"):
                self.update_active_regions_e_values(statistical_model, th, logger=logger)
            elif isequal_string(m, "x0_values"):
                self.update_active_regions_x0_values(statistical_model, th, logger=logger)
            elif isequal_string(m, "lsa"):
                self.update_active_regions_lsa(statistical_model, th, logger=logger)
            elif isequal_string(m, "seeg"):
                self.update_active_regions_seeg(statistical_model, th, projection=kwargs.get("projection"),
                                                seeg_inds=kwargs.get("seeg_inds"), logger=LOG)

    def select_seeg_contacts(self, active_regions=None, projection=None, projection_th=0.5,
                                   seeg_power=None, seeg_power_inds=[], seeg_power_th=0.5, logger=LOG):
        seeg_inds = []
        if active_regions is not None:
            logger.info("Selecting SEEG contacts based on projections from active regions...")
            if projection is None:
                projection = self.get_projection(raise_error=True).T[active_regions]
                for proj in projection:
                    seeg_inds += select_greater_values_array_inds(proj, projection_th).tolist()
        if seeg_power is not None:
            logger.info("Selecting SEEG contacts based on their total power per time point...")
            seeg_inds += seeg_power_inds[select_greater_values_array_inds(seeg_power, seeg_power_th)]
        return np.unique(seeg_inds).tolist()

    def configure_statistical_model_for_vep_autoregress(self, statistical_model_name, euler_method="backward",
                                                        observation_model="logpower",
                                                        observation_expression="x1z_offset",
                                                        active_regions=[],  n_signals=0, n_times=0,
                                                        logger=LOG, **kwargs):
        parameters = []
        # Generative model:
        # Epileptor:
        parameters.append(Parameter("x1eq",
                                    low=kwargs.get("x1eq_lo", X1_DEF),
                                    high=kwargs.get("x1eq_hi", X1_EQ_CR_DEF),
                                    loc=kwargs.get("x1eq_loc", self.model_config.x1EQ),
                                    scale="parameter sig_eq",
                                    shape=(self.self.hypothesis.number_of_regions,),
                                    pdf="normal"))
        parameters.append(Parameter("K",
                                    low=kwargs.get("K_lo", self.model_config.K / 10),
                                    high=kwargs.get("K_hi", 10*self.model_config.K),
                                    loc=kwargs.get("K_loc", self.model_config.K),
                                    scale=kwargs.get("K_sc", self.model_config.K),
                                    shape=(1,),
                                    pdf="gamma"))
        parameters.append(Parameter("tau1",
                                    low=kwargs.get("tau1_lo", 0.1),
                                    high=kwargs.get("tau1_hi", 0.9),
                                    loc=kwargs.get("tau1_loc", self.get_default_tau1()),
                                    scale=None,
                                    shape=(1,),
                                    pdf="uniform"))
        tau0_def = self.get_default_tau0()
        parameters.append(Parameter("tau0",
                                    low=kwargs.get("tau0_lo", 3.0),
                                    high=kwargs.get("tau0_hi", 30000.0),
                                    loc=kwargs.get("tau0_loc", tau0_def),
                                    scale=kwargs.get("tau0_sc", tau0_def),
                                    shape=(1,),
                                    pdf="gamma"))
        # Coupling:
        parameters.append(Parameter("EC",
                                    low=kwargs.get("ec_lo", 10 ** -6),
                                    high=kwargs.get("ec_hi", 100.0),
                                    loc=kwargs.get("ec_loc", self.model_config.connectivity_matrix),
                                    scale=kwargs.get("ec_sc", 1.0),
                                    shape=self.model_config.connectivity_matrix.shape,
                                    pdf="gamma"))
        # Integration:
        sig_def = self.get_default_sig()
        parameters.append(Parameter("sig",
                                    low=kwargs.get("sig_lo", sig_def / 10.0),
                                    high=kwargs.get("sig_hi", 10*sig_def),
                                    loc=kwargs.get("sig_loc", sig_def),
                                    scale=kwargs.get("sig_sc", sig_def),
                                    shape=(1,),
                                    pdf="gamma"))
        sig_eq_def = self.get_default_sig_eq()
        parameters.append(Parameter("sig_eq",
                                    low=kwargs.get("sig_eq_lo", sig_eq_def / 10.0),
                                    high=kwargs.get("sig_eq_hi", 3 * sig_eq_def),
                                    loc=kwargs.get("sig_eq_loc", sig_eq_def),
                                    scale=kwargs.get("sig_eq_sc", sig_eq_def),
                                    shape=(1,),
                                    pdf="gamma"))
        sig_init_def = self.get_default_sig_init()
        parameters.append(Parameter("sig_init",
                                    low=kwargs.get("sig_init_lo", sig_init_def / 10.0),
                                    high=kwargs.get("sig_init_hi", 3 * sig_init_def),
                                    loc=kwargs.get("sig_init_loc", sig_init_def),
                                    scale=kwargs.get("sig_init_sc", sig_init_def),
                                    shape=(1,),
                                    pdf="gamma"))

        # Observation model
        parameters.append(Parameter("eps",
                                    low=kwargs.get("eps_lo", 0.0),
                                    high=kwargs.get("eps_hi", 1.0),
                                    loc=kwargs.get("eps_loc", 0.1),
                                    scale=kwargs.get("eps_sc", 0.1),
                                    shape=(1,),
                                    pdf="gamma"))
        parameters.append(Parameter("scale_signal",
                                    low=kwargs.get("scale_signal_lo", 0.1),
                                    high=kwargs.get("scale_signal_hi", 2.0),
                                    loc=kwargs.get("scale_signal_loc", 1.0),
                                    scale=kwargs.get("scale_signal", 1.0),
                                    shape=(1,),
                                    pdf="gamma"))
        parameters.append(Parameter("offset_signal",
                                    low=kwargs.get("offset_signal_lo", 0.0),
                                    high=kwargs.get("offset_signal_hi", 1.0),
                                    loc=kwargs.get("offset_signal_loc", 0.0),
                                    scale=kwargs.get("offset_signal", 0.1),
                                    shape=(1,),
                                    pdf="gamma"))

        return StatisticalModel(statistical_model_name, "vep_autoregress", parameters,
                                self.hypothesis.number_of_regions, active_regions, n_signals, n_times, euler_method,
                                observation_model, observation_expression)



    # def prepare_mode_data(self, statistical_model, logger=LOG, **kwargs):
    #
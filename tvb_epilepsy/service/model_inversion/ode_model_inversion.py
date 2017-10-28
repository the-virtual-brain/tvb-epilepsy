import numpy as np

from tvb_epilepsy.base.constants import X1_EQ_CR_DEF, X1_DEF, X0_DEF, X0_CR_DEF
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.utils.math_utils import select_greater_values_array_inds
from tvb_epilepsy.base.computations.calculations_utils import calc_x0cr_r
from tvb_epilepsy.base.model.statistical_models.ode_statistical_model import OdeStatisticalModel
from tvb_epilepsy.service.model_inversion.model_inversion_service import ModelInversionService
from tvb_epilepsy.tvb_api.epileptor_models import *


LOG = initialize_logger(__name__)


class OdeModelInversionService(ModelInversionService):

    def __init__(self, model_configuration, hypothesis=None, head=None, dynamical_model=None,
                 model=None, model_code=None, model_code_path="", target_data=None, target_data_type="", time=None,
                 logger=LOG, **kwargs):

        super(ModelInversionService, self).__init__(model_configuration, hypothesis, head, dynamical_model,
                                                    model, model_code, model_code_path, target_data, target_data_type,
                                                    logger)

        (self.n_times, self.n_signals) = self.observation_shape
        if time is None:
            self.time = np.linespace(self.n_times)
            self.dt = 1
        else:
            if len(time) == self.n_times and self.n_times != 0:
                raise_value_error("The length of the input time vector (" + str(len(time)) +
                                  ") does not match the one of the target_data (" + str(self.n_times) + ")!")
            else:
                self.time = time
                self.dt = np.mean(self.time)
        self.mixing = None

    def get_default_sig_init(self):
        return 0.1

    def get_projection(self, raise_error=False):
        projection = None
        if self.head is not None:
            if isinstance(self.head.sensorsSEEG, dict):
                projection = self.head.sensorsSEEG.values()[0]
        if projection is None and raise_error:
            raise_value_error("No projection found!")
        else:
            return projection

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
        self.observation_shape = self.target_data.shape
        (self.n_times, self.n_signals) = self.observation_shape
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
        else: # statistical_models.observation_expression == "x1"
            self.target_data = target_data["x1"]
        if statistical_model.observation_model.find("seeg") > 0:
            self.target_data = (np.dot(kwargs.get("projection"), self.target_data.T)).T
        self.observation_shape = self.target_data.shape
        (self.n_times, self.n_signals) = self.observation_shape
        self.time = target_data["time"]
        if self.time.size != self.n_times:
            raise_value_error("Input time is not a vector of length equal to target_data.shape[0]!" +
                              "\ntime = " + str(self.time))
        self.dt = np.mean(self.time)

    def get_epileptor_parameters(self):
        self.logger.info("Unpacking epileptor parameters...")
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

    def update_active_regions_e_values(self, statistical_model, active_regions_th=0.1, reset=False):
        if reset:
            statistical_model.update_active_regions([])
        return statistical_model.update_active_regions(statistical_model.active_regions +
                        select_greater_values_array_inds(self.model_config.e_values, active_regions_th).tolist())

    def update_active_regions_x0_values(self, statistical_model, active_regions_th=0.1, reset=False):
        if reset:
            statistical_model.update_active_regions([])
        return statistical_model.update_active_regions(statistical_model.active_regions +
                       select_greater_values_array_inds(self.model_config.x0_values, active_regions_th).tolist())

    def update_active_regions_lsa(self, statistical_model, active_regions_th=0.1, reset=False):
        if reset:
            statistical_model.update_active_regions([])
        if len(self.hypothesis.propagation_strengths) > 0:
            ps_strengths = self.hypothesis.propagation_strengths / np.max(self.hypothesis.propagation_strengths)
        return statistical_model.update_active_regions(statistical_model.active_regions +
                                             select_greater_values_array_inds(ps_strengths, active_regions_th).tolist())

    def update_active_regions_seeg(self, statistical_model, active_regions_th=0.5, projection=None, seeg_inds=None,
                                   reset=False):
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

    def update_active_regions(self, statistical_model, methods=["e_values", "LSA"], **kwargs):
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
                self.update_active_regions_e_values(statistical_model, th, logger=self.logger)
            elif isequal_string(m, "x0_values"):
                self.update_active_regions_x0_values(statistical_model, th, logger=self.logger)
            elif isequal_string(m, "lsa"):
                self.update_active_regions_lsa(statistical_model, th, logger=self.logger)
            elif isequal_string(m, "seeg"):
                self.update_active_regions_seeg(statistical_model, th, projection=kwargs.get("projection"),
                                                seeg_inds=kwargs.get("seeg_inds"), logger=self.logger)

    def select_seeg_contacts(self, active_regions=None, projection=None, projection_th=0.5,
                                   seeg_power=None, seeg_power_inds=[], seeg_power_th=0.5):
        seeg_inds = []
        if active_regions is not None:
            self.logger.info("Selecting SEEG contacts based on projections from active regions...")
            if projection is None:
                projection = self.get_projection(raise_error=True).T[active_regions]
                for proj in projection:
                    seeg_inds += select_greater_values_array_inds(proj, projection_th).tolist()
        if seeg_power is not None:
            self.logger.info("Selecting SEEG contacts based on their total power per time point...")
            seeg_inds += seeg_power_inds[select_greater_values_array_inds(seeg_power, seeg_power_th)]
        return np.unique(seeg_inds).tolist()

    def generate_statistical_model(self, statistical_model_name, **kwargs):
        return OdeStatisticalModel(statistical_model_name, kwargs, self.n_regions,
                                              kwargs.get("active_regions", []), self.n_signals, self.n_times, self.dt,
                                              kwargs.get("euler_method"), kwargs.get("observation_model"),
                                              kwargs.get("observation_expression"))


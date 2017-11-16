
import os

import numpy as np

from tvb_epilepsy.base.constants import X1_EQ_CR_DEF, X1_DEF, X0_DEF, X0_CR_DEF
from tvb_epilepsy.base.configurations import FOLDER_RES, STATS_MODELS_PATH
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.utils.math_utils import select_greater_values_array_inds
from tvb_epilepsy.base.computations.calculations_utils import calc_x0cr_r
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.stochastic_parameter import generate_stochastic_parameter
from tvb_epilepsy.base.model.statistical_models.ode_statistical_model import ODEStatisticalModel
from tvb_epilepsy.service.model_inversion.model_inversion_service import ModelInversionService
from tvb_epilepsy.tvb_api.epileptor_models import *


LOG = initialize_logger(__name__)


class ODEModelInversionService(ModelInversionService):

    def __init__(self, model_configuration, hypothesis=None, head=None, dynamical_model=None, pystan=None,
                 model_name="vep_ode", model=None, model_dir=os.path.join(FOLDER_RES, "model_inversion"),
                 model_code=None, model_code_path=os.path.join(STATS_MODELS_PATH, "vep_ode.stan"),
                 fitmode="sampling", logger=LOG, **kwargs):
        super(ODEModelInversionService, self).__init__(model_configuration, hypothesis, head, dynamical_model, pystan,
                                                       model_name, model, model_dir, model_code, model_code_path,
                                                       fitmode, logger)
        self.time = None
        self.dt = 0.0
        self.n_times = 0
        self.n_signals = 0
        self.children_dict = {"StatisticalModel": ODEStatisticalModel("StatsModel"),
                              "StochasticParameter": generate_stochastic_parameter("StochParam"),
                              "PystanService": self.pystan}

    def get_default_sig_init(self):
        return 0.1

    def set_time(self, time=None):
        if time is not None:
            time = np.array(time)
            try:
                if time.size == 1:
                    self.dt = time
                    self.time = np.arange(self.dt * (self.n_times - 1))
                elif time.size == self.n_times:
                    self.time = time
                    self.dt = np.mean(self.time)
                else:
                    raise_value_error("Input time is neither a scalar nor a vector of length equal to target_data.shape[0]!" +
                                      "\ntime = " + str(time))
            except:
                raise_value_error(
                    "Input time is neither a scalar nor a vector of length equal to target_data.shape[0]!" +
                    "\ntime = " + str(time))
        else:
            raise_value_error("Input time is neither a scalar nor a vector of length equal to target_data.shape[0]!" +
                              "\ntime = " + str(time))

    def set_target_data_and_time(self, target_data_type, target_data, time=None, **kwargs):
        if isequal_string(target_data_type, "simulated"):
            self.set_simulated_target_data(target_data, statistical_model=kwargs.get("statistical_model", None),
                                           time=kwargs.get("time", None), projection=kwargs.get("projection", None))
            self.target_data_type = "simulated"
        else:  # isequal_string(target_data_type, "empirical"):
            self.set_empirical_target_data(target_data, time)
            self.target_data_type = "empirical"
        if time is None and isinstance(target_data, dict):
            self.set_time(target_data.get("time", None))

    def set_empirical_target_data(self, target_data, time=None):
        if isinstance(target_data, dict):
            self.target_data = target_data.get("signals", target_data.get("target_data", None))
        self.observation_shape = self.target_data.shape
        (self.n_times, self.n_signals) = self.observation_shape

    def set_simulated_target_data(self, target_data,  statistical_model=None, time=None, projection=None):
        #TODO: this function needs to be improved substantially. It lacks generality right now.
        if statistical_model is None or not(isequal_string(target_data, dict)):
            self.set_empirical_target_data(target_data, time)
        else:
            if statistical_model.observation_expression == "x1z_offset":
                self.target_data = ((target_data["x1"].T - np.expand_dims(self.model_config.x1EQ, 1)).T + \
                                   (target_data["z"].T - np.expand_dims(self.model_config.zEQ, 1)).T) / 2.75
                # TODO: a better normalization
            elif statistical_model.observation_expression == "x1_offset":
                # TODO: a better normalization
                self.target_data = (target_data["x1"].T - np.expand_dims(self.model_config.x1EQ, 1)).T / 2.0
            else: # statistical_models.observation_expression == "x1"
                self.target_data = target_data["x1"]
            self.observation_shape = self.target_data.shape
            (self.n_times, self.n_signals) = self.observation_shape
            if self.time.size != self.n_times:
                raise_value_error("Input time is not a vector of length equal to target_data.shape[0]!" +
                                  "\ntime = " + str(self.time))
            if statistical_model.observation_model.find("seeg") > 0:
                if projection is not None:
                    self.target_data = (np.dot(projection[statistical_model.active_regions], self.target_data.T)).T

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

    def generate_model_parameters(self, **kwargs):
        parameters = super(ODEModelInversionService, self).generate_model_parameters(**kwargs)
        # Integration
        parameter = kwargs.get("sig_init", None)
        if not(isinstance(parameter, Parameter)):
            sig_init_def = kwargs.get("sig_init_def", 0.1)
            parameter = generate_stochastic_parameter("sig_init",
                                                      low=kwargs.get("sig_init_lo", 0.0),
                                                      high=kwargs.get("sig_init_hi", 2 * sig_init_def),
                                                      p_shape=(),
                                                      probability_distribution=kwargs.get("sig_init_pdf", "lognormal"),
                                                      optimize=False,
                                                      mean=sig_init_def,
                                                      sigma=kwargs.get("tau1_sig", sig_init_def))
        parameters.append(parameter)

        # Observation model
        parameter = kwargs.get("scale_signal")
        if not(isinstance(parameter, Parameter)):
            scale_signal_def = kwargs.get("scale_signal_def", 1.0)
            parameter = generate_stochastic_parameter("scale_signal",
                                                      low=kwargs.get("scale_signal_lo", 0.1),
                                                      high=kwargs.get("scale_signal_hi", 2.0),
                                                      p_shape=(),
                                                      probability_distribution=
                                                      kwargs.get("scale_signal_pdf", "lognormal"),
                                                      optimize=True,
                                                      mode=scale_signal_def,
                                                      std=kwargs.get("scale_signal_sig", scale_signal_def))
        parameters.append(parameter)

        parameter = kwargs.get("offset_signal")
        if not(isinstance(parameter, Parameter)):
            offset_signal_def = kwargs.get("offset_signal_def", 0.0)
            parameter = generate_stochastic_parameter("offset_signal",
                                                      low=kwargs.get("offset_signal_lo", -1.0),
                                                      high=kwargs.get("offset_signal_hi", 1.0),
                                                      p_shape=(),
                                                      probability_distribution=
                                                                            kwargs.get("offset_signal_pdf", "normal"),
                                                      optimize=False,
                                                      mean=offset_signal_def,
                                                      sigma=kwargs.get("scale_signal_sig", offset_signal_def))
        parameters.append(parameter)
        return parameters

    def generate_statistical_model(self, statistical_model_name="vep_ode", **kwargs):
        return ODEStatisticalModel(statistical_model_name, self.generate_model_parameters(**kwargs), self.n_regions,
                                   kwargs.get("active_regions", []), self.n_signals, self.n_times, self.dt,
                                   kwargs.get("euler_method"), kwargs.get("observation_model"),
                                   kwargs.get("observation_expression"))


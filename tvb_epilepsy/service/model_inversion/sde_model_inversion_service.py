
import os

import numpy as np

from tvb_epilepsy.base.constants import EPILEPTOR_MODEL_NVARS
from tvb_epilepsy.base.configurations import STATS_MODELS_PATH, FOLDER_RES
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.stochastic_parameter import generate_stochastic_parameter
from tvb_epilepsy.service.probability_distribution_factory import AVAILABLE_DISTRIBUTIONS
from tvb_epilepsy.base.model.statistical_models.ode_statistical_model import \
                                                        EULER_METHODS, OBSERVATION_MODEL_EXPRESSIONS, OBSERVATION_MODELS
from tvb_epilepsy.base.model.statistical_models.sde_statistical_model import SDEStatisticalModel
from tvb_epilepsy.service.model_inversion.ode_model_inversion_service import ODEModelInversionService
from tvb_epilepsy.service.epileptor_model_factory import model_noise_intensity_dict


LOG = initialize_logger(__name__)


class SDEModelInversionService(ODEModelInversionService):

    def __init__(self, model_configuration, hypothesis=None, head=None, dynamical_model=None, pystan=None,
                 model_name=None, model=None, model_dir=os.path.join(FOLDER_RES, "model_inversion"),
                 model_code=None, model_code_path=None, fitmode="sampling", sde_mode="dWt", logger=LOG):
        self.sde_mode = sde_mode
        if isequal_string(sde_mode, "dWt"):
            self.x1var = "x1_dWt"
            self.zvar = "z_dWt"
            default_model = "vep_sde_dWt"
        else:
            self.x1var = "x1"
            self.zvar = "z"
            default_model = "vep_sde"
        if model_name is None:
            model_name = default_model
        if model_code_path is None:
            model_code_path = os.path.join(STATS_MODELS_PATH, model_name + ".stan")

        super(SDEModelInversionService, self).__init__(model_configuration, hypothesis, head, dynamical_model, pystan,
                                                       model_name, model, model_dir, model_code, model_code_path,
                                                       fitmode, logger)

        self.children_dict = {"SDEStatisticalModel": SDEStatisticalModel("SDEStatsModel"),
                              "StochasticParameter": generate_stochastic_parameter("StochParam"),
                              "PystanService": self.pystan}

    def get_default_sig(self):
            if EPILEPTOR_MODEL_NVARS.get([self.dynamical_model]) == 2:
                return model_noise_intensity_dict[self.dynamical_model][1]
            elif EPILEPTOR_MODEL_NVARS.get([self.dynamical_model]) > 2:
                return model_noise_intensity_dict[self.dynamical_model][2]
            else:
                return

    def generate_state_variables_parameters(self, parameters, **kwargs):
        if isequal_string(self.sde_mode, "dWt"):
            parameters.append(kwargs.get("x1_dWt", generate_stochastic_parameter("x1_dWt",
                                                                             low=kwargs.get("x1_dWt_lo", -1.0),
                                                                             high=kwargs.get("x1_dWt_hi", 1.0),
                                                                             p_shape=(
                                                                             self.n_times, self.n_active_regions),
                                                                             probability_distribution="normal"),
                                                                             mean=0.0, sigma=1.0))
            parameters.append(kwargs.get("z_dWt", generate_stochastic_parameter("z_dWt",
                                                                            low=kwargs.get("z_dWt_lo", -1.0),
                                                                            high=kwargs.get("z_dWt_hi", 1.0),
                                                                            p_shape=(
                                                                            self.n_times, self.n_active_regions),
                                                                            probability_distribution="normal"),
                                                                            mean=0.0, sigma=1.0))
        else:
            parameters.append(kwargs.get("x1", generate_stochastic_parameter("x1",
                                                                             low=kwargs.get("x1_lo", -2.0),
                                                                             high=kwargs.get("x1_hi", 2.0),
                                                                             p_shape=(self.n_times, self.n_active_regions),
                                                                             probability_distribution="normal"),
                                                                             mean=0.0, sigma=1.0))
            parameters.append(kwargs.get("z", generate_stochastic_parameter("z",
                                                                            low=kwargs.get("z_lo", 2.0),
                                                                            high=kwargs.get("z_hi", 5.0),
                                                                            p_shape=(self.n_times, self.n_active_regions),
                                                                            probability_distribution="normal"),
                                                                            mean=3.5, sigma=1.0))
        return parameters

    def generate_model_parameters(self, **kwargs):
        parameters = super(SDEModelInversionService, self).generate_model_parameters(**kwargs)
        # State variables:
        parameters = self.generate_state_variables_parameters(parameters, **kwargs)
        # Integration
        parameter = kwargs.get("sig", None)
        if not(isinstance(parameter, Parameter)):
            sig_def = kwargs.get("sig_def", 10 ** -4)
            parameter = generate_stochastic_parameter("sig",
                                                      low=kwargs.get("sig_lo", 0.0),
                                                      high=kwargs.get("sig_hi", 10 * sig_def),
                                                      p_shape=(),
                                                      probability_distribution=kwargs.get("sig_pdf", "gamma"),
                                                      optimize=True,
                                                      mode=sig_def, std=kwargs.get("sig_sig", sig_def))
        parameters.append(parameter)
        return parameters
                
    def generate_statistical_model(self, statistical_model_name=None, **kwargs):
        if statistical_model_name is None:
            statistical_model_name = self.pystan.model_name
        return SDEStatisticalModel(statistical_model_name, self.generate_model_parameters(**kwargs),
                                   self.n_regions, kwargs.get("active_regions", []), self.n_signals,
                                   self.n_times, self.dt,
                                   kwargs.get("euler_method"), kwargs.get("observation_model"),
                                   kwargs.get("observation_expression"))

    def generate_model_data(self, statistical_model, projection):
        super(SDEModelInversionService, self).generate_model_data(statistical_model, projection, x1var=self.x1var,
                                                                                                 zvar=self.zvar)

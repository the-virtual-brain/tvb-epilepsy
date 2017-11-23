
import time

import numpy as np

from tvb_epilepsy.base.constants.model_constants import model_noise_intensity_dict
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, construct_import_path
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.sde_statistical_model import SDEStatisticalModel
from tvb_epilepsy.base.model.statistical_models.stochastic_parameter import generate_stochastic_parameter
from tvb_epilepsy.service.epileptor_model_factory import AVAILABLE_DYNAMICAL_MODELS_NAMES, EPILEPTOR_MODEL_NVARS
from tvb_epilepsy.service.model_inversion.ode_model_inversion_service import ODEModelInversionService


LOG = initialize_logger(__name__)


class SDEModelInversionService(ODEModelInversionService):

    SIG_DEF = 10 ** -4
    X1_MIN = -3.0
    X1_MAX = 3.0
    Z_MIN = 0.0
    Z_MAX = 7.0

    def __init__(self, model_configuration, hypothesis=None, head=None, dynamical_model=None, model_name=None, 
                 sde_mode="dWt", logger=LOG, **kwargs):
        self.sde_mode = sde_mode
        if isequal_string(sde_mode, "dWt"):
            self.x1var = "x1_dWt"
            self.zvar = "z_dWt"
            default_model = "vep_dWt"
        else:
            self.x1var = "x1"
            self.zvar = "z"
            default_model = "vep_sde"
        if not(isinstance(model_name, basestring)):
            model_name = default_model
        self.sig = kwargs.get("sig", self.SIG_DEF)
        if np.in1d(dynamical_model, AVAILABLE_DYNAMICAL_MODELS_NAMES):
           self.get_default_sig(dynamical_model)
        super(SDEModelInversionService, self).__init__(model_configuration, hypothesis, head, dynamical_model, 
                                                       model_name, logger, **kwargs)
        self.context_str = "from " + construct_import_path(__file__) + " import " + self.__class__.__name__
        self.context_str += "; from tvb_epilepsy.base.model.model_configuration import ModelConfiguration"
        self.create_str = "ODEModelInversionService(ModelConfiguration())"

    def get_default_sig(self, dynamical_model):
            if EPILEPTOR_MODEL_NVARS.get([self.dynamical_model]) == 2:
                return model_noise_intensity_dict[dynamical_model][1]
            elif EPILEPTOR_MODEL_NVARS.get([dynamical_model]) > 2:
                return model_noise_intensity_dict[dynamical_model][2]

    def generate_state_variables_parameters(self, parameters, **kwargs):
        if isequal_string(self.sde_mode, "dWt"):
            parameters.update({"x1_dWt": kwargs.get("x1_dWt", generate_stochastic_parameter("x1_dWt",
                                                                             low=kwargs.get("x1_dWt_lo", -1.0),
                                                                             high=kwargs.get("x1_dWt_hi", 1.0),
                                                                             p_shape=(),
                                                                             probability_distribution="normal",
                                                                             mean=0.0, sigma=1.0))})
            parameters.update({"z_dWt": kwargs.get("z_dWt", generate_stochastic_parameter("z_dWt",
                                                                            low=kwargs.get("z_dWt_lo", -1.0),
                                                                            high=kwargs.get("z_dWt_hi", 1.0),
                                                                            p_shape=(),
                                                                            probability_distribution="normal",
                                                                            mean=0.0, sigma=1.0))})
        else:
            parameters.update({"x1": kwargs.get("x1", generate_stochastic_parameter("x1",
                                                                             low=kwargs.get("x1_lo", self.X1_MIN),
                                                                             high=kwargs.get("x1_hi", self.X1_MAX),
                                                                             p_shape=(),
                                                                             probability_distribution="normal",
                                                                             mean=self.x1EQ, sigma=1.0))})
            parameters.update({"z": kwargs.get("z", generate_stochastic_parameter("z",
                                                                            low=kwargs.get("z_lo", self.Z_MIN),
                                                                            high=kwargs.get("z_hi", self.Z_MAX),
                                                                            p_shape=(),
                                                                            probability_distribution="normal",
                                                                            mean=self.zEQ, sigma=1.0))})
        return parameters

    def generate_model_parameters(self, **kwargs):
        parameters = super(SDEModelInversionService, self).generate_model_parameters(**kwargs)
        # State variables:
        parameters = self.generate_state_variables_parameters(parameters, **kwargs)
        # Integration
        parameter = kwargs.get("sig", None)
        if not(isinstance(parameter, Parameter)):
            sig_def = kwargs.get("sig_def", self.sig)
            parameter = generate_stochastic_parameter("sig",
                                                      low=kwargs.get("sig_lo", 0.0),
                                                      high=kwargs.get("sig_hi", 10 * sig_def),
                                                      p_shape=(),
                                                      probability_distribution=kwargs.get("sig_pdf", "gamma"),
                                                      optimize=True,
                                                      mode=sig_def, std=kwargs.get("sig_sig", sig_def))
        mm = np.max([parameter.mean, parameter.mode])
        if parameter.high < 10*mm:
            parameter.high = 10*mm
        parameters.update({parameter.name: parameter})
        return parameters
                
    def generate_statistical_model(self, model_name=None, **kwargs):
        if model_name is None:
            model_name = self.model_name
        tic = time.time()
        self.logger.info("Generating model...")
        model = SDEStatisticalModel(model_name, self.generate_model_parameters(**kwargs), self.n_regions,
                                    kwargs.get("active_regions", []), self.n_signals, self.n_times, self.dt, **kwargs)
        self.model_generation_time = time.time() - tic
        self.logger.info(str(self.model_generation_time) + ' sec required for model generation')
        return model

    def generate_model_data(self, statistical_model, signals, projection=None):
        return super(SDEModelInversionService, self).generate_model_data(statistical_model, signals, projection,
                                                                         x1var=self.x1var, zvar=self.zvar)

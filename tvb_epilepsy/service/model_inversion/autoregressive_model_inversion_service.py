import numpy as np

from tvb_epilepsy.base.constants import EPILEPTOR_MODEL_NVARS

from tvb_epilepsy.base.model.statistical_models.autoregressive_statistical_model import AutoregressiveStatisticalModel
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning
from tvb_epilepsy.base.model.statistical_models.probability_distributions.probability_distribution import \
                                                                                                AVAILABLE_DISTRIBUTIONS
from tvb_epilepsy.base.model.statistical_models.ode_statistical_model import \
                                                        EULER_METHODS, OBSERVATION_MODEL_EXPRESSIONS, OBSERVATION_MODELS
from tvb_epilepsy.service.epileptor_model_factory import model_noise_intensity_dict
from tvb_epilepsy.base.model.parameter import Parameter
from tvb_epilepsy.base.model.statistical_models.probability_distributions.probability_distribution import \
                                                                                                   generate_distribution
from tvb_epilepsy.service.model_inversion.ode_model_inversion_service import OdeModelInversionService


LOG = initialize_logger(__name__)


class AutoregressiveModelInversionService(OdeModelInversionService):

    def __init__(self, model_configuration, hypothesis=None, head=None, dynamical_model=None,
                 model=None, model_code=None, model_code_path="", target_data=None, target_data_type="", time=None,
                 logger=LOG):

        super(AutoregressiveModelInversionService, self).__init__(model_configuration, hypothesis, head,
                                                                  dynamical_model, model, model_code, model_code_path,
                                                                  target_data, target_data_type, time, logger)

    def get_default_sig(self):
            if EPILEPTOR_MODEL_NVARS.get([self.dynamic_model]) == 2:
                return model_noise_intensity_dict[self.dynamic_model][1]
            elif EPILEPTOR_MODEL_NVARS.get([self.dynamic_model]) > 2:
                return model_noise_intensity_dict[self.dynamic_model][2]
            else:
                return

    def generate_model_parameters(self, **kwargs):
        parameters = OdeModelInversionService.generate_model_parameters(**kwargs)
        # State variables:
        parameters.append(kwargs.get("x1", Parameter("x1",
                                                          low=kwargs.get("x1_lo", -2.0),
                                                          high=kwargs.get("x1_hi", 2.0),
                                                          probability_distribution="normal",
                                                          shape=(self.n_times, self.n_active_regions))))
        parameters.append(kwargs.get("z", Parameter("z",
                                                         low=kwargs.get("z_lo", 2.0),
                                                         high=kwargs.get("z_hi", 5.0),
                                                         probability_distribution="normal",
                                                         shape=(self.n_times, self.n_active_regions))))

        # Integration
        parameter = kwargs.get("sig", None)
        if parameter is None:
            probability_distribution = kwargs.get("sig_pdf", "gamma")
            if isinstance(probability_distribution, basestring):
                sig_def = kwargs.get("sig_def", 10 ** -4)
                probability_distribution = generate_distribution(probability_distribution,
                                                                 target_shape=(),
                                                                 mode=sig_def,
                                                                 std=kwargs.get("sig_sig", sig_def))
                parameter = Parameter("sig",
                                      low=kwargs.get("sig_lo", sig_def / 10.0),
                                      high=kwargs.get("sig_hi", 10 * sig_def),
                                      probability_distribution=probability_distribution,
                                      shape=())
        parameters.append(parameter)
        return parameters
                
    def generate_statistical_model(self, statistical_model_name, **kwargs):
        return AutoregressiveStatisticalModel(statistical_model_name, self.generate_model_parameters(**kwargs),
                                              self.n_regions, kwargs.get("active_regions", []), self.n_signals,
                                              self.n_times, self.dt,
                                              kwargs.get("euler_method"), kwargs.get("observation_model"),
                                              kwargs.get("observation_expression"))

    def generate_model_data(self, statistical_model, projection, logger=LOG, **kwargs):
        active_regions_flag = np.zeros((statistical_model.n_regions,), dtype="i")
        active_regions_flag[statistical_model.active_regions] = 1
        self.model_data = {"n_regions": statistical_model.n_regions,
                           "n_times": statistical_model.n_times,
                           "n_signals": statistical_model.n_signals,
                           "n_active_regions": statistical_model.n_active_regions,
                           "n_nonactive_regions": statistical_model.n_nonactive_regions,
                           "active_regions_flag": active_regions_flag,
                           "active_regions": statistical_model.active_regions,
                           "nonactive_regions": np.where(1 - active_regions_flag)[0],
                           "x0_nonactive": self.model_configuration.x0[~active_regions_flag.astype("bool")],
                           "dt": self.dt,
                           "euler_method": np.where(np.in1d(EULER_METHODS, statistical_model.euler_method))[0] - 1,
                           "observation_model": np.where(np.in1d(OBSERVATION_MODELS,
                                                                 statistical_model.observation_model))[0],
                           "observation_expression": np.where(np.in1d(OBSERVATION_MODEL_EXPRESSIONS,
                                                                      statistical_model.observation_expression))[0],
                           "signals": self.target_data,
                           "mixing": projection,
                           "x1eq0": statistical_model.paramereters["x1eq"].mean}
        for key, val in self.get_epileptor_parameters(logger=logger).iteritems():
            self.model_data.update({key: val})
        for p in statistical_model.paramereters.values():
            self.model_data.update({p.name + "_lo": p.low, p.name + "_hi": p.high,
                                    p.name + "_pdf": np.where(np.in1d(AVAILABLE_DISTRIBUTIONS,
                                                                      p.probability_distribution.name))[0]})
            if isequal_string(p.name, "x1") or isequal_string(p.name, "z"):
                pass
            elif isequal_string(p.name, "x1eq") or isequal_string(p.name, "x1init") or isequal_string(p.name, "zinit"):
                    warning("For the moment only normal distribution is allowed for parameters " + p.name +
                            "!\nIgnoring the selected probability distribution!")
            else:
                self.model_data.update({p.name + "_pdf":
                                        np.where(np.in1d(AVAILABLE_DISTRIBUTIONS, p.probability_distribution.name))[0]})
                pdf_params = p.probability_distribution.params().values()
                self.model_data.update({p.name + "_p1": pdf_params[0]})
                if len(pdf_params) == 1:
                    self.model_data.update({p.name + "_p2": pdf_params[0]})
                else:
                    self.model_data.update({p.name + "_p2": pdf_params[1]})


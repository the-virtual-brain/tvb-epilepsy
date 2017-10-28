import numpy as np

from tvb_epilepsy.base.constants import EPILEPTOR_MODEL_NVARS
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list
from tvb_epilepsy.base.model.statistical_models.autoregressive_statistical_model import AutoregressiveStatisticalModel
from tvb_epilepsy.service.model_inversion.ode_model_inversion import OdeModelInversionService
from tvb_epilepsy.service.epileptor_model_factory import model_noise_intensity_dict


LOG = initialize_logger(__name__)


class AutoregressiveModelInversionService(OdeModelInversionService):

    def __init__(self, model_configuration, hypothesis=None, head=None, dynamical_model=None,
                 model=None, model_code=None, model_code_path="", target_data=None, target_data_type="", time=None,
                 logger=LOG):

        super(OdeModelInversionService, self).__init__(model_configuration, hypothesis, head, dynamical_model, model,
                                                       model_code, model_code_path, target_data, target_data_type, time,
                                                       logger)

    def get_default_sig(self):
            if EPILEPTOR_MODEL_NVARS.get([self.dynamic_model]) == 2:
                return model_noise_intensity_dict[self.dynamic_model][1]
            elif EPILEPTOR_MODEL_NVARS.get([self.dynamic_model]) > 2:
                return model_noise_intensity_dict[self.dynamic_model][2]
            else:
                return

    def generate_statistical_model(self, statistical_model_name, **kwargs):
        return AutoregressiveStatisticalModel(statistical_model_name, kwargs, self.n_regions,
                                              kwargs.get("active_regions", []), self.n_signals,
                                              self.n_times, self.dt,
                                              kwargs.get("euler_method"), kwargs.get("observation_model"),
                                              kwargs.get("observation_expression"))

# def generate_model_data(self, statistical_model, logger=LOG, **kwargs):
    #     active_regions_flag = np.zeros((statistical_model.n_regions,), dtype="i")
    #     active_regions_flag[statistical_model.active_regions] = 1
    #     self.model_data = {"n_regions": statistical_model.n_regions,
    #                        "n_times": statistical_model.n_times,
    #                        "n_signals": statistical_model.n_signals,
    #                        "n_active_regions": statistical_model.n_active_regions,
    #                        "n_nonactive_regions": statistical_model.n_nonactive_regions,
    #                        "active_regions_flag": active_regions_flag,
    #                        "active_regions": statistical_model.active_regions,
    #                        "nonactive_regions": np.where(1 - active_regions_flag)[0],
    #                        "x0_nonactive": self.model_configuration.x0[~active_regions_flag.astype("bool")],
    #                        "x1eq0": self.model_configuration.x1EQ,
    #                        "dt": self.dt,
    #                        "euler_method": np.where(
    #                            np.in1d(statistical_model.euler_method, ["backward", "midpoint", "forward"])) - 1,
    #                        "observation_model": np.where(
    #                            np.in1d(statistical_model.observation_model, ["seeg_power", "seeg_logpower",
    #                                                                          "lfp_power", "lfp_logpower"])),
    #                        "observation_expression": np.where(
    #                            np.in1d(statistical_model.observation_expression, ["x1z_offset", "x1_offset", "x1"])),
    #                        "signals": self.target_data}
    #     for key, val in self.get_epileptor_parameters(logger=logger).iteritems():
    #         self.model_data.update({key: val})
    #     for p in statistical_model.paramereters.values():
    #         self.model_data.update({p.name + "_lo": p.low,
    #                                 p.name + "_hi": p.high})
    #         pdf_params = ensure_list(p.get_pdf_params())
    #         self.model_data.update({p.name + "_p1": pdf_params[0]})
    #         if len(ensure_list(pdf_params)) == 1:
    #             self.model_data.update({p.name + "_p2": 0.0})
    #         else:
    #             self.model_data.update({p.name + "_p2": pdf_params[1]})
    #     channel_inds = self.select_seeg_contacts(statistical_model.active_regions,
    #                                              projection=self.head.sensorsSEEG.values()[0],
    #                                              projection_th=kwargs.get("projection_th", 0.5),
    #                                              seeg_power=kwargs.get("seeg_power"),
    #                                              seeg_power_inds=kwargs.get("seeg_power_inds"),
    #                                              seeg_power_th=kwargs.get("seeg_power_th", 0.5), logger=logger)
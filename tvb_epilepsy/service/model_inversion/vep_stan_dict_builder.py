import numpy as np
from copy import deepcopy
from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, sort_dict
from tvb_epilepsy.base.constants.model_inversion_constants import *
from tvb_epilepsy.base.computations.probability_distributions import ProbabilityDistributionTypes


def set_mixing(statistical_model, model_inversion, sensors=None, gain_matrix=None):
    if gain_matrix is None:
        if statistical_model.observation_model.value in OBSERVATION_MODELS.SEEG.value:
            mixing = deepcopy(sensors.gain_matrix)
        else:
            mixing = np.eye(statistical_model.number_of_regions)
    else:
        mixing = deepcopy(gain_matrix)
    if mixing.shape[0] > len(model_inversion.signals_inds):
        mixing = mixing[model_inversion.signals_inds]
    return mixing


def set_time(statistical_model, time=None):
    if time is None:
        time = np.arange(0, statistical_model.dt, statistical_model.time_length)
    return time


# def build_stan_model_dict(statistical_model, signals, model_inversion,
#                           time=None, sensors=None, gain_matrix=None):
# TODO: needs updating
#     """
#     Builds a dictionary with data needed for stan models.
#     :param statistical_model: StatisticalModel object
#     :param signals:
#     :param model_inversion: ModelInversionService object
#     :param gain_matrix: array
#     :return: dictionary with stan data
#     """
#     active_regions_flag = np.zeros((statistical_model.number_of_regions,), dtype="i")
#     active_regions_flag[statistical_model.active_regions] = 1
#     SC = statistical_model.model_config.model_connectivity
#     model_data = {"number_of_regions": statistical_model.number_of_regions,
#                   "n_times": statistical_model.time_length,
#                   "n_signals": statistical_model.number_of_signals,
#                   "n_active_regions": statistical_model.number_of_active_regions,
#                   "n_nonactive_regions": statistical_model.number_of_nonactive_regions,
#                   "n_connections": statistical_model.number_of_regions * (statistical_model.number_of_regions - 1) / 2,
#                   "active_regions_flag": np.array(active_regions_flag),
#                   "active_regions": np.array(statistical_model.active_regions) + 1,  # cmdstan cannot take lists!
#                   "nonactive_regions": np.array(statistical_model.nonactive_regions) + 1,  # indexing starts from 1!
#                   "x1eq_min": statistical_model.parameters["x1eq"].low,
#                   "x1eq_max": statistical_model.parameters["x1eq"].high,
#                   "SC": SC[np.triu_indices(SC.shape[0], 1)],
#                   "dt": statistical_model.dt,
#                   "signals": signals,
#                   "time": set_time(statistical_model, time),
#                   "mixing": set_mixing(statistical_model, model_inversion, sensors, gain_matrix),
#                   "observation_model": statistical_model.observation_model.value,
#                   # "observation_expression": np.where(np.in1d(OBSERVATION_MODEL_EXPRESSIONS,
#                   #                                            statistical_model.observation_expression))[0][0],
#                   # "euler_method": np.where(np.in1d(EULER_METHODS, statistical_model.euler_method))[0][0] - 1,
#                   }
#     for p in ["a", "b", "d", "yc", "Iext1", "slope"]:
#         model_data.update({p: getattr(statistical_model.model_config, p)})
#     for p in statistical_model.parameters.values():
#         model_data.update({p.name + "_lo": p.low, p.name + "_hi": p.high})
#         if not (isequal_string(p.type, "normal")):
#             model_data.update({p.name + "_loc": p.loc, p.name + "_scale": p.scale,
#                                p.name + "_pdf":
#                                    np.where(np.in1d(ProbabilityDistributionTypes.available_distributions, p.type))[0][
#                                        0],
#                                p.name + "_p": (np.array(p.pdf_params().values()).T * np.ones((2,))).squeeze()})
#     model_data["x1eq_star_loc"] = statistical_model.parameters["x1eq_star"].mean
#     model_data["x1eq_star_scale"] = statistical_model.parameters["x1eq_star"].std
#     model_data["MC_scale"] = statistical_model.MC_scale
#     MCsplit_shape = np.ones(statistical_model.parameters["MCsplit"].p_shape)
#     model_data["MCsplit_loc"] = statistical_model.parameters["MCsplit"].mean * MCsplit_shape
#     model_data["MCsplit_scale"] = statistical_model.parameters["MCsplit"].std * MCsplit_shape
#     model_data["offset_signal_p"] = np.array(statistical_model.parameters["offset_signal"].pdf_params().values())
#     return sort_dict(model_data)


def build_stan_model_dict_to_interface_ins(statistical_model, signals, model_inversion,
                                           time=None, sensors=None, gain_matrix=None,
                                           parameter_names={"K": "k", "MC": "FC", "tau1": "time_scale",
                                                            "x1init": "x_init", "zinit": "z_init",
                                                            "x1": "x", "dX1t": "x_eta", "dZt": "z_eta",
                                                            "scale": "amplitude",
                                                            }):
    """
    Usually takes as input the model_data created with <build_stan_model_dict> and adds fields that are needed to
    interface the ins stan model.
    :param statistical_model: StatisticalModel object
    :param model_inversion:  ModelInversionService object
    :param gain_matrix: ndarray
    :return: dict with data needed for stan models
    """
    active_regions = statistical_model.active_regions
    nonactive_regions = statistical_model.nonactive_regions
    SC = statistical_model.model_config.model_connectivity
    if time is None:
        time = np.arange(0, statistical_model.dt, statistical_model.time_length)
    for p_old_name, p_new_name in parameter_names.iteritems():
        try:
            statistical_model.parameters[p_new_name] = statistical_model.parameters[p_old_name]
        except:
            warning("Parameter " + p_old_name + " not found in statistical model\n" + str(statistical_model))
    vep_data = {"nn": statistical_model.number_of_regions,
                "nt": statistical_model.time_length,
                "ns": statistical_model.number_of_signals,
                "dt": statistical_model.dt,
                "I1": statistical_model.model_config.Iext1,
                "x0_star_mu": statistical_model.parameters["x0"].star_mean,
                "x0_star_std": statistical_model.parameters["x0"].star_std,
                "x0_lo": statistical_model.parameters["x0"].low,
                "x0_hi": statistical_model.parameters["x0"].high,
                "x_init_mu": statistical_model.parameters["x1init"].mean,
                "z_init_mu": statistical_model.parameters["zinit"].mean,
                "x_eq_def": statistical_model.model_config.x1eq[nonactive_regions].mean(),
                "init_std": np.mean(statistical_model.parameters["x1init"].std),
                "tau0": 10.0,  # statistical_model.parameters["tau0"].mean,
                # "K_lo": statistical_model.parameters["k"].low,
                # "K_u": statistical_model.parameters["k"].mode,
                # "K_v": statistical_model.parameters["k"].var,
                "time_scale_mu": statistical_model.parameters["time_scale"].mean,
                "time_scale_std": statistical_model.parameters["time_scale"].std,
                "k_mu": statistical_model.parameters["k"].mean,
                "k_std": statistical_model.parameters["k"].std,
                "SC": SC[active_regions][:, active_regions],
                "SC_var": 5.0,  # 1/36 = 0.02777777,
                "Ic": np.sum(SC[active_regions][:, nonactive_regions], axis=1),
                "sigma_mu": statistical_model.parameters["sigma"].mean,
                "sigma_std": statistical_model.parameters["sigma"].std,
                "epsilon_mu": statistical_model.parameters["epsilon"].mean,
                "epsilon_std": statistical_model.parameters["epsilon"].std,
                "sig_hi": 0.025,  # model_data["sig_hi"],
                "amplitude_mu": statistical_model.parameters["amplitude"].mean,
                "amplitude_std": statistical_model.parameters["amplitude"].std,
                "amplitude_lo": 0.3,
                "offset_mu": statistical_model.parameters["offset"].mean,
                "offset_std": statistical_model.parameters["offset"].std,
                "seeg_log_power": signals,
                # 9.0 * model_data["signals"] - 4.0,  # scale from (0, 1) to (-4, 5)
                "gain": set_mixing(statistical_model, model_inversion, sensors, gain_matrix),
                "time": set_time(statistical_model, time)
                }
    return vep_data

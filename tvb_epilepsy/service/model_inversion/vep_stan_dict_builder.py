import numpy as np
# from copy import deepcopy
from tvb_epilepsy.base.utils.log_error_utils import warning
from tvb_epilepsy.base.utils.data_structures_utils import ensure_list  # isequal_string, sort_dict
# from tvb_epilepsy.base.constants.model_inversion_constants import *
# from tvb_epilepsy.base.computations.probability_distributions import ProbabilityDistributionTypes


INS_PARAMS_NAMES_DICT={"K": "k", "MC": "FC", "tau1": "time_scale", "tau0": "tau",
                       "scale": "amplitude", "target_data": "seeg_log_power", "fit_target_data": "mu_seeg_log_power",
                       "x1init": "x_init", "zinit": "z_init", "x1": "x", "dX1t": "x_eta", "dZt": "z_eta"}


def set_time(probabilistic_model, time=None):
    if time is None:
        time = np.arange(0, probabilistic_model.dt, probabilistic_model.time_length)
    return time


def convert_params_names_from_ins(dicts_list, parameter_names=INS_PARAMS_NAMES_DICT):
    output = []
    for lst in ensure_list(dicts_list):
        for dct in ensure_list(lst):
            for p, p_ins in parameter_names.iteritems():
                try:
                    dct[p] = dct[p_ins]
                except:
                    warning("Parameter " + p_ins + " not found in \n" + str(dicts_list))
        output.append(lst)

    return tuple(output)


def convert_params_names_to_ins(dicts_list, parameter_names=INS_PARAMS_NAMES_DICT):
    output = []
    for lst in ensure_list(dicts_list):
        for dct in ensure_list(lst):
            for p, p_ins in parameter_names.iteritems():
                try:
                    dct[p_ins] = dct[p]
                except:
                    warning("Parameter " + p + " not found in \n" + str(dicts_list))
        output.append(lst)

    return tuple(output)


def build_stan_model_dict_to_interface_ins(probabilistic_model, signals, connectivity_matrix, gain_matrix, time=None,
                                           parameter_names=INS_PARAMS_NAMES_DICT):
    """
    Usually takes as input the model_data created with <build_stan_model_dict> and adds fields that are needed to
    interface the ins stan model.
    :param
    """
    active_regions = probabilistic_model.active_regions
    nonactive_regions = probabilistic_model.nonactive_regions
    if time is None:
        time = np.arange(0, probabilistic_model.dt, probabilistic_model.time_length)
    probabilistic_model.parameters = convert_params_names_to_ins(probabilistic_model.parameters, parameter_names)[0]
    vep_data = {"nn": probabilistic_model.number_of_active_regions,
                "nt": probabilistic_model.time_length,
                "ns": probabilistic_model.number_of_target_data,
                "dt": probabilistic_model.dt,
                "I1": probabilistic_model.model_config.Iext1,
                "x0_star_mu": probabilistic_model.parameters["x0"].star_mean[active_regions],
                "x0_star_std": probabilistic_model.parameters["x0"].star_std[active_regions],
                "x0_lo": probabilistic_model.parameters["x0"].low,
                "x0_hi": probabilistic_model.parameters["x0"].high,
                "x_init_mu": probabilistic_model.parameters["x_init"].mean[active_regions],
                "x_init_std": np.mean(probabilistic_model.parameters["x_init"].std),
                "z_init_mu": probabilistic_model.parameters["z_init"].mean[active_regions],
                "z_init_std": np.mean(probabilistic_model.parameters["z_init"].std),
                "x_eq_def": probabilistic_model.model_config.x1eq[nonactive_regions].mean(),
                "tau0": probabilistic_model.tau0,  # 10.0
                "time_scale_mu": probabilistic_model.parameters["time_scale"].mean,
                "time_scale_std": probabilistic_model.parameters["time_scale"].std,
                "k_mu": np.mean(probabilistic_model.parameters["k"].mean),
                "k_std": np.mean(probabilistic_model.parameters["k"].std),
                "SC": connectivity_matrix[active_regions][:, active_regions],
                "SC_var": 5.0,  # 1/36 = 0.02777777,
                "Ic": np.sum(connectivity_matrix[active_regions][:, nonactive_regions], axis=1),
                "sigma_mu": probabilistic_model.parameters["sigma"].mean,
                "sigma_std": probabilistic_model.parameters["sigma"].std,
                "epsilon_mu": probabilistic_model.parameters["epsilon"].mean,
                "epsilon_std": probabilistic_model.parameters["epsilon"].std,
                "sig_hi": 0.025,  # model_data["sig_hi"],
                "amplitude_mu": probabilistic_model.parameters["amplitude"].mean,
                "amplitude_std": probabilistic_model.parameters["amplitude"].std,
                "amplitude_lo": 0.3,
                "offset_mu": probabilistic_model.parameters["offset"].mean,
                "offset_std": probabilistic_model.parameters["offset"].std,
                "seeg_log_power": signals,
                # 9.0 * model_data["signals"] - 4.0,  # scale from (0, 1) to (-4, 5)
                "gain": gain_matrix,
                "time": set_time(probabilistic_model, time),
                "active_regions": np.array(probabilistic_model.active_regions),
                }
    return vep_data



# def build_stan_model_dict(probabilistic_model, signals, 
#                           time=None, sensors=None, gain_matrix=None):
# TODO: needs updating
#     """
#     Builds a dictionary with data needed for stan models.
#     :param probabilistic_model: ProbabilisticModel object
#     :param signals:
#     :param model_inversion: ModelInversionService object
#     :param gain_matrix: array
#     :return: dictionary with stan data
#     """
#     active_regions_flag = np.zeros((probabilistic_model.number_of_regions,), dtype="i")
#     active_regions_flag[probabilistic_model.active_regions] = 1
#     SC = probabilistic_model.model_config.model_connectivity
#     model_data = {"number_of_regions": probabilistic_model.number_of_regions,
#                   "n_times": probabilistic_model.time_length,
#                   "n_signals": probabilistic_model.number_of_signals,
#                   "n_active_regions": probabilistic_model.number_of_active_regions,
#                   "n_nonactive_regions": probabilistic_model.number_of_nonactive_regions,
#                   "n_connections": probabilistic_model.number_of_regions * (probabilistic_model.number_of_regions - 1) / 2,
#                   "active_regions_flag": np.array(active_regions_flag),
#                   "active_regions": np.array(probabilistic_model.active_regions) + 1,  # cmdstan cannot take lists!
#                   "nonactive_regions": np.array(probabilistic_model.nonactive_regions) + 1,  # indexing starts from 1!
#                   "x1eq_min": probabilistic_model.parameters["x1eq"].low,
#                   "x1eq_max": probabilistic_model.parameters["x1eq"].high,
#                   "SC": SC[np.triu_indices(SC.shape[0], 1)],
#                   "dt": probabilistic_model.dt,
#                   "signals": signals,
#                   "time": set_time(probabilistic_model, time),
#                   "mixing": set_mixing(probabilistic_model, sensors, gain_matrix),
#                   "observation_model": probabilistic_model.observation_model.value,
#                   # "observation_expression": np.where(np.in1d(OBSERVATION_MODEL_EXPRESSIONS,
#                   #                                            probabilistic_model.observation_expression))[0][0],
#                   # "euler_method": np.where(np.in1d(EULER_METHODS, probabilistic_model.euler_method))[0][0] - 1,
#                   }
#     for p in ["a", "b", "d", "yc", "Iext1", "slope"]:
#         model_data.update({p: getattr(probabilistic_model.model_config, p)})
#     for p in probabilistic_model.parameters.values():
#         model_data.update({p.name + "_lo": p.low, p.name + "_hi": p.high})
#         if not (isequal_string(p.type, "normal")):
#             model_data.update({p.name + "_loc": p.loc, p.name + "_scale": p.scale,
#                                p.name + "_pdf":
#                                    np.where(np.in1d(ProbabilityDistributionTypes.available_distributions, p.type))[0][
#                                        0],
#                                p.name + "_p": (np.array(p.pdf_params().values()).T * np.ones((2,))).squeeze()})
#     model_data["x1eq_star_loc"] = probabilistic_model.parameters["x1eq_star"].mean
#     model_data["x1eq_star_scale"] = probabilistic_model.parameters["x1eq_star"].std
#     model_data["MC_scale"] = probabilistic_model.MC_scale
#     MCsplit_shape = np.ones(probabilistic_model.parameters["MCsplit"].p_shape)
#     model_data["MCsplit_loc"] = probabilistic_model.parameters["MCsplit"].mean * MCsplit_shape
#     model_data["MCsplit_scale"] = probabilistic_model.parameters["MCsplit"].std * MCsplit_shape
#     model_data["offset_signal_p"] = np.array(probabilistic_model.parameters["offset_signal"].pdf_params().values())
#     return sort_dict(model_data)


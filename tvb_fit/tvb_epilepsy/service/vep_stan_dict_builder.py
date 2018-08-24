
import numpy as np

from tvb_fit.tvb_epilepsy.base.constants.model_inversion_constants import XModes, OBSERVATION_MODELS, X1_MIN, X1_MAX, \
    X1_LOGMU_DEF, X1_LOGSIGMA_DEF, X1_LOGLOC_DEF
from tvb_fit.base.utils.log_error_utils import warning
from tvb_fit.base.utils.data_structures_utils import ensure_list


INS_PARAMS_NAMES_DICT={"K": "k", "MC": "FC", "tau1": "time_scale", "tau0": "tau",
                       "scale": "amplitude", "target_data": "seeg_log_power", "fit_target_data": "mu_seeg_log_power",
                       "x1_init": "x_init", "x1": "x", "dX1t": "x_eta_star", "dZt": "z_eta_star"}


def set_time(probabilistic_model, time=None):
    if time is None:
        time = np.arange(0, probabilistic_model.dt, probabilistic_model.time_length)
    return time


def convert_params_names_from_ins(dicts_list, parameter_names=INS_PARAMS_NAMES_DICT):
    output = []
    for lst in ensure_list(dicts_list):
        for dct in ensure_list(lst):
            for p, p_ins in parameter_names.items():
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
            for p, p_ins in parameter_names.items():
                try:
                    dct[p_ins] = dct[p]
                except:
                    warning("Parameter " + p + " not found in \n" + str(dicts_list))
        output.append(lst)
    return tuple(output)


def build_stan_model_data_dict_to_interface_ins(probabilistic_model, signals, connectivity_matrix, gain_matrix,
                                                time=None, parameter_names=INS_PARAMS_NAMES_DICT):
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
    if "k" in probabilistic_model.parameters.keys():
        k_mu = np.mean(probabilistic_model.parameters["k"].mean)
        k_std = np.mean(probabilistic_model.parameters["k"].std)
    else:
        k_mu = np.mean(probabilistic_model.model_config.K)
        k_std = 1.0
    vep_data = {"nn": probabilistic_model.number_of_active_regions,
                "nt": probabilistic_model.time_length,
                "ns": probabilistic_model.number_of_target_data,
                "dt": probabilistic_model.dt,
                "I1": np.mean(probabilistic_model.model_config.Iext1),
                "x0_star_mu": probabilistic_model.parameters["x0"].star_mean[active_regions],
                "x0_star_std": probabilistic_model.parameters["x0"].star_std[active_regions],
                "x0_lo": probabilistic_model.parameters["x0"].low,
                "x0_hi": probabilistic_model.parameters["x0"].high,
                "x_init_mu": probabilistic_model.parameters["x_init"].mean[active_regions],
                "x_init_std": np.mean(probabilistic_model.parameters["x_init"].std),
                "z_init_mu": probabilistic_model.parameters["z_init"].mean[active_regions],
                "z_init_std": np.mean(probabilistic_model.parameters["z_init"].std),
                "x1_eq_def": probabilistic_model.model_config.x1eq[nonactive_regions].mean(),
                "tau0": probabilistic_model.tau0,  # 10.0
                "time_scale_mu": probabilistic_model.parameters["time_scale"].mean,
                "time_scale_std": probabilistic_model.parameters["time_scale"].std,
                "k_mu": k_mu,
                "k_std": k_std,
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
                "gain": gain_matrix,
                "time": set_time(probabilistic_model, time),
                "active_regions": np.array(probabilistic_model.active_regions),
                }
    return vep_data


def build_stan_model_data_dict(probabilistic_model, signals, connectivity_matrix, gain_matrix,
                               time=None, X1_PRIOR=False):
    """
    Usually takes as input the model_data created with <build_stan_model_dict> and adds fields that are needed to
    interface the ins stan model.
    :param
    """
    active_regions = probabilistic_model.active_regions
    nonactive_regions = probabilistic_model.nonactive_regions
    if time is None:
        time = np.arange(0, probabilistic_model.dt, probabilistic_model.time_length)
    x = probabilistic_model.xmode
    vep_data = {"n_active_regions": probabilistic_model.number_of_active_regions,
                "n_times": probabilistic_model.time_length,
                "n_target_data": probabilistic_model.number_of_target_data,
                "dt": probabilistic_model.dt,
                "Iext1": np.mean(probabilistic_model.model_config.Iext1),
                "XMODE": int(probabilistic_model.xmode == XModes.X1EQMODE.value),
                "x_star_mu": probabilistic_model.parameters[x].star_mean[active_regions],
                "x_star_std": probabilistic_model.parameters[x].star_std[active_regions],
                "x_lo": probabilistic_model.parameters[x].low,
                "x_hi": probabilistic_model.parameters[x].high,
                "X1_PRIOR": int(X1_PRIOR),
                "x1_init_lo": np.min(probabilistic_model.parameters["x1_init"].low),
                "x1_init_hi": np.max(probabilistic_model.parameters["x1_init"].high),
                "x1_init_mu": probabilistic_model.parameters["x1_init"].mean[active_regions],
                "z_init_lo": np.min(probabilistic_model.parameters["z_init"].low),
                "z_init_hi": np.max(probabilistic_model.parameters["z_init"].high),
                "x1_init_std": np.mean(probabilistic_model.parameters["x1_init"].std),
                "z_init_mu": probabilistic_model.parameters["z_init"].mean[active_regions],
                "z_init_std": np.mean(probabilistic_model.parameters["z_init"].std),
                "x1_eq_def": probabilistic_model.model_config.x1eq[nonactive_regions].mean(),
                "SC": connectivity_matrix[active_regions][:, active_regions],
                "Ic": np.sum(connectivity_matrix[active_regions][:, nonactive_regions], axis=1),
                "epsilon_mu": probabilistic_model.parameters["epsilon"].mean,
                "epsilon_std": probabilistic_model.parameters["epsilon"].std,
                "scale_mu": probabilistic_model.parameters["scale"].mean,
                "scale_std": probabilistic_model.parameters["scale"].std,
                "scale_lo": probabilistic_model.parameters["scale"].low,
                "offset_mu": probabilistic_model.parameters["offset"].mean,
                "offset_std": probabilistic_model.parameters["offset"].std,
                "offset_lo": probabilistic_model.parameters["offset"].low,
                "offset_hi": probabilistic_model.parameters["offset"].high,
                "log_target_data": int(probabilistic_model.observation_model == OBSERVATION_MODELS.SEEG_LOGPOWER.value),
                "target_data": signals,
                "gain": gain_matrix,
                "time": set_time(probabilistic_model, time),
                "active_regions": np.array(probabilistic_model.active_regions),
                }
    NO_PRIOR_CONST = 0.001
    for pkey, pflag in zip(["sigma", "tau1", "tau0", "K"], ["SDE", "TAU1_PRIOR", "TAU0_PRIOR", "K_PRIOR"]):
        param = probabilistic_model.parameters.get(pkey, None)
        if param is None:
            mean = np.mean(getattr(probabilistic_model,pkey, NO_PRIOR_CONST))
            vep_data.update({pflag: int(0), pkey+"_mu": mean, pkey + "_std": NO_PRIOR_CONST,
                             pkey + "_lo": mean-NO_PRIOR_CONST, pkey + "_hi": mean+NO_PRIOR_CONST})
        else:
            vep_data.update({pflag: int(1), pkey+"_mu": np.mean(param.mean), pkey + "_std": np.mean(param.std),
                             pkey + "_lo": np.min(param.low), pkey + "_hi": np.max(param.high)})
    return vep_data


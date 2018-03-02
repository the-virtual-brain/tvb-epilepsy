import numpy as np
from copy import deepcopy
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, sort_dict
from tvb_epilepsy.base.constants.model_inversion_constants import OBSERVATION_MODELS
from tvb_epilepsy.base.computations.probability_distributions import ProbabilityDistributionTypes


def build_stan_model_dict(statistical_model, signals, model_inversion, gain_matrix=None):
    """
    Builds a dictionary with data needed for stan models.
    :param statistical_model: StatisticalModel object
    :param signals:
    :param model_inversion: ModelInversionService object
    :param gain_matrix: array
    :return: dictionary with stan data
    """
    active_regions_flag = np.zeros((statistical_model.number_of_regions,), dtype="i")
    active_regions_flag[statistical_model.active_regions] = 1
    if gain_matrix is None:
        if statistical_model.observation_model.find("seeg") >= 0:
            gain_matrix = model_inversion.gain_matrix
            mixing = deepcopy(gain_matrix)
        else:
            mixing = np.eye(statistical_model.number_of_regions)
    if mixing.shape[0] > len(model_inversion.signals_inds):
        mixing = mixing[model_inversion.signals_inds]
    SC = model_inversion.get_SC()
    model_data = {"number_of_regions": statistical_model.number_of_regions,
                  "n_times": statistical_model.n_times,
                  "n_signals": statistical_model.n_signals,
                  "n_active_regions": statistical_model.n_active_regions,
                  "n_nonactive_regions": statistical_model.n_nonactive_regions,
                  "n_connections": statistical_model.number_of_regions * (statistical_model.number_of_regions - 1) / 2,
                  "active_regions_flag": np.array(active_regions_flag),
                  "active_regions": np.array(statistical_model.active_regions) + 1,  # cmdstan cannot take lists!
                  "nonactive_regions": np.where(1 - active_regions_flag)[0] + 1,  # indexing starts from 1!
                  "x1eq_min": statistical_model.x1eq_min,
                  "x1eq_max": statistical_model.x1eq_max,
                  "SC": SC[np.triu_indices(SC.shape[0], 1)],
                  "dt": statistical_model.dt,
                  # "euler_method": np.where(np.in1d(EULER_METHODS, statistical_model.euler_method))[0][0] - 1,
                  "observation_model": np.where(np.in1d(OBSERVATION_MODELS,
                                                        statistical_model.observation_model))[0][0],
                  # "observation_expression": np.where(np.in1d(OBSERVATION_MODEL_EXPRESSIONS,
                  #                                            statistical_model.observation_expression))[0][0],
                  "signals": signals,
                  "time": model_inversion.time,
                  "mixing": mixing}
    for key, val in model_inversion.epileptor_parameters.iteritems():
        model_data.update({key: val})
    for p in statistical_model.parameters.values():
        model_data.update({p.name + "_lo": p.low, p.name + "_hi": p.high})
        if not (isequal_string(p.type, "normal")):
            model_data.update({p.name + "_loc": p.loc, p.name + "_scale": p.scale,
                               p.name + "_pdf":
                                   np.where(np.in1d(ProbabilityDistributionTypes.available_distributions, p.type))[0][
                                       0],
                               p.name + "_p": (np.array(p.pdf_params().values()).T * np.ones((2,))).squeeze()})
    model_data["x1eq_star_loc"] = statistical_model.parameters["x1eq_star"].mean
    model_data["x1eq_star_scale"] = statistical_model.parameters["x1eq_star"].std
    model_data["MC_scale"] = statistical_model.MC_scale
    MCsplit_shape = np.ones(statistical_model.parameters["MCsplit"].p_shape)
    model_data["MCsplit_loc"] = statistical_model.parameters["MCsplit"].mean * MCsplit_shape
    model_data["MCsplit_scale"] = statistical_model.parameters["MCsplit"].std * MCsplit_shape
    model_data["offset_signal_p"] = np.array(statistical_model.parameters["offset_signal"].pdf_params().values())
    return sort_dict(model_data)


def build_stan_model_dict_to_interface_ins(model_data, statistical_model, model_inversion, gain_matrix=None):
    """
    Usually takes as input the model_data created with <build_stan_model_dict> and adds fields that are needed to
    interface the ins stan model.
    :param model_data: dictionary
    :param statistical_model: StatisticalModel object
    :param model_inversion:  ModelInversionService object
    :param gain_matrix: ndarray
    :return: dict with data needed for stan models
    """
    active_regions = model_data["active_regions"] - 1
    nonactive_regions = model_data["nonactive_regions"] - 1
    SC = statistical_model.parameters["MC"].mode
    act_reg_ones = np.ones((model_data["n_active_regions"],))
    x0_lo = -4.0
    x0_hi = 0.0
    x0_star_mu = x0_hi - model_inversion.x0[active_regions].mean() * act_reg_ones
    x0_star_std = np.minimum((x0_hi - x0_lo) / 4.0, x0_star_mu / 3.0) * act_reg_ones
    x_init_mu = statistical_model.parameters["x1init"].mean[active_regions].mean() * act_reg_ones
    z_init_mu = statistical_model.parameters["zinit"].mean[active_regions].mean() * act_reg_ones
    vep_data = {"nn": model_data["n_active_regions"],
                "nt": model_data["n_times"],
                "ns": model_data["n_signals"],
                "dt": model_data["dt"],
                "I1": model_data["Iext1"],
                "x0_star_mu": x0_star_mu, # model_inversion.x0[statistical_model.active_regions],
                "x0_star_std": x0_star_std,
                "x0_lo": x0_lo,
                "x0_hi": x0_hi,
                "x_init_mu": x_init_mu,
                "z_init_mu": z_init_mu,
                "x_eq_def": model_inversion.x1EQ[nonactive_regions].mean(),
                "init_std": np.mean(statistical_model.parameters["x1init"].std),
                "tau0": 10.0,  # statistical_model.parameters["tau0"].mean,
                # "K_lo": statistical_model.parameters["K"].low,
                # "K_u": statistical_model.parameters["K"].mode,
                # "K_v": statistical_model.parameters["K"].var,
                "time_scale_mu": statistical_model.parameters["tau1"].mean,
                "time_scale_std": statistical_model.parameters["tau1"].std,
                "k_mu": statistical_model.parameters["K"].mean/2,
                "k_std": statistical_model.parameters["K"].std/2,
                "SC": SC[active_regions][:, active_regions],
                "SC_var": 5.0,  # 1/36 = 0.02777777,
                "Ic": np.sum(SC[active_regions][:, nonactive_regions], axis=1),
                "sigma_mu": statistical_model.parameters["sig"].mean,
                "sigma_std": statistical_model.parameters["sig"].std,
                "epsilon_mu": statistical_model.parameters["eps"].mean,
                "epsilon_std": statistical_model.parameters["eps"].std,
                "sig_hi": 0.025,  # model_data["sig_hi"],
                "amplitude_mu": statistical_model.parameters["scale_signal"].mean,
                "amplitude_std": statistical_model.parameters["scale_signal"].std / 6,
                "amplitude_lo": 0.3,
                "offset_mu": statistical_model.parameters["offset_signal"].mean,
                "offset_std": statistical_model.parameters["offset_signal"].std,
                "seeg_log_power": model_data["signals"],
                # 9.0 * model_data["signals"] - 4.0,  # scale from (0, 1) to (-4, 5)
                "time": model_data["time"]
                }
    if gain_matrix is None:
        if statistical_model.observation_model.find("seeg") >= 0:
            gain_matrix = model_inversion.gain_matrix
            mixing = deepcopy(gain_matrix)[:, statistical_model.active_regions]
        else:
            mixing = np.eye(vep_data["nn"])
        if mixing.shape[0] > vep_data["ns"]:
            mixing = mixing[model_inversion.signals_inds]
        vep_data["gain"] = mixing
    return vep_data, x0_star_mu, x_init_mu, z_init_mu

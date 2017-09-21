import numpy as np
import pystan as ps

from tvb_epilepsy.base.constants import X1_DEF, X1_EQ_CR_DEF, X0_DEF, X0_CR_DEF
from tvb_epilepsy.base.utils import raise_not_implemented_error
from tvb_epilepsy.base.computations.calculations_utils import calc_x0cr_r
from tvb_epilepsy.base.computations.equilibrium_computation import calc_eq_z
from tvb_epilepsy.service.sampling_service import gamma_from_mu_std
from tvb_epilepsy.service.epileptor_model_factory import model_noise_intensity_dict

def prepare_for_fitting(model_configuration, hypothesis, fs, signals, model_path, active_regions=None,
                        active_regions_th=0.1, observation_model=3, mixing=None, **kwargs):

    active_regions_flag = np.zeros((hypothesis.number_of_regions, ), dtype="i")

    if active_regions is None:
        if len(hypothesis.propagation_strengths) > 0:
            active_regions = np.where(hypothesis.propagation_strengths / np.max(hypothesis.propagation_strengths)
                                      > active_regions_th)[0]
        else:
            raise_not_implemented_error("There is no other way of automatic selection of " +
                                        "active regions implemented yet!")

    active_regions_flag[active_regions] = 1
    n_active = len(active_regions)

    # Gamma distributions' parameters
    tau1 = gamma_from_mu_std(kwargs.get("tau1_mu", 0.2), kwargs.get("tau1_std", 0.1))
    tau0 = gamma_from_mu_std(kwargs.get("tau0_mu", 10000.0), kwargs.get("tau0_std", 10000.0))
    K = gamma_from_mu_std(kwargs.get("K_mu", 10.0 / hypothesis.number_of_regions),
                          kwargs.get("K_std", 10.0 / hypothesis.number_of_regions))
    # zero effective connectivity:
    conn0 = gamma_from_mu_std(kwargs.get("conn0_mu", 0.001), kwargs.get("conn0_std", 0.001))
    sig_mu = np.mean(model_noise_intensity_dict["EpileptorDP2D"])
    sig = gamma_from_mu_std(kwargs.get("sig_mu", sig_mu), kwargs.get("sig_std", sig_mu))
    sig_eq_mu = 0.1/3.0
    sig_eq = gamma_from_mu_std(kwargs.get("sig_eq_mu", sig_eq_mu), kwargs.get("sig_eq_std", sig_eq_mu))
    sig_init_mu = sig_eq_mu
    sig_init = gamma_from_mu_std(kwargs.get("sig_init_mu", sig_init_mu), kwargs.get("sig_init_std", sig_init_mu))

    if mixing is None:
        observation_model = 3;

    data = {"n_regions": hypothesis.number_of_regions,
            "n_active_regions": n_active,
            "n_nonactive_regions": hypothesis.number_of_regions-n_active,
            "active_regions_flag": active_regions_flag,
            "n_time": signals.shape[0],
            "n_signals": signals.shape[1],
            "x0_nonactive": model_configuration.x0[~active_regions_flag.astype("bool")],
            "x1eq0": model_configuration.x1EQ,
            "x1eq_lo": kwargs.get("x1eq_lo", -2.0),
            "x1eq_hi": kwargs.get("x1eq_hi", X1_EQ_CR_DEF),
            "x1init_lo": kwargs.get("x1init_lo", -2.0),
            "x1init_hi": kwargs.get("x1init_hi", -1.0),
            "x1_lo": kwargs.get("x1_lo", -2.5),
            "x1_hi": kwargs.get("x1_hi", 1.5),
            "z_lo": kwargs.get("z_lo", 2.0),
            "z_hi": kwargs.get("z_hi", 5.0),
            "tau1_lo": kwargs.get("tau1_lo", 0.001),
            "tau1_hi": kwargs.get("tau1_hi", 1.0),
            "tau0_lo": kwargs.get("tau0_lo", 1000.0),
            "tau0_hi": kwargs.get("tau0_hi", 100000.0),
            "tau1_a": kwargs.get("tau1_a", tau1["alpha"]),
            "tau1_b": kwargs.get("tau1_b", tau1["beta"]),
            "tau0_a": kwargs.get("tau0_a", tau0["alpha"]),
            "tau0_b": kwargs.get("tau0_b", tau0["beta"]),
            "SC": model_configuration.connectivity_matrix,
            "SC_sig": kwargs.get("SC_sig", 0.1),
            "K_lo": kwargs.get("K_lo", 1.0 / hypothesis.number_of_regions),
            "K_hi": kwargs.get("K_hi", 100.0 / hypothesis.number_of_regions),
            "K_a": kwargs.get("K_a", K["alpha"]),
            "K_b": kwargs.get("K_b", K["beta"]),
            "gamma0": kwargs.get("gamma0", np.array([conn0["alpha"], conn0["beta"]])),
            "dt": 1000.0 / fs,
            "sig_hi": kwargs.get("sig_hi", 1.0 / fs),
            "sig_a": kwargs.get("sig_a", sig["alpha"]),
            "sig_b": kwargs.get("sig_b", sig["beta"]),
            "sig_eq_hi": kwargs.get("sig_eq_hi", 3*sig_eq_mu),
            "sig_eq_a": kwargs.get("sig_eq_a", sig_eq["alpha"]),
            "sig_eq_b": kwargs.get("sig_eq_b", sig_eq["beta"]),
            "sig_init_hi": kwargs.get("sig_init_hi", 3 * sig_init_mu),
            "sig_init_a": kwargs.get("sig_init_a", sig_init["alpha"]),
            "sig_init_b": kwargs.get("sig_init_b", sig_init["beta"]),
            "observation_model": observation_model,
            "signals": signals,
            "mixing": mixing,
            "eps_hi": kwargs.get("eps_hi", (np.max(signals.flatten()) - np.min(signals.flatten()) / 100.0)),
            "eps_x0": kwargs.get("eps_x0", 0.1),
    }

    zeq_lo = calc_eq_z(data["x1eq_lo"], data["yc"], data["Iext1"], "2d", x2=0.0, slope=data["slope"], a=data["a"],
                        b=data["b"], d=data["d"])
    zeq_hi = calc_eq_z(data["x1eq_hi"], data["yc"], data["Iext1"], "2d", x2=0.0, slope=data["slope"], a=data["a"],
                        b=data["b"], d=data["d"])
    data.update({"zeq_lo": kwargs.get("zeq_lo", zeq_lo),
                 "zeq_hi": kwargs.get("zeq_hi", zeq_hi)})
    data.update({"zinit_hi": kwargs.get("zinit_hi", zeq_hi+sig_init_mu),
                 "zeq_hi": kwargs.get("zeq_hi", zeq_lo-sig_init_mu)})

    for p in ["a", "b", "d", "yc", "Iext1", "slope"]:

        temp = getattr(model_configuration, p)
        if isinstance(temp, ("ndarray", "list")):
            if np.all(temp[0], np.array(temp)):
                temp = temp[0]
            else:
                raise_not_implemented_error("Statistical models where not all regions have the same value " +
                                            " for parameter " + p + " are not implemented yet!")
        data.update({p: temp})

    x0cr, rx0 = calc_x0cr_r(data["yc"], data["Iext1"], data["a"], data["b"], data["d"], zmode=np.array("lin"),
                            x1_rest=X1_DEF, x1_cr=X1_EQ_CR_DEF, x0def=X0_DEF, x0cr_def=X0_CR_DEF, test=False,
                            shape=None, calc_mode="non_symbol")

    data.update({"x0cr": x0cr, "rx0": rx0})

    model = ps.StanModel(file=model_path)

    return model, data


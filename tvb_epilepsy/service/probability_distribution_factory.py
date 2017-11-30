
from collections import OrderedDict

import numpy as np
from scipy.optimize import minimize

from tvb_epilepsy.base.utils.log_error_utils import raise_value_error, warning
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, dicts_of_lists_to_lists_of_dicts


AVAILABLE_DISTRIBUTIONS = ["uniform", "normal", "gamma", "lognormal", "exponential", "beta", "chisquare",
                           "binomial", "bernoulli", "poisson"]

CONSTRAINT_ABS_TOL = 0.001


def generate_distribution(distrib_type, target_shape=None, **target):
    if np.in1d(distrib_type.lower(), AVAILABLE_DISTRIBUTIONS):
        exec("from tvb_epilepsy.base.model.statistical_models.probability_distributions."
             + distrib_type.lower() + "_distribution import " + distrib_type.title() + "Distribution")
        distribution = eval(distrib_type.title() + "Distribution()")
        if len(target) > 0:
            try:
                distribution.update(**target)
            except:
                target = compute_pdf_params(distribution.type, target)
                distribution.update_params(**target)
        if isinstance(target_shape, tuple):
            distribution.__shape_parameters__(target_shape)
        return distribution
    else:
        raise_value_error(distrib_type + " is not one of the available distributions!: " + str(AVAILABLE_DISTRIBUTIONS))


# This function converts the parameters' vector to the parameters' dictionary
def construct_pdf_params_dict(p, pdf):
    # Make sure p in denormalized and of float64 type
    # TODO solve the problem for integer distribution parameters...
    p = p.astype(np.float64)
    params = OrderedDict()
    for ik, p_key in enumerate(pdf.pdf_params().keys()):
        inds = range(ik, len(p), pdf.n_params)
        params.update({p_key: np.reshape(p[inds], pdf.p_shape)})
    return params


# Scalar objective  function
def fobj(p, pdf, target_stats):
    params = construct_pdf_params_dict(p, pdf)
    pdf.__update_params__(check_constraint=False, **params)
    f = 0.0
    norm = 0.0
    for ts_key, ts_val in target_stats.iteritems():
        norm += ts_val ** 2
        f += (getattr(pdf, "calc_" + ts_key)(use="manual") - ts_val) ** 2
    # if np.isnan(f) or np.isinf(f):
    #     print("WTF?")
    if norm > 0.0:
        f /= norm
    return f


# Vector constraint function. By default expr >= 0
def fconstr(p, pdf):
    params = construct_pdf_params_dict(p, pdf)
    pdf.__update_params__(check_constraint=False, **params)
    f = pdf.constraint() - CONSTRAINT_ABS_TOL
    return f


# Vector constraint function for gamma distribution median optimization. By default expr >= 0
def fconstr_gamma_mode(p, pdf):
    params = construct_pdf_params_dict(p, pdf)
    f = params["shape"] - 1.0 - CONSTRAINT_ABS_TOL
    return f


# Vector constraint function for beta distribution mode and median optimization. By default expr >= 0
def fconstr_beta_mode_median(p, pdf):
    params = construct_pdf_params_dict(p, pdf)
    f = np.stack(params.values()) - 1.0 - CONSTRAINT_ABS_TOL
    return f


def prepare_contraints(distribution, target_stats):
    # Preparing contraints:
    constraints = [{"type": "ineq", "fun": lambda p: fconstr(p, distribution)}]
    if isequal_string(distribution.type, "gamma") and np.any(np.in1d("mode", target_stats.keys())):
        constraints.append({"type": "ineq", "fun": lambda p: fconstr_gamma_mode(p, distribution)})
    elif isequal_string(distribution.type, "beta") and np.any(np.in1d(["mode", "median"], target_stats.keys())):
        constraints.append({"type": "ineq", "fun": lambda p: fconstr_beta_mode_median(p, distribution)})
    return constraints


def prepare_target_stats(distribution, target_stats):
    # Make sure that the shapes of target stats are all matching one to the other:
    target_shape = np.ones(())
    try:
        for ts in target_stats.values():
            target_shape = target_shape * np.ones(np.array(ts).shape)
    except:
        raise_value_error("Target statistics (" + str([np.array(ts).shape for ts in target_stats.values()]) +
                          ") and distribution (" + str(distribution.p_shape) + ") shapes do not propagate!")
    for ts_key in target_stats.keys():
        target_stats[ts_key] *= target_shape
        if  np.sum(target_stats[ts_key].shape) > 0:
            target_stats[ts_key] = target_stats[ts_key].flatten()
    target_size = target_shape.size
    target_shape = target_shape.shape
    target_stats_array = np.around(np.vstack(target_stats.values()).T, decimals=2)
    target_stats_unique = np.unique(target_stats_array, axis=0)
    # target_stats_unique = np.vstack({tuple(row) for row in target_stats_array})
    target_stats_unique = dict(zip(target_stats.keys(),
                                   [np.around(target_stats_unique[:, ii], decimals=3) for ii in range(distribution.n_params)]))
    target_stats_unique = dicts_of_lists_to_lists_of_dicts(target_stats_unique)
    return target_stats_unique, target_stats_array, target_shape, target_size


def prepare_intial_condition(distribution, low_limit=-10.0, high_limit=10):
    # TODO: find a better to initialize this...
    # Preparing initial conditions' parameters' vector:
    p0 = np.stack(distribution.pdf_params().values())
    # Bounding initial condition:
    p0[np.where(p0 > high_limit)[0]] = high_limit
    p0[np.where(p0 < low_limit)[0]] = low_limit
    return p0


def compute_pdf_params(distrib_type, target_stats):
    distribution = generate_distribution(distrib_type, target_shape=())
    # Check if the number of target stats is exactly the same as the number of distribution parameters to optimize:
    if len(target_stats) != distribution.n_params:
        raise_value_error("Target parameters are " + str(len(target_stats)) +
                          ", whereas the characteristic parameters of distribution " + distribution.type +
                          " are " + str(distribution.n_params) + "!")
    target_stats_unique, target_stats_array, target_shape, target_size = \
        prepare_target_stats(distribution, target_stats)
    constraints = prepare_contraints(distribution, target_stats)
    p0 = prepare_intial_condition(distribution, low_limit=-10.0, high_limit=10)
    # p0 = [2.0, 2.0]
    # Run optimization
    sol_params = np.ones((target_size, distribution.n_params)) * p0
    sol_params_sum = np.zeros(p0.shape)
    for ii, ts in enumerate(target_stats_unique):
        if ii > 0:
            p0 = sol_params_sum / ii
        try:
            # For: "COBYLA"  options={tol": 10 ** -3, "catol": CONSTRAINT_ABS_TOL, 'rhobeg': CONSTRAINT_ABS_TOL}
            sol = minimize(fobj, p0, args=(distribution, ts), method="SLSQP", constraints=constraints, tol=None,
                           options={"ftol": 10 ** -6})
            if sol.success:
                if np.any([np.any(np.isnan(sol.x)), np.any(np.isinf(sol.x))]):
                    raise_value_error("nan or inf values in solution x\n" + sol.message)
                if sol.fun > 10 ** -3:
                    warning("Not accurate solution! sol.fun = " + str(sol.fun))
                inds = np.where([np.all(target_stats_array[ii] == np.array(ts.values())) for ii in range(target_size)])[0]
                sol_params[inds] = sol.x
                sol_params_sum += sol.x
            else:
                raise_value_error(sol.message)
        except:
            print("WTF?")
    sol_params= dict(zip(distribution.pdf_params().keys(),
                    [np.reshape(sol_params[:, ii], target_shape) for ii in range(distribution.n_params)]))
    return sol_params

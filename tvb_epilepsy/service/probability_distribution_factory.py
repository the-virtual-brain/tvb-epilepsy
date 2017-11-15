
from collections import OrderedDict

import numpy as np
from scipy.optimize import minimize

from tvb_epilepsy.base.utils.log_error_utils import raise_value_error, warning
from tvb_epilepsy.base.utils.data_structures_utils import shape_to_size, dicts_of_lists, \
                                                    dicts_of_lists_to_lists_of_dicts, list_of_dicts_to_dicts_of_ndarrays


AVAILABLE_DISTRIBUTIONS = ["uniform", "normal", "gamma", "lognormal", "exponential", "beta", "chisquare",
                           "binomial", "bernoulli", "poisson"]


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
    for ts_key, ts_val in target_stats.iteritems():
        f += np.sum((getattr(pdf, "calc_" + ts_key)(use="manual") - ts_val) ** 2)
    if np.isnan(f):
        print("WTF?")
    return f.flatten()


# Vector constraint function. By default expr >= 0
def fconstr(p, pdf):
    params = construct_pdf_params_dict(p, pdf)
    pdf.__update_params__(check_constraint=False, **params)
    return pdf.constraint()


def compute_pdf_params(distrib_type, target_stats):
    distribution = generate_distribution(distrib_type, target_shape=())
    # Check if the number of target stats is exactly the same as the number of distribution parameters to optimize:
    if len(target_stats) != distribution.n_params:
        raise_value_error("Target parameters are " + str(len(target_stats)) +
                          ", whereas the characteristic parameters of distribution " + distribution.type +
                          " are " + str(distribution.n_params) + "!")
    # Preparing initial conditions' parameters' vector:
    params_vector = np.stack(distribution.pdf_params().values())
    # Bounding initial condition:
    params_vector[np.where(params_vector > 10.0)[0]] = 10.0
    params_vector[np.where(params_vector < -10.0)[0]] = -10.0
    # Make sure that the shapes of target stats are all matching one to the other:
    target_shape = np.ones((1,))
    try:
        for ts in target_stats.values():
            target_shape = target_shape * np.ones(np.array(ts).shape)
    except:
        raise_value_error("Target statistics (" + str([np.array(ts).shape for ts in target_stats.values()]) +
                          ") and distribution (" + str(distribution.p_shape) + ") shapes do not propagate!")
    for ts_key in target_stats.keys():
        target_stats[ts_key] *= target_shape
    # Preparing contraints:
    constraints = {"type": "ineq", "fun": lambda p: fconstr(p, distribution)}
    # Run optimization
    target_size = shape_to_size(target_shape)
    target_stats = dicts_of_lists(target_stats, n=target_size)
    target_stats = dicts_of_lists_to_lists_of_dicts(target_stats)
    sol_params = []
    for ts in target_stats:
        sol = minimize(fobj, params_vector, args=(distribution, ts), method="COBYLA", constraints=constraints,
                       options={"tol": 10 ** -6, "catol": 10 ** -6})
        if sol.success:
            if np.any([np.any(np.isnan(sol.x)), np.any(np.isinf(sol.x))]):
                raise_value_error("nan or inf values in solution x\n" + sol.message)
            if sol.fun > 10 ** -6:
                warning("Not accurate solution! sol.fun = " + str(sol.fun))
        sol_params.append(construct_pdf_params_dict(sol.x, distribution))
    else:
        raise_value_error(sol.message)
    sol_params = list_of_dicts_to_dicts_of_ndarrays(sol_params, shape=target_shape)
    return sol_params



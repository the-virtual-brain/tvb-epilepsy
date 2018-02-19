import numpy as np
from scipy.optimize import minimize
from collections import OrderedDict
from tvb_epilepsy.base.computations.probability_distributions import ProbabilityDistributionTypes
from tvb_epilepsy.base.computations.probability_distributions.bernoulli_distribution import BernoulliDistribution
from tvb_epilepsy.base.computations.probability_distributions.beta_distribution import BetaDistribution
from tvb_epilepsy.base.computations.probability_distributions.binomial_distribution import BinomialDistribution
from tvb_epilepsy.base.computations.probability_distributions.chisquare_distribution import ChisquareDistribution
from tvb_epilepsy.base.computations.probability_distributions.exponential_distribution import ExponentialDistribution
from tvb_epilepsy.base.computations.probability_distributions.gamma_distribution import GammaDistribution
from tvb_epilepsy.base.computations.probability_distributions.lognormal_distribution import LognormalDistribution
from tvb_epilepsy.base.computations.probability_distributions.normal_distribution import NormalDistribution
from tvb_epilepsy.base.computations.probability_distributions.poisson_distribution import PoissonDistribution
from tvb_epilepsy.base.computations.probability_distributions.uniform_distribution import UniformDistribution
from tvb_epilepsy.base.utils.log_error_utils import raise_value_error, initialize_logger
from tvb_epilepsy.base.utils.data_structures_utils import isequal_string, dicts_of_lists_to_lists_of_dicts

CONSTRAINT_ABS_TOL = 0.01

logger = initialize_logger(__name__)


def probability_distribution_factory(distrib_type, get_instance=True):
    """
    Factory function that will return by default an instance for the distrib_type. If get_instance=False it will return
    only the class name.
    :param distrib_type: type of the wanted distribution.
    :param get_instance: specifies if the return type is an object or a class name.
    :return: ProbabilityDistribution derived object or class name
    """
    if distrib_type == ProbabilityDistributionTypes.UNIFORM: return UniformDistribution() if get_instance else UniformDistribution
    if distrib_type == ProbabilityDistributionTypes.NORMAL: return NormalDistribution() if get_instance else NormalDistribution
    if distrib_type == ProbabilityDistributionTypes.GAMMA: return GammaDistribution() if get_instance else GammaDistribution
    if distrib_type == ProbabilityDistributionTypes.LOGNORMAL: return LognormalDistribution() if get_instance else LognormalDistribution
    if distrib_type == ProbabilityDistributionTypes.EXPONENTIAL: return ExponentialDistribution() if get_instance else ExponentialDistribution
    if distrib_type == ProbabilityDistributionTypes.BETA: return BetaDistribution() if get_instance else BetaDistribution
    if distrib_type == ProbabilityDistributionTypes.CHISQUARE: return ChisquareDistribution() if get_instance else ChisquareDistribution
    if distrib_type == ProbabilityDistributionTypes.BINOMIAL: return BinomialDistribution() if get_instance else BinomialDistribution
    if distrib_type == ProbabilityDistributionTypes.POISSON: return PoissonDistribution() if get_instance else PoissonDistribution
    if distrib_type == ProbabilityDistributionTypes.BERNOULLI: return BernoulliDistribution() if get_instance else BernoulliDistribution

    logger.warning("Distribution %s is not implemented or the name is wrong")
    return None


# TODO: this functionality should be reviewed. At first sight it looks like it does the same thing on both branches
def generate_distribution(distrib_type, loc=0.0, scale=1.0, use="manual", target_shape=None, optimize_pdf=True,
                          **pdf_params):
    if np.in1d(distrib_type.lower(), ProbabilityDistributionTypes.available_distributions):
        distribution = probability_distribution_factory(distrib_type.lower())  # generate an agnostic distribution
        success = True
        if len(pdf_params) > 0:
            distribution.update_params(loc, scale, use, **pdf_params)  # update with desired parameters
            # test whether the distribution is correctly set:
            for p_key, p_val in pdf_params.iteritems():
                if np.any(p_val != getattr(distribution, p_key)):
                    success = False
        if success is False:
            # if the distribution is not correct, try to optimize it
            if optimize_pdf:
                distribution = optimize_distribution(distrib_type, loc, scale, use, target_shape=None,
                                                     **pdf_params)
                success = True
                for p_key, p_val in pdf_params.iteritems():
                    if np.any(np.abs(p_val - getattr(distribution, p_key)) > 0.1):
                        success = False
            # if still we don't get the desired distribution raise an error
        if success is False:
            raise_value_error("Cannot generate probability distribution of type " + distrib_type +
                              " with parameters " + str(pdf_params) + " !")
        if isinstance(target_shape, tuple):
            distribution.__shape_parameters__(target_shape, loc, scale, use)
        return distribution
    else:
        raise_value_error(distrib_type + " is not one of the available distributions!: " + str(
            ProbabilityDistributionTypes.available_distributions))


def optimize_distribution(distrib_type, loc=0.0, scale=1.0, use="manual", target_shape=None, **target):
    distribution = generate_distribution(distrib_type, loc, scale, use, target_shape=None, **target)
    if len(target) > 0:
        try:
            distribution.update_params(loc, scale, use, **target)
        except:
            target = compute_pdf_params(distribution.type, target, loc, scale, use)
            distribution.update_params(loc, scale, use, **target)
    if isinstance(target_shape, tuple):
        distribution.__shape_parameters__(target_shape, loc, scale, use)
    return distribution


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


# We take into consideration loc and scale

# Scalar objective  function
def fobj(p, pdf, target_stats, loc=0.0, scale=1.0, use="manual"):
    params = construct_pdf_params_dict(p, pdf)
    pdf.update_params(loc, scale, use, **params)
    f = 0.0
    norm = 0.0
    for ts_key, ts_val in target_stats.iteritems():
        # norm += ts_val ** 2
        try:
            f += (getattr(pdf, "_calc_" + ts_key)(loc, scale, use) - ts_val) ** 2
        except:
            try:
                f += (getattr(pdf, ts_key) - ts_val) ** 2
            except:
                raise_value_error("Failed to calculate and/or return target statistic or parameter " + ts_key + " !")
    # if np.isnan(f) or np.isinf(f):
    #     print("WTF?")
    # if norm > 0.0:
    #     f /= norm
    return f


# Vector constraint function. By default expr >= 0
def fconstr(p, pdf, loc=0.0, scale=1.0, use="manual"):
    params = construct_pdf_params_dict(p, pdf)
    pdf.__update_params__(loc, scale, use, check_constraint=False, **params)
    f = pdf.constraint() - CONSTRAINT_ABS_TOL
    return f


# Vector constraint function for gamma distribution median optimization. By default expr >= 0
def fconstr_gamma_mode(p, pdf, loc=0.0, scale=1.0, use="manual"):
    params = construct_pdf_params_dict(p, pdf)
    f = params["alpha"] - 1.0 - CONSTRAINT_ABS_TOL
    return f


# Vector constraint function for beta distribution mode and median optimization. By default expr >= 0
def fconstr_beta_mode_median(p, pdf):
    params = construct_pdf_params_dict(p, pdf)
    f = np.stack(params.values()) - 1.0 - CONSTRAINT_ABS_TOL
    return f


def prepare_constraints(distribution, target_stats, loc=0.0, scale=1.0, use="manual"):
    # Preparing constraints:
    constraints = [{"type": "ineq", "fun": lambda p: fconstr(p, distribution, loc, scale, use)}]
    if isequal_string(distribution.type, "gamma") and np.any(np.in1d("mode", target_stats.keys())):
        constraints.append({"type": "ineq", "fun": lambda p: fconstr_gamma_mode(p, distribution)})
    elif isequal_string(distribution.type, "beta") and np.any(np.in1d(["mode", "median"], target_stats.keys())):
        constraints.append({"type": "ineq", "fun": lambda p: fconstr_beta_mode_median(p, distribution)})
    return constraints


def prepare_target_stats(distribution, target_stats, loc=0.0, scale=1.0):
    # Make sure that the shapes of target stats are all matching one to the other:
    target_shape = np.ones(()) * loc * scale
    target_shape = np.ones(target_shape.shape)
    try:
        for ts in target_stats.values():
            target_shape = target_shape * np.ones(np.array(ts).shape)
    except:
        raise_value_error("Target statistics (" + str([np.array(ts).shape for ts in target_stats.values()]) +
                          ") and distribution (" + str(distribution.p_shape) + ") shapes do not propagate!")
    for ts_key in target_stats.keys():
        target_stats[ts_key] *= target_shape
        if np.sum(target_stats[ts_key].shape) > 0:
            target_stats[ts_key] = target_stats[ts_key].flatten()
    target_size = target_shape.size
    target_shape = target_shape.shape
    target_stats_array = np.around(np.vstack(target_stats.values()).T, decimals=2)
    target_stats_unique = np.unique(target_stats_array, axis=0)
    # target_stats_unique = np.vstack({tuple(row) for row in target_stats_array})
    target_stats_unique = dict(zip(target_stats.keys(),
                                   [np.around(target_stats_unique[:, ii], decimals=3) for ii in
                                    range(distribution.n_params)]))
    target_stats_unique = dicts_of_lists_to_lists_of_dicts(target_stats_unique)
    return target_stats_unique, target_stats_array, target_shape, target_size


def prepare_intial_condition(pdf, target_stats):  # , low_limit=-10.0, high_limit=10
    # TODO: find a better to initialize this...
    # Preparing initial conditions' parameters' vector:
    p0 = dict(pdf.pdf_params())
    for ts_key, ts_val in target_stats.iteritems():
        # norm += ts_val ** 2
        if p0.get(ts_key, None) is not None:
            p0[ts_key] = ts_val
    p0 = np.stack(p0.values())
    # # Bounding initial condition:
    # p0[np.where(p0 > high_limit)[0]] = high_limit
    # p0[np.where(p0 < low_limit)[0]] = low_limit
    return p0


def compute_pdf_params(distrib_type, target_stats, loc=0.0, scale=1.0, use="manual"):
    distribution = generate_distribution(distrib_type)
    # Check if the number of target stats is exactly the same as the number of distribution parameters to optimize:
    if len(target_stats) != distribution.n_params:
        raise_value_error("Target parameters are " + str(len(target_stats)) +
                          ", whereas the characteristic parameters of distribution " + distribution.type +
                          " are " + str(distribution.n_params) + "!")
    target_stats_unique, target_stats_array, target_shape, target_size = \
        prepare_target_stats(distribution, target_stats, loc, scale)
    constraints = prepare_constraints(distribution, target_stats, loc, scale, use)
    p0 = prepare_intial_condition(distribution, target_stats)
    # p0 = [2.0, 2.0]
    # Run optimization
    sol_params = np.ones((target_size, distribution.n_params)) * p0
    sol_params_sum = np.zeros(p0.shape)
    for ii, ts in enumerate(target_stats_unique):
        if ii > 0:
            p0 = sol_params_sum / ii
        # For: "COBYLA"  options={tol": 10 ** -3, "catol": CONSTRAINT_ABS_TOL, 'rhobeg': CONSTRAINT_ABS_TOL}
        sol = minimize(fobj, p0, args=(distribution, ts), method="SLSQP", constraints=constraints, tol=None,
                       options={"ftol": 10 ** -6, "eps": CONSTRAINT_ABS_TOL})  # p0.min() / 100
        if sol.success:
            if np.any([np.any(np.isnan(sol.x)), np.any(np.isinf(sol.x))]):
                raise_value_error("nan or inf values in solution x\n" + sol.message)
            if sol.fun > 10 ** -3:
                logger.warning("Not accurate solution! sol.fun = " + str(sol.fun))
            inds = np.where([np.all(target_stats_array[ii] == np.array(ts.values())) for ii in range(target_size)])[0]
            sol_params[inds] = sol.x
            sol_params_sum += sol.x
        else:
            raise_value_error(sol.message)
    sol_params = dict(zip(distribution.pdf_params().keys(),
                          [np.reshape(sol_params[:, ii], target_shape) for ii in range(distribution.n_params)]))
    return sol_params

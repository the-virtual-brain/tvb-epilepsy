
from collections import OrderedDict

import numpy as np

from tvb_epilepsy.base.utils.data_structures_utils import sort_dict, isequal_string


STAN_STATIC_OPTIONS = sort_dict({"int_time": 2*np.pi})  # int_time > 0

STAN_NUTS_OPTIONS = sort_dict({"max_depth": 10})  # int > 0

STAN_HMC_OPTIONS = sort_dict({"engine": "nuts",
                              "metric": "diag_e",     # others: "unit_e", "dense_e"
                              "stepsize": 1,          # stepsize > 0
                              "stepsize_jitter": 0})  # 0 <= stepsize_jitter <= 1

STAN_SAMPLE_ADAPT_OPTIONS = sort_dict({"engaged": 1,       # 0, 1
                                       "gamma": 0.05,      # gamma > 0
                                       "delta": 0.8,       # 1 > delta > 0
                                       "kappa": 0.75,      # kappa > 0
                                       "t0": 10,           # t0 > 0
                                       "init_buffer": 75,  # int > 0
                                       "term_buffer": 50,  # int > 0
                                       "window": 25})      # int > 0

STAN_SAMPLE_OPTIONS = sort_dict({"num_samples": 1000,  # num_samples >= 0
                                 "num_warmup": 1000,   # warmup >= 0
                                 "save_warmup": 0,     # 0, 1
                                 "thin": 1,            # thin > 0
                                 "algorithm": "hmc"})  # "hmc", "fixed_param"

STAN_BFGS_OPTIONS = sort_dict({"init_alpha": 0.001,      # init_alpha >= 0
                               "tol_obj": 10 ** -12,     # tol_obj >= 0
                               "tol_rel_obj": 10 ** 4,   # tol_rel_obj >= 0
                               "tol_grad": 10 ** -8,     # tol_grad >= 0
                               "tol_rel_grad": 10 ** 7,  # tol_rel_grad >= 0
                               "tol_param": 10 ** -8,    # tol_param >= 0
                               "history_size": 5})       # int > 0

STAN_OPTIMIZE_OPTIONS = sort_dict({"algorithm": "lbfgs",   # others: "bfgs", "newton"
                                   "iter": 2000,           # int > 0
                                   "save_iterations": 0})  # 0, 1

STAN_VARIATIONAL_ADAPT_OPTIONS = sort_dict({"engaged": 1,             # 0, 1
                                            "iter": 50,               # int > 0
                                            "tol_rel_obj": 0.01,      # tol_rel_obj >= 0
                                            "eval_elbo": 100,         # int > 0
                                            "output_samples": 1000})  # int > 0

STAN_VARIATIONAL_OPTIONS = sort_dict({"algorithm": "meanfield",  # or "fullrank"
                                      "iter": 10000,             # int > 0
                                      "grad_samples": 1,         # int > 0
                                      "elbo_samples": 100,       # int > 0
                                      "eta": 1.0})               # eta > 0

STAN_DIAGNOSE_TEST_GRADIENT_OPTIONS = sort_dict({"epsilon": 10 ** -6,  # epsilon > 0
                                                 "error": 10 ** -6})   # error > 0

STAN_OUTPUT_OPTIONS = sort_dict({"file": "output.csv",
                                 "diagnostic_file": "diagnostic.csv",
                                 "refresh": 100})                    # int >0


def generate_cmdstan_options(method, **kwargs):
    options = OrderedDict()
    if isequal_string(method, "sample"):  # for sample or sampling
        for option, value in STAN_SAMPLE_OPTIONS.iteritems():
            options.update({option: kwargs.pop(option, value)})
        if isequal_string(options["algorithm"], "hmc"):
            for option, value in STAN_HMC_OPTIONS.iteritems():
                options.update({option: kwargs.pop(option, value)})
            if isequal_string(options["engine"], "nuts"):
                options.update({"max_depth": kwargs.pop("max_depth", STAN_NUTS_OPTIONS["max_depth"])})
            elif isequal_string(options["engine"], "static"):
                options.update({"int_time": kwargs.pop("int_time", STAN_STATIC_OPTIONS["int_time"])})
        for option, value in STAN_SAMPLE_ADAPT_OPTIONS.iteritems():
            options.update({option: kwargs.pop(option, value)})
    elif isequal_string(method, "variational"):  # for variational or vb or advi
        for option, value in STAN_VARIATIONAL_OPTIONS.iteritems():
            options.update({option: kwargs.pop(option, value)})
        for option, value in STAN_VARIATIONAL_ADAPT_OPTIONS.iteritems():
            options.update({option: kwargs.pop(option, value)})
    elif isequal_string(method, "optimize"):  # for optimization or optimizing or optimize
        for option, value in STAN_OPTIMIZE_OPTIONS.iteritems():
            options.update({option: kwargs.pop(option, value)})
        if (options["algorithm"].find("bfgs") >= 0):
            for option, value in STAN_BFGS_OPTIONS.iteritems():
                options.update({option: kwargs.pop(option, value)})
    elif isequal_string(method, "diagnose"):  # for diagnose or diagnosing
        for option, value in STAN_DIAGNOSE_TEST_GRADIENT_OPTIONS.iteritems():
            options.update({option: kwargs.pop(option, value)})
    for option, value in STAN_OUTPUT_OPTIONS.iteritems():
        options.update({option: kwargs.pop(option, value)})
    options.update({"init": kwargs.get("init", "random")})
    options.update({"random_seed": kwargs.get("random_seed", 12345)})
    options.update({"random_seed": kwargs.get("seed", options["random_seed"])})
    options.update({"refresh": kwargs.get("refresh", 100)})
    options.update({"chains": kwargs.get("chains", 4)})
    return options

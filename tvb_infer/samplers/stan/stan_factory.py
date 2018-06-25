import os
from collections import OrderedDict
import numpy as np
from tvb_infer.base.utils.data_structures_utils import sort_dict, isequal_string

STAN_STATIC_OPTIONS = sort_dict({"int_time": 2 * np.pi})  # int_time > 0

STAN_NUTS_OPTIONS = sort_dict({"max_depth": 10})  # int > 0

STAN_HMC_OPTIONS = sort_dict({"metric": "diag_e",     # others: "unit_e", "dense_e"
                              "stepsize": 1,          # stepsize > 0
                              "stepsize_jitter": 0})  # 0 <= stepsize_jitter <= 1
STAN_HMC_OPTIONS.update({"engine": "nuts"}) # "static"and

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
                                 "thin": 1})   # thin > 0
STAN_SAMPLE_OPTIONS.update({"algorithm": "hmc"})   # "hmc", "fixed_param"

STAN_BFGS_OPTIONS = sort_dict({"init_alpha": 0.001,      # init_alpha >= 0
                               "tol_obj": 10 ** -12,     # tol_obj >= 0
                               "tol_rel_obj": 10 ** 4,   # tol_rel_obj >= 0
                               "tol_grad": 10 ** -8,     # tol_grad >= 0
                               "tol_rel_grad": 10 ** 7,  # tol_rel_grad >= 0
                               "tol_param": 10 ** -8,    # tol_param >= 0
                               "history_size": 5})       # int > 0

STAN_OPTIMIZE_OPTIONS = sort_dict({"iter": 2000,           # int > 0
                                   "save_iterations": 0})  # 0, 1
STAN_OPTIMIZE_OPTIONS.update({"algorithm": "lbfgs"})   # others: "bfgs", "newton"

STAN_VARIATIONAL_ADAPT_OPTIONS = sort_dict({"engaged": 1,             # 0, 1
                                            "adapt_iter": 50,               # int > 0
                                            "tol_rel_obj": 0.01,      # tol_rel_obj >= 0
                                            "eval_elbo": 100,         # int > 0
                                            "output_samples": 1000})  # int > 0

STAN_VARIATIONAL_OPTIONS = sort_dict({"iter": 10000,             # int > 0
                                      "grad_samples": 1,         # int > 0
                                      "elbo_samples": 100,       # int > 0
                                      "eta": 1.0})               # eta > 0
STAN_VARIATIONAL_OPTIONS.update({"algorithm": "meanfield"})    # or "fullrank"

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
    options.update({"init": kwargs.get("init", 2)})
    options.update({"random_seed": kwargs.get("random_seed", 12345)})
    options.update({"random_seed": kwargs.get("seed", options["random_seed"])})
    options.update({"n_chains_or_runs": kwargs.get("n_chains_or_runs", 4)})
    return options


def generate_cmdstan_fit_command(fitmethod, options, model_path, model_data_path, output_filepath, diagnostic_filepath,
                                 command_path=None):
    command = model_path
    if isequal_string(fitmethod, "sample"):
        command += " method=sample"' \\' + "\n"
        for option in STAN_SAMPLE_OPTIONS.keys():
            command += "\t\t" + option + "=" + str(options[option]) + ' \\' + "\n"
        if isequal_string(options["algorithm"], "hmc"):
            for option in STAN_HMC_OPTIONS.keys():
                command += "\t\t\t\t" + option + "=" + str(options[option]) + ' \\' + "\n"
            if isequal_string(options["engine"], "nuts"):
                command += "\t\t\t\t\t\tmax_depth=" + str(options["max_depth"]) + ' \\' + "\n"
            elif isequal_string(options["engine"], "static"):
                command += "\t\t\t\t\t\tint_time=" + str(options["int_time"]) + ' \\' + "\n"
    elif isequal_string(fitmethod, "variational"):
        command += " method=variational"' \\' + "\n"
        for option in STAN_VARIATIONAL_OPTIONS.keys():
            # due to sort_dict, we know that algorithm is the first option
            command += "\t\t\t\t" + option + "=" + str(options[option]) + ' \\' + "\n"
    elif isequal_string(fitmethod, "optimize"):
        command += " method=optimize"' \\' + "\n"
        for option in STAN_OPTIMIZE_OPTIONS.keys():
            command += "\t\t" + option + "=" + str(options[option]) + ' \\' + "\n"
        if (options["algorithm"].find("bfgs") >= 0):
            for option in STAN_BFGS_OPTIONS.keys():
                command += "\t\t\t\t" + option + "=" + str(options[option]) + ' \\' + "\n"
    # + " data file=" + model_data_path
    elif isequal_string(fitmethod, "diagnose"):
        command += " method=diagnose"' \\' + "\n" + "\t\ttest=gradient "
        for option in STAN_DIAGNOSE_TEST_GRADIENT_OPTIONS.keys():
            command += "\t\t\t\t" + + option + "=" + str(options[option]) + ' \\' + "\n"
    if isequal_string(fitmethod, "sample") or isequal_string(fitmethod, "variational"):
        command += "\t\tadapt"' \\' + "\n"
        if isequal_string(fitmethod, "sample"):
            adapt_options = STAN_SAMPLE_ADAPT_OPTIONS
        else:
            adapt_options = STAN_VARIATIONAL_ADAPT_OPTIONS
        for option in adapt_options.keys():
            if option.find("iter") < 0:
                command += "\t\t\t\t" + option + "=" + str(options[option]) + ' \\' + "\n"
            else:
                command += "\t\t\t\t" + "iter=" + str(options[option]) + ' \\' + "\n"
    command += "\t\tdata file=" + model_data_path + ' \\' + "\n"
    command += "\t\tinit=" + str(options["init"]) + ' \\' + "\n"
    command += "\t\trandom seed=" + str(options["random_seed"]) + ' \\' + "\n"
    if diagnostic_filepath == "":
        diagnostic_filepath = os.path.join(os.path.dirname(output_filepath), STAN_OUTPUT_OPTIONS["diagnostic_file"])
    if options["n_chains_or_runs"] > 1:
        command = ("for i in {1.." + str(options["n_chains_or_runs"]) + "}\ndo\n" +
                   "\t" + command +
                   "\t\tid=$i" + ' \\' + "\n" +
                   "\t\toutput file=" + output_filepath[:-4] + "$i.csv"' \\' + "\n" +
                   "\t\tdiagnostic_file=" + diagnostic_filepath[:-4] + "$i.csv"' \\' + "\n" +
                   "\t\trefresh=" + str(options["refresh"]) + " &" + "\n" +
                   "done")
    else:
        command += "\t\toutput file=" + output_filepath + ' \\' + "\n"
        command += "\t\tdiagnostic_file=" + diagnostic_filepath + ' \\' + "\n"
        command += "\t\trefresh=" + str(options["refresh"])
    command = "chmod +x " + model_path + "\n" + command
    command = ''.join(command)
    if isinstance(command_path, basestring):
        command_path = os.path.join(os.path.dirname(output_filepath), "command.sh")
        command_file = open(command_path, "w")
        command_file.write("#!/bin/bash\n" + command.replace("\t", ""))
        command_file.close()
    return command, output_filepath, diagnostic_filepath

"""
Mechanism for parameter search exploration for LSA and simulations (it will have TVB or custom implementations)
"""

import subprocess
import warnings
import numpy
from tvb_epilepsy.base.constants import AVAILABLE_SIMULATORS
from tvb_epilepsy.base.hypothesis import Hypothesis
from tvb_epilepsy.tvb_api.simulator_tvb import SimulatorTVB
from tvb_epilepsy.custom.simulator_custom import SimulatorCustom


def hypo_out_fun(hypothesis, **kwargs):
    return hypothesis.lsa_ps


def hypo_run_fun(hypothesis, param_names, param_values, param_indexes, out_fun=hypo_out_fun, iE=[], E=[], ix0=[], x0=[],
                 seizure_indices=[]):

    try:

        for ip in range(len(param_names)):

            if param_names[ip] is "E":
                iE.append(param_indexes[ip])
                E.append(param_values[ip])

            elif param_names[ip] is "x0":
                ix0.append(param_indexes[ip])
                x0.append(param_values[ip])

            else:
                temp = getattr(hypothesis, param_names[ip])
                temp[param_indexes[ip]] = param_values[ip]
                setattr(hypothesis, param_names[ip], temp)

        hypothesis.configure_hypothesis(iE, E, ix0, x0, seizure_indices)

        output = out_fun(hypothesis)

        return True, output

    except:

        return False, None


def sim_run_fun(simulator, param_names, param_values, param_indexes, out_fun, **kwargs):

    try:

        for ip in range(len(param_names)):
            pass
            # TODO: search for a parameter inside hypothesis and model, and change if it exists
            # if param_names[ip] is "E":
            #     iE.append(param_indexes[ip])
            #     E.append(param_values[ip])
            #
            # elif param_names[ip] is "x0":
            #     ix0.append(param_indexes[ip])
            #     x0.append(param_values[ip])
            #
            # else:
            #     temp = getattr(hypothesis, param_names[ip])
            #     temp[param_indexes[ip]] = param_values[ip]
            #     setattr(hypothesis, param_names[ip], temp)

        # TODO: reconfigure hypothesis
        # hypothesis.configure_hypothesis(iE, E, ix0, x0, seizure_indices)

        _, _, status = simulator.launch()
        #output = out_fun(hypothesis)

        #return True, output
        pass

    except:

        return False, None


def sim_out_fun(simulator, **kwargs):
    pass


class PSE(object):

    def __init__(self, task, hypothesis=None, simulator=None, params_pse=None, run_fun=None, out_fun=None,
                 grid_mode=True):

        if task not in ["LSA", "SIMULATION"]:
            raise ValueError("\ntask = " + str(task) + " is not a valid pse task." +
                             "\nSelect one of 'LSA', or 'SIMULATION' to perform parameter search exploration of " +
                             "\n hypothesis Linear Stability Analysis, or simulation, " + "respectively")

        self.task = task

        if task is "LSA":

            if isinstance(hypothesis, Hypothesis):
                self.pse_object = hypothesis

            else:
                raise ValueError("\ntask = LSA" + str(task) + " but hypothesis is not a Hypothesis object!")

            def_run_fun = hypo_run_fun
            def_out_fun = hypo_out_fun

        else:

            if isinstance(simulator, AVAILABLE_SIMULATORS):
                self.pse_object = simulator

            else:
                raise ValueError("\ntask = 'SIMULATION'" + str(task) + " but simulator is not an object of" +
                                 " one of the available simulator classes!")

            def_run_fun = hypo_run_fun
            def_out_fun = hypo_out_fun

        if not (callable(run_fun)):
            warnings.warn("\nUser defined run_fun is not callable. Using default one for task " + str(task) +"!")
            self.run_fun = def_run_fun
        else:
            self.run_fun = run_fun

        if not (callable(out_fun)):
            warnings.warn("\nUser defined out_fun is not callable. Using default one for task " + str(task) +"!")
            self.out_fun = def_out_fun
        else:
            self.out_fun = out_fun

        if isinstance(params_pse, dict):

            self.pse_name_list = []
            self.n_params_vals = []
            self.params_indexes = []
            temp = []
            for key, value in params_pse.iteritems():
                temp.append(value[0])
                self.n_params_vals.append(len(value[0]))
                self.params_indexes.append(value[1])
                self.params_names_list.append(key)

            self.n_params_vals = numpy.array(self.n_params_vals)
            self.n_params = len(self.params_names_list)

            if grid_mode:
                temp = list(numpy.meshgrid(tuple(temp), "indexing", "ij"))
                for ip in range(self.n_params):
                    temp[ip] = numpy.flatten(temp[ip])

            else:
                if not(numpy.all(self.params_names_list == self.params_names_list[0])):
                    raise ValueError("\ngrid_mode = False but not all parameters have the same number of values!: " +
                                     "\n" + str(self.params_names_list) + " = " + str( self.n_params_vals))

            self.pse_params = numpy.vstack(temp)
            self.n_loops = self.pse_params.shape[0]

            print "\nGenerated a parameter search exploration for " + str(task) + ","
            print "with " + str(self.n_params) + " parameters of " + str(self.n_params_vals) + " each,"
            print "leading to " + str(self.n_loops) + " total execution loops"
            if grid_mode:
                print "in grid mode"

        else:
            raise ValueError("\nparams_pse is not a dictionary!")

    def run_pse(self, grid_mode=False, **kwargs):

        results = []
        execution_status = []

        for iloop in range(self.n_loops):

            params = self.pse_params[iloop, :].tolist()

            print "\nExecuting loop " + str(iloop) + "of " + str(self.n_loops)
            print "Parameters " + str(self.params_names_list) + " = " + str(params)

            status = False
            output = None

            try:
                status, output = self.run_fun(self.pse_object, self.params_names_list, params, self.param_indexes,
                                              self.out_fun, kwargs)

            except:
                pass

            if not status:
                warnings.warn("\nExecution of loop " + str(iloop) + "failed!")

            results.append(output)
            execution_status.append(status)

        if grid_mode:
            results = numpy.reshape(numpy.array(results, dtype="O"), tuple(self.n_params_vals))
            execution_status = numpy.reshape(numpy.array(execution_status), tuple(self.n_params_vals))

        return results, execution_status

    def run_pse_parallel(self, grid_mode=False):
        # TODO: start each loop on a separate process, gather results and return them
        raise NotImplementedError
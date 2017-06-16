"""
Mechanism for parameter search exploration for LSA and simulations (it will have TVB or custom implementations)
"""

import subprocess
import warnings
import numpy
from copy import deepcopy
from tvb_epilepsy.base.constants import EIGENVECTORS_NUMBER_SELECTION, K_DEF, YC_DEF, I_EXT1_DEF, A_DEF, B_DEF
from tvb_epilepsy.base.simulators import ABCSimulator
from tvb_epilepsy.base.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.model_configuration import ModelConfiguration
from tvb_epilepsy.base.model_configuration_service import ModelConfigurationService
from tvb_epilepsy.base.lsa_service import LSAService
from tvb_epilepsy.base.epileptor_model_factory import model_build_dict
from tvb_epilepsy.tvb_api.simulator_tvb import SimulatorTVB
from tvb_epilepsy.custom.simulator_custom import SimulatorCustom, custom_model_builder
from tvb_epilepsy.custom.read_write import read_ts


def set_object_attribute_recursively(object, name, values, indexes):

    # Convert the parameter's name to a list of strings separated by "."
    name = name.split(".")

    # If there is more than one levels...
    if len(name) > 1:
        #...call the function recursively
        set_object_attribute_recursively(getattr(object, name[0]), ".".join(name[1:]), values, indexes)

    else:
        # ...else, set the parameter values for the specified indexes
        temp = getattr(object, name[0])
        if len(indexes) > 0:
            temp[indexes] = values #index has to be linear... i.e., 1D...
        else:
            temp = values
        setattr(object, name[0], temp)


def pop_object_parameters(object_type, param_names, param_values, param_indexes):

    object_param_names = []
    object_param_values = []
    object_param_indexes = []

    for ip in range(len(param_names)):

        if param_names[ip].split(".")[0] == object_type:
            object_param_names.append(param_names.pop(ip).split(".")[1])
            object_param_values.append(param_values.pop(ip))
            object_param_indexes.append(param_indexes.pop(ip))

    return object_param_names, object_param_values, object_param_indexes, param_names, param_values, param_indexes


def update_object(object, object_type, param_names, param_values, param_indexes):
    update_flag = False
    object_param_names, object_param_values, object_param_indexes, param_names, param_values, param_indexes = \
        pop_object_parameters(object_type, param_names, param_values, param_indexes)
    for ip in range(len(object_param_names)):
        set_object_attribute_recursively(object, object_param_names[ip], object_param_values[ip],
                                         object_param_indexes[ip])
        update_flag = True
    return object, param_names, param_values, param_indexes, update_flag


def update_hypothesis(hypothesis_input, param_names, param_values, param_indexes,
                      model_configuration_service_input=None,
                      yc=YC_DEF, Iext1=I_EXT1_DEF, K=K_DEF, a=A_DEF, b=B_DEF, x1eq_mode="optimize"):

    # Assign possible hypothesis parameters on a new hypothesis object:
    hypothesis = deepcopy(hypothesis_input)
    hypothesis, param_names, param_values, param_indexes = \
        update_object(hypothesis, "hypothesis", param_names, param_values, param_indexes)[:4]

    # ...create/update a model configuration service:
    if isinstance(model_configuration_service_input, ModelConfigurationService):
        model_configuration_service = deepcopy(model_configuration_service_input)
    else:
        model_configuration_service = ModelConfigurationService(yc=yc, Iext1=Iext1, K=K, a=a, b=b, x1eq_mode=x1eq_mode)

    # ...modify possible related parameters:
    model_configuration_service, param_names, param_values, param_indexes = \
        update_object(model_configuration_service, "model_configuration_service", param_names, param_values,
                      param_indexes)[:4]

    # ...and compute a new model_configuration:
    if hypothesis.type == "Epileptogenicity":
        model_configuration = model_configuration_service.configure_model_from_E_hypothesis(hypothesis)
    else:
        model_configuration = model_configuration_service.configure_model_from_hypothesis(hypothesis)

    return hypothesis, model_configuration, param_names, param_values, param_indexes


def lsa_out_fun(hypothesis, **kwargs):
    return hypothesis.lsa_ps


def lsa_run_fun(hypothesis_input, param_names, param_values, param_indexes, out_fun=lsa_out_fun,
                model_configuration_service_input=None,
                yc=YC_DEF, Iext1=I_EXT1_DEF, K=K_DEF, a=A_DEF, b=B_DEF, x1eq_mode="optimize",
                lsa_service_input=None,
                n_eigenvectors=EIGENVECTORS_NUMBER_SELECTION, weighted_eigenvector_sum=True):

    try:
        # Update hypothesis and create a new model_configuration:
        hypothesis, model_configuration, param_names, param_values, param_indexes\
            = update_hypothesis(hypothesis_input, param_names, param_values, param_indexes,
                                model_configuration_service_input, yc, Iext1, K, a, b, x1eq_mode)

        # ...create/update lsa service:
        if isinstance(lsa_service_input, LSAService):
            lsa_service = deepcopy(lsa_service_input)
        else:
            lsa_service = LSAService(n_eigenvectors=n_eigenvectors, weighted_eigenvector_sum=weighted_eigenvector_sum)

        # ...and modify possible related parameters:
        lsa_service = \
            update_object(lsa_service, "lsa_service", param_names, param_values, param_indexes)[0]

        # Run LSA:
        lsa_hypothesis = lsa_service.run_lsa(hypothesis, model_configuration)

        if callable(hypothesis):
            output = out_fun(lsa_hypothesis)
        else:
            output = lsa_hypothesis

        return True, output

    except:

        return False, None


def sim_out_fun(simulator, time, data, **kwargs):

    if data is None:
        time, data = read_ts(simulator.results_path, data="data")

    return {"time": time, "data": data}


def sim_run_fun(simulator_input, param_names, param_values, param_indexes, out_fun=sim_out_fun, hypothesis_input=None,
                model_configuration_service_input=None,
                yc=YC_DEF, Iext1=I_EXT1_DEF, K=K_DEF, a=A_DEF, b=B_DEF, x1eq_mode="optimize",
                update_initial_conditions=True):

    # Create new objects from the input simulator
    simulator = deepcopy(simulator_input)
    model_configuration = deepcopy(simulator_input.model_configuration)
    model = deepcopy(simulator_input.model)

    try:

        # First try to update model_configuration via an input hypothesis...:
        if isinstance(hypothesis_input, DiseaseHypothesis):
            hypothesis, model_configuration, param_names, param_values, param_indexes = \
                update_hypothesis(hypothesis_input, param_names, param_values, param_indexes,
                                  model_configuration_service_input, yc, Iext1, K, a, b, x1eq_mode)
            # Update model configuration:
            simulator.model_configuration = model_configuration
            # ...in which case a model has to be regenerated:
            if isinstance(simulator, SimulatorTVB):
                model = model_build_dict[model._ui_name](model_configuration, zmode=model.zmode)
            else:
                model = custom_model_builder(model_configuration)

        # Now (further) update model if needed:
        model, param_names, param_values, param_indexes = \
            update_object(model, "model", param_names, param_values, param_indexes)
        simulator.model = model

        # Now, update other possible remaining parameters, i.e., concerning the integrator, noise etc...
        for ip in range(len(param_names)):
            set_object_attribute_recursively(simulator, param_names[ip], param_values[ip], param_indexes[ip])

        # Now, recalculate the default initial conditions...
        # If initial conditions were parameters, then, this flag can be set to False
        if update_initial_conditions:
            simulator.configure_initial_conditions()

        time, data, status = simulator.launch()

        if status:
            output = out_fun(time, data, simulator)

        return True, output

    except:

        return False, None


class PSE(object):

    def __init__(self, task, hypothesis=None, simulator=None, params_pse=None, run_fun=None, out_fun=None,
                 grid_mode=True):

        if task not in ["LSA", "SIMULATION"]:
            raise ValueError("\ntask = " + str(task) + " is not a valid pse task." +
                             "\nSelect one of 'LSA', or 'SIMULATION' to perform parameter search exploration of " +
                             "\n hypothesis Linear Stability Analysis, or simulation, " + "respectively")

        self.task = task

        if task is "LSA":

            # TODO: this will not work anymore
            if isinstance(hypothesis, DiseaseHypothesis):
                self.pse_object = hypothesis

            else:
                raise ValueError("\ntask = LSA" + str(task) + " but hypothesis is not a Hypothesis object!")

            def_run_fun = lsa_run_fun
            def_out_fun = lsa_out_fun

        else:

            if isinstance(simulator, ABCSimulator):
                self.pse_object = simulator

            else:
                raise ValueError("\ntask = 'SIMULATION'" + str(task) + " but simulator is not an object of" +
                                 " one of the available simulator classes!")

            def_run_fun = sim_run_fun
            def_out_fun = sim_out_fun

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
                if not(numpy.all(self.n_params_vals == self.n_params_vals[0])):
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
                                              self.out_fun, **kwargs)

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
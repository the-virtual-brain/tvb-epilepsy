"""
Mechanism for parameter search exploration for LSA and simulations (it will have TVB or custom implementations)
"""

from copy import deepcopy
import numpy as np
from tvb_epilepsy.base.constants.module_constants import EIGENVECTORS_NUMBER_SELECTION
from tvb_epilepsy.base.constants.model_constants import K_DEF, YC_DEF, I_EXT1_DEF, A_DEF, B_DEF
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning, raise_value_error, \
    raise_not_implemented_error
from tvb_epilepsy.base.utils.data_structures_utils import formal_repr
from tvb_epilepsy.base.model.disease_hypothesis import DiseaseHypothesis
from tvb_epilepsy.base.model.model_configuration import ModelConfiguration
from tvb_epilepsy.base.simulators import ABCSimulator
from tvb_epilepsy.service.epileptor_model_factory import model_build_dict
from tvb_epilepsy.service.model_configuration_service import ModelConfigurationService
from tvb_epilepsy.service.lsa_service import LSAService
from tvb_epilepsy.tvb_api.simulator_tvb import SimulatorTVB
from tvb_epilepsy.custom.simulator_custom import custom_model_builder
from tvb_epilepsy.io.h5_reader import H5Reader

logger = initialize_logger(__name__)


def set_object_attribute_recursively(object, path, values, indices):
    # Convert the parameter's path to a list of strings separated by "."
    path = path.split(".")
    # If there is more than one levels...
    if len(path) > 1:
        # ...call the function recursively
        set_object_attribute_recursively(getattr(object, path[0]), ".".join(path[1:]), values, indices)
    else:
        # ...else, set the parameter values for the specified indices
        temp = getattr(object, path[0])
        if len(indices) > 0:
            temp[indices] = values  # index has to be linear... i.e., 1D...
        else:
            temp = values
        setattr(object, path[0], temp)


def pop_object_parameters(object_type, params_paths, params_values, params_indices):
    object_params_paths = []
    object_params_values = []
    object_params_indices = []
    items_to_delete = []
    for ip in range(len(params_paths)):
        if params_paths[ip].split(".")[0] == object_type:
            object_params_paths.append(params_paths[ip].split(".")[1])
            object_params_values.append(params_values[ip])
            object_params_indices.append(params_indices[ip])
            items_to_delete.append(ip)
    params_paths = np.delete(params_paths, items_to_delete)
    params_values = np.delete(params_values, items_to_delete)
    params_indices = np.delete(params_indices, items_to_delete)
    return object_params_paths, object_params_values, object_params_indices, params_paths, params_values, params_indices


def update_object(object, object_type, params_paths, params_values, params_indices):
    update_flag = False
    object_params_paths, object_params_values, object_params_indices, params_paths, params_values, params_indices = \
        pop_object_parameters(object_type, params_paths, params_values, params_indices)
    for ip in range(len(object_params_paths)):
        set_object_attribute_recursively(object, object_params_paths[ip], object_params_values[ip],
                                         object_params_indices[ip])
        update_flag = True
    return object, params_paths, params_values, params_indices, update_flag


def update_hypothesis(hypothesis_input, model_connectivity, params_paths, params_values, params_indices,
                      model_configuration_service_input=None,
                      yc=YC_DEF, Iext1=I_EXT1_DEF, K=K_DEF, a=A_DEF, b=B_DEF, x1eq_mode="optimize"):
    # Assign possible hypothesis parameters on a new hypothesis object:
    hypothesis = deepcopy(hypothesis_input)
    hypothesis, params_paths, params_values, params_indices = \
        update_object(hypothesis, "hypothesis", params_paths, params_values, params_indices)[:4]
    hypothesis.update(name=hypothesis.name)
    # ...create/update a model configuration service:
    if isinstance(model_configuration_service_input, ModelConfigurationService):
        model_configuration_service = deepcopy(model_configuration_service_input)
    else:
        model_configuration_service = ModelConfigurationService(hypothesis_input.number_of_regions,
                                                                yc=yc, Iext1=Iext1, K=K, a=a, b=b, x1eq_mode=x1eq_mode)
    # ...modify possible related parameters:
    model_configuration_service, params_paths, params_values, params_indices = \
        update_object(model_configuration_service, "model_configuration_service", params_paths, params_values,
                      params_indices)[:4]
    # ...and compute a new model_configuration:
    if hypothesis.type == "Epileptogenicity":
        model_configuration = model_configuration_service.configure_model_from_E_hypothesis(hypothesis,
                                                                                            model_connectivity)
    else:
        model_configuration = model_configuration_service.configure_model_from_hypothesis(hypothesis,
                                                                                          model_connectivity)
    return hypothesis, model_configuration, params_paths, params_values, params_indices


def lsa_out_fun(hypothesis, model_configuration=None, **kwargs):
    if isinstance(model_configuration, ModelConfiguration):
        return {"lsa_propagation_strengths": hypothesis.lsa_propagation_strengths,
                "x0_values": model_configuration.x0_values,
                "e_values": model_configuration.e_values, "x1EQ": model_configuration.x1EQ,
                "zEQ": model_configuration.zEQ, "Ceq": model_configuration.Ceq}
    else:
        hypothesis.lsa_propagation_strengths


def lsa_run_fun(hypothesis_input, model_connectivity, params_paths, params_values, params_indices, out_fun=lsa_out_fun,
                model_configuration_service_input=None,
                yc=YC_DEF, Iext1=I_EXT1_DEF, K=K_DEF, a=A_DEF, b=B_DEF, x1eq_mode="optimize",
                lsa_service_input=None,
                n_eigenvectors=EIGENVECTORS_NUMBER_SELECTION, weighted_eigenvector_sum=True):
    try:
        # Update hypothesis and create a new model_configuration:
        hypothesis, model_configuration, params_paths, params_values, params_indices \
            = update_hypothesis(hypothesis_input, model_connectivity, params_paths, params_values, params_indices,
                                model_configuration_service_input, yc, Iext1, K, a, b, x1eq_mode)
        # ...create/update lsa service:
        if isinstance(lsa_service_input, LSAService):
            lsa_service = deepcopy(lsa_service_input)
        else:
            lsa_service = LSAService(n_eigenvectors=n_eigenvectors, weighted_eigenvector_sum=weighted_eigenvector_sum)
        # ...and modify possible related parameters:
        lsa_service = \
            update_object(lsa_service, "lsa_service", params_paths, params_values, params_indices)[0]
        # Run LSA:
        lsa_hypothesis = lsa_service.run_lsa(hypothesis, model_configuration)
        if callable(out_fun):
            output = out_fun(lsa_hypothesis, model_configuration=model_configuration)
        else:
            output = lsa_hypothesis
        return True, output
    except:
        return False, None


def sim_out_fun(simulator, time, data, **kwargs):
    if data is None:
        time, data = H5Reader().read_timeseries(simulator.results_path, data="data")
    return {"time": time, "data": data}


def sim_run_fun(simulator_input, model_connectivity, params_paths, params_values, params_indices, out_fun=sim_out_fun,
                hypothesis_input=None,
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
            hypothesis, model_configuration, params_paths, params_values, params_indices = \
                update_hypothesis(hypothesis_input, model_connectivity, params_paths, params_values, params_indices,
                                  model_configuration_service_input, yc, Iext1, K, a, b, x1eq_mode)
            # Update model configuration:
            simulator.model_configuration = model_configuration
            # ...in which case a model has to be regenerated:
            if isinstance(simulator, SimulatorTVB):
                model = model_build_dict[model._ui_name](model_configuration, zmode=model.zmode)
            else:
                model = custom_model_builder(model_configuration)
            simulator.model = model
        # Now (further) update model if needed:
        model, params_paths, params_values, params_indices = \
            update_object(model, "model", params_paths, params_values, params_indices)
        # Now, update other possible remaining parameters, i.e., concerning the integrator, noise etc...
        for ip in range(len(params_paths)):
            set_object_attribute_recursively(simulator, params_paths[ip], params_values[ip], params_indices[ip])
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

#TODO: this is deprecated. We are waiting for Denis to verify the service/pse.
class PSEService(object):

    def __init__(self, task, hypothesis=[], simulator=[], params_pse=None, run_fun=None, out_fun=None):
        if task not in ["LSA", "SIMULATION"]:
            warning("\ntask = " + str(task) + " is not a valid pse task." +
                    "\nSelect one of 'LSA', or 'SIMULATION' to perform parameter search exploration of " +
                    "\n hypothesis Linear Stability Analysis, or simulation, " + "respectively")
        self.task = task
        self.params_names = []
        self.params_paths = []
        self.n_params_vals = []
        self.params_indices = []
        self.n_loops = 0
        if task == "LSA":
            # TODO: this will not work anymore
            if isinstance(hypothesis, DiseaseHypothesis):
                self.pse_object = hypothesis
            else:
                warning("\ntask = " + str(task) + " but hypothesis is not a Hypothesis object!")
            def_run_fun = lsa_run_fun
            def_out_fun = lsa_out_fun
        elif task == "SIMULATION":
            if isinstance(simulator, ABCSimulator):
                self.pse_object = simulator
            else:
                warning("\ntask = " + str(task) + " but simulator is not an object of" +
                        " one of the available simulator classes!")
            def_run_fun = sim_run_fun
            def_out_fun = sim_out_fun
        else:
            self.pse_object = []
            def_run_fun = None
            def_out_fun = None
        if not (callable(run_fun)):
            warning("\nUser defined run_fun is not callable. Using default one for task " + str(task) + "!")
            self.run_fun = def_run_fun
        else:
            self.run_fun = run_fun
        if not (callable(out_fun)):
            warning("\nUser defined out_fun is not callable. Using default one for task " + str(task) + "!")
            self.out_fun = def_out_fun
        else:
            self.out_fun = out_fun
        if isinstance(params_pse, list):
            temp = []
            for param in params_pse:
                # parameter path:
                self.params_paths.append(param["path"])
                # parameter values/samples:
                temp2 = param["samples"].flatten()
                temp.append(temp2)
                self.n_params_vals.append(temp2.size)
                # parameter indices:
                indices = param.get("indices", [])
                self.params_indices.append(indices)
                self.params_names.append(param.get("name", param["path"].rsplit('.', 1)[-1] + str(indices)))
            self.n_params_vals = np.array(self.n_params_vals)
            self.n_params = len(self.params_paths)
            if not (np.all(self.n_params_vals == self.n_params_vals[0])):
                raise_value_error("\nNot all parameters have the same number of samples!: " +
                                  "\n" + str(self.params_paths) + " = " + str(self.n_params_vals))
            else:
                self.n_params_vals = self.n_params_vals[0]
            self.pse_params = np.vstack(temp).T
            self.params_paths = np.array(self.params_paths)
            self.params_indices = np.array(self.params_indices)
            self.n_loops = self.pse_params.shape[0]
            print "\nGenerated a parameter search exploration for " + str(task) + ","
            print "with " + str(self.n_params) + " parameters of " + str(self.n_params_vals) + " values each,"
            print "leading to " + str(self.n_loops) + " total execution loops"
        else:
            warning("\nparams_pse is not a list of tuples!")

    def __repr__(self):
        d = {"01. Task": self.task,
             "02. Main PSE object": self.pse_object,
             "03. Number of computation loops": self.n_loops,
             "04. Parameters": np.array(["%s" % l for l in self.params_names]),
             }
        return formal_repr(self, d)

    def __str__(self):
        return self.__repr__()

    def run_pse(self, model_connectivity, grid_mode=False, **kwargs):
        results = []
        execution_status = []
        loop_tenth = 1
        for iloop in range(self.n_loops):
            params = self.pse_params[iloop, :]
            if iloop == 0 or iloop + 1 >= loop_tenth * self.n_loops / 10.0:
                print "\nExecuting loop " + str(iloop + 1) + " of " + str(self.n_loops)
                if iloop > 0:
                    loop_tenth += 1
            # print "\nParameters:"
            # for ii in range(len(pdf_params)):
            #      print self.params_paths[ii] + "[" + str(self.params_indices[ii]) + "] = " + str(pdf_params[ii])
            status = False
            output = None
            try:
                status, output = self.run_fun(self.pse_object, model_connectivity,
                                              self.params_paths, params, self.params_indices, self.out_fun, **kwargs)
            except:
                pass
            if not status:
                warning("\nExecution of loop " + str(iloop) + " failed!")
            results.append(output)
            execution_status.append(status)
        if grid_mode:
            results = np.reshape(np.array(results, dtype="O"), tuple(self.n_params_vals))
            execution_status = np.reshape(np.array(execution_status), tuple(self.n_params_vals))
        return results, execution_status

    def run_pse_parallel(self, model_connectivity, grid_mode=False):
        # TODO: start each loop on a separate process, gather results and return them
        raise_not_implemented_error

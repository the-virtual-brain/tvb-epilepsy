from copy import deepcopy
import numpy as np
from abc import abstractmethod, ABCMeta
from tvb_epilepsy.base.constants.model_constants import K_DEF, YC_DEF, I_EXT1_DEF, A_DEF, B_DEF
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning, raise_value_error
from tvb_epilepsy.service.model_configuration_service import ModelConfigurationService


class ABCPSEService(object):
    __metaclass__ = ABCMeta

    logger = initialize_logger(__name__)

    params_vals = []
    params_paths = []
    params_indices = []
    params_names = []
    n_params_vals = []
    n_params = 0

    def run_pse(self, conn_matrix, grid_mode, **kwargs):
        results = []
        execution_status = []
        loop_tenth = 1
        for iloop in range(self.n_loops):
            params = self.params_vals[iloop]
            if iloop == 0 or iloop + 1 >= loop_tenth * self.n_loops / 10.0:
                print "\nExecuting loop " + str(iloop + 1) + " of " + str(self.n_loops)
                if iloop > 0:
                    loop_tenth += 1

            status = False
            output = None
            try:
                status, output = self.run(conn_matrix, **kwargs)
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

    @abstractmethod
    def run_pse_parallel(self):
        # TODO: start each loop on a separate process, gather results and return them
        pass

    @abstractmethod
    def run(self, *kwargs):
        pass

    @abstractmethod
    def prepare_run_results(self, *kwargs):
        pass

    def prepare_params(self, params_pse):
        if isinstance(params_pse, list):
            temp = []
            for param in params_pse:
                self.params_paths.append(param["path"])
                temp2 = param["samples"].flatten()
                temp.append(temp2)
                self.n_params_vals.append(temp2.size)
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
            self.params_vals = np.vstack(temp).T
            self.params_paths = np.array(self.params_paths)
            self.params_indices = np.array(self.params_indices)
            self.n_loops = self.params_vals.shape[0]
            print "\nGenerated a parameter search exploration for " + str("lsa/sim task") + ","
            print "with " + str(self.n_params) + " parameters of " + str(self.n_params_vals) + " values each,"
            print "leading to " + str(self.n_loops) + " total execution loops"
        else:
            warning("\nparams_pse is not a list of tuples!")

    def update_hypothesis(self, conn_matrix, model_config_service_input=None,
                           yc=YC_DEF, Iext1=I_EXT1_DEF, K=K_DEF, a=A_DEF, b=B_DEF, x1eq_mode="optimize",):
        # Copy and update hypothesis
        hypo_copy = deepcopy(self.hypothesis)
        hypo_copy.update_for_pse(self.params_vals, self.params_paths, self.params_indices)
        # Create a ModelConfigService and update it
        if isinstance(model_config_service_input, ModelConfigurationService):
            model_configuration_service = deepcopy(model_config_service_input)
        else:
            model_configuration_service = ModelConfigurationService(hypo_copy.number_of_regions,
                                                                    yc=yc, Iext1=Iext1, K=K, a=a, b=b,
                                                                    x1eq_mode=x1eq_mode)
        model_configuration_service.update_for_pse(self.params_vals, self.params_paths, self.params_indices)
        # Obtain Modelconfiguration
        if hypo_copy.type == "Epileptogenicity":
            model_configuration = model_configuration_service.configure_model_from_E_hypothesis(hypo_copy,
                                                                                                conn_matrix)
        else:
            model_configuration = model_configuration_service.configure_model_from_hypothesis(hypo_copy,
                                                                                              conn_matrix)
        return hypo_copy, model_configuration

    def set_object_attribute_recursively(self, obj, values):
        path = self.params_paths.split(".")
        # If there is more than one levels, call function recursively
        if len(path) > 1:
            self.set_object_attribute_recursively(getattr(obj, path[0]), ".".join(path[1:]), values)

        else:
            temp = getattr(obj, path[0])
            if len(self.params_indices) > 0:
                temp[self.params_indices] = values
            else:
                temp = values
            setattr(obj, path[0], temp)

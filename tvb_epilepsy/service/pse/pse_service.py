import numpy
from abc import abstractmethod, ABCMeta
from tvb_epilepsy.base.h5_model import convert_to_h5_model
from tvb_epilepsy.base.utils.log_error_utils import initialize_logger, warning, raise_value_error


class ABCPSEService(object):
    __metaclass__ = ABCMeta

    logger = initialize_logger(__name__)

    params_vals = []
    params_paths = []
    params_indices = []
    params_names = []
    n_params_vals = []
    n_params = 0

    def run_pse(self, conn_matrix, grid_mode, *kwargs):
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
                status, output = self.run(conn_matrix, params, *kwargs)

            except:
                pass

            if not status:
                warning("\nExecution of loop " + str(iloop) + " failed!")

            results.append(output)
            execution_status.append(status)

        if grid_mode:
            results = numpy.reshape(numpy.array(results, dtype="O"), tuple(self.n_params_vals))
            execution_status = numpy.reshape(numpy.array(execution_status), tuple(self.n_params_vals))

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

            self.n_params_vals = numpy.array(self.n_params_vals)
            self.n_params = len(self.params_paths)

            if not (numpy.all(self.n_params_vals == self.n_params_vals[0])):
                raise_value_error("\nNot all parameters have the same number of samples!: " +
                                  "\n" + str(self.params_paths) + " = " + str(self.n_params_vals))
            else:
                self.n_params_vals = self.n_params_vals[0]

            self.params_vals = numpy.vstack(temp).T
            self.params_paths = numpy.array(self.params_paths)
            self.params_indices = numpy.array(self.params_indices)
            self.n_loops = self.params_vals.shape[0]

            print "\nGenerated a parameter search exploration for " + str("lsa/sim task") + ","
            print "with " + str(self.n_params) + " parameters of " + str(self.n_params_vals) + " values each,"
            print "leading to " + str(self.n_loops) + " total execution loops"

        else:
            warning("\nparams_pse is not a list of tuples!")

    def _prepare_for_h5(self):
        h5_model = convert_to_h5_model({"task": "LSA", "n_loops": self.n_loops,
                                        "params_names": self.params_names,
                                        "params_paths": self.params_paths,
                                        "params_indices": numpy.array([str(inds) for inds in self.params_indices],
                                                                      dtype="S"),
                                        "params_samples": self.params_vals.T})
        h5_model.add_or_update_metadata_attribute("EPI_Type", "HypothesisModel")
        return h5_model

    def write_to_h5(self, folder, filename=""):
        if filename == "":
            filename = "LSA_PSEService.h5"
        h5_model = self._prepare_for_h5()
        h5_model.write_to_h5(folder, filename)

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
